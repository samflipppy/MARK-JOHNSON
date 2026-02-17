#!/usr/bin/env python3
"""
Full-year bankroll simulation: $1,000 starting → where do you end up?

Simulates 365 trading days with compounding Kelly sizing across three
scenarios: conservative, base case, and optimistic. Each day randomly
draws market conditions from realistic distributions.

This answers: "If I start with $1,000, where will I be at year end?"
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy import stats

from models.forecast import TemperatureDistribution

# ─── Parameters ──────────────────────────────────────────────────────────────

STARTING_BANKROLL = 1000.0
TRADING_DAYS = 365
KALSHI_FEE = 0.01  # per side
BAND_WIDTH = 2
NUM_BANDS_EACH_SIDE = 5
NUM_SIMULATIONS = 100  # Monte Carlo paths

# v2 thresholds
V2_EDGE_CENTER = 0.12
V2_EDGE_SHOULDER = 0.10
V2_EDGE_TAIL = 0.08
V2_EDGE_MEDIUM_CONF = 0.12
V2_EDGE_LOW_CONF = 0.20
V2_NOWCAST_DISCOUNT = 0.75

# Kelly
KELLY_FRACTION = 0.25
KELLY_MAX_PCT = 0.05  # max 5% of bankroll per single bet
MIN_BET = 1.0  # don't bother with bets < $1

EDGE_STRONG = 0.12
EDGE_EXTREME = 0.20

SEED = 42


# ─── Day condition profiles ─────────────────────────────────────────────────
# Estimated from real Kalshi temperature market characteristics

# How often does each market condition occur? (must sum to 1.0)
# These are calibrated to be realistic, not optimistic.
DAY_PROFILES = {
    # (weight, description, params)
    "efficient": {
        "weight": 0.35,  # 35% of days, market is well-priced
        "model_mean_err": (0.0, 0.3),  # (mean, std) of model error
        "model_std_err": (1.0, 0.05),
        "market_noise": (0.02, 0.005),
        "market_bias": "none",
        "market_bias_str": 0.0,
        "stale_shift": 0.0,
        "nowcast": 0.0,
    },
    "slight_model_edge": {
        "weight": 0.20,  # 20% of days, model is slightly better
        "model_mean_err": (0.0, 0.2),
        "model_std_err": (1.0, 0.08),
        "market_noise": (0.03, 0.01),
        "market_bias": "tail_under",
        "market_bias_str": 0.04,
        "stale_shift": 0.0,
        "nowcast": 0.0,
    },
    "nowcast_edge": {
        "weight": 0.10,  # 10% of days, we catch a temperature shift
        "model_mean_err": (0.2, 0.2),
        "model_std_err": (1.0, 0.05),
        "market_noise": (0.02, 0.005),
        "market_bias": "stale",
        "market_bias_str": 0.0,
        "stale_shift": 1.5,  # market is 1.5°F stale
        "nowcast": 1.0,
    },
    "panic_market": {
        "weight": 0.05,  # 5% of days, market overprices tails
        "model_mean_err": (0.0, 0.3),
        "model_std_err": (1.0, 0.05),
        "market_noise": (0.03, 0.01),
        "market_bias": "tail_over",
        "market_bias_str": 0.06,
        "stale_shift": 0.0,
        "nowcast": 0.0,
    },
    "model_worse": {
        "weight": 0.15,  # 15% of days, model is meaningfully wrong
        "model_mean_err": (0.0, 1.2),
        "model_std_err": (1.0, 0.15),
        "market_noise": (0.02, 0.005),
        "market_bias": "none",
        "market_bias_str": 0.0,
        "stale_shift": 0.0,
        "nowcast": 0.0,
    },
    "high_uncertainty": {
        "weight": 0.10,  # 10% of days, frontal/storm days
        "model_mean_err": (0.0, 0.8),
        "model_std_err": (1.1, 0.1),
        "market_noise": (0.04, 0.01),
        "market_bias": "none",
        "market_bias_str": 0.0,
        "stale_shift": 0.0,
        "nowcast": 0.0,
        "true_std_override": 5.0,
    },
    "no_signal": {
        "weight": 0.05,  # 5% of days, no tradeable markets (holidays, low volume)
        "skip": True,
    },
}


# ─── Core functions ──────────────────────────────────────────────────────────

def generate_bands(center):
    bands = []
    low_end = center - NUM_BANDS_EACH_SIDE * BAND_WIDTH
    high_end = center + NUM_BANDS_EACH_SIDE * BAND_WIDTH
    bands.append((None, low_end, f"{low_end:.0f}°F or below"))
    for i in range(NUM_BANDS_EACH_SIDE * 2):
        lo = low_end + i * BAND_WIDTH
        hi = lo + BAND_WIDTH
        bands.append((lo, hi, f"{lo:.0f}°–{hi:.0f}°F"))
    bands.append((high_end, None, f"{high_end:.0f}°F or above"))
    return bands


def compute_probs(bands, mean, std):
    d = stats.norm(loc=mean, scale=std)
    probs = []
    for bmin, bmax, _ in bands:
        if bmin is None and bmax is not None:
            probs.append(float(d.cdf(bmax)))
        elif bmax is None and bmin is not None:
            probs.append(float(1.0 - d.cdf(bmin)))
        elif bmin is not None and bmax is not None:
            probs.append(float(d.cdf(bmax) - d.cdf(bmin)))
        else:
            probs.append(0.0)
    return probs


def add_noise(probs, noise_std, bias_type, bias_str, center_idx=None):
    n = len(probs)
    out = list(probs)
    if center_idx is None:
        center_idx = n // 2
    for i in range(n):
        out[i] += np.random.normal(0, noise_std)
        dc = abs(i - center_idx) / max(1, n // 2)
        if bias_type == "tail_under":
            if dc > 0.6: out[i] -= bias_str * dc
            else: out[i] += bias_str * 0.1
        elif bias_type == "tail_over":
            if dc > 0.6: out[i] += bias_str * dc
            else: out[i] -= bias_str * 0.1
        elif bias_type == "center_heavy":
            if dc < 0.3: out[i] += bias_str * 0.15
            elif dc > 0.5: out[i] -= bias_str * dc * 0.3
    out = [max(0.01, min(0.99, p)) for p in out]
    total = sum(out)
    return [p / total for p in out]


def classify_band(bmin, bmax, dist_mean, dist_std):
    if bmin is None or bmax is None:
        return "tail"
    mid = (bmin + bmax) / 2.0
    sigma = abs(mid - dist_mean) / dist_std if dist_std > 0 else abs(mid - dist_mean) / 2.0
    if sigma <= 0.75: return "center"
    elif sigma <= 1.5: return "shoulder"
    else: return "tail"


def get_threshold(band_pos, confidence, nowcast):
    if band_pos == "tail": t = V2_EDGE_TAIL
    elif band_pos == "shoulder": t = V2_EDGE_SHOULDER
    else: t = V2_EDGE_CENTER
    if confidence == "LOW": t = max(t, V2_EDGE_LOW_CONF)
    elif confidence == "MEDIUM": t = max(t, V2_EDGE_MEDIUM_CONF)
    if abs(nowcast) > 0.5: t *= V2_NOWCAST_DISCOUNT
    return t


def kelly_contracts(model_prob, market_prob, edge, bankroll):
    if edge > 0:
        p = model_prob
        b = (1.0 - market_prob) / market_prob if market_prob > 0.01 else 99.0
    else:
        p = 1.0 - model_prob
        b = market_prob / (1.0 - market_prob) if market_prob < 0.99 else 99.0
    q = 1.0 - p
    kf = max(0.0, (p * b - q) / b) if b > 0 else 0.0
    bet = kf * KELLY_FRACTION * bankroll
    bet = min(bet, bankroll * KELLY_MAX_PCT)  # max 5% of bankroll per bet
    if bet < MIN_BET:
        return 0.0
    return bet


def settle(temp, bmin, bmax):
    if bmin is None and bmax is not None: return temp < bmax
    if bmax is None and bmin is not None: return temp >= bmin
    if bmin is not None and bmax is not None: return bmin <= temp < bmax
    return False


# ─── Year simulation ────────────────────────────────────────────────────────

def pick_day_profile():
    """Randomly select a day's market condition profile."""
    r = random.random()
    cumulative = 0.0
    for name, profile in DAY_PROFILES.items():
        cumulative += profile["weight"]
        if r < cumulative:
            return name, profile
    return list(DAY_PROFILES.keys())[-1], list(DAY_PROFILES.values())[-1]


def simulate_one_year(starting_bankroll):
    """Simulate 365 trading days, return daily bankroll curve."""
    bankroll = starting_bankroll
    curve = [bankroll]
    daily_pnl = []
    total_bets = 0
    total_wins = 0
    total_signals = 0
    peak = bankroll
    max_dd = 0.0
    busted = False

    for day in range(TRADING_DAYS):
        if bankroll < 10.0:
            # Busted — can't trade anymore
            busted = True
            curve.append(bankroll)
            daily_pnl.append(0.0)
            continue

        profile_name, profile = pick_day_profile()

        if profile.get("skip"):
            curve.append(bankroll)
            daily_pnl.append(0.0)
            continue

        # Draw day's conditions
        true_std = profile.get("true_std_override", np.random.uniform(1.5, 3.5))
        true_mean = np.random.normal(50, 8)  # varies by city/season
        model_mean_err = np.random.normal(*profile["model_mean_err"])
        model_std_err = max(0.5, np.random.normal(*profile["model_std_err"]))
        market_noise = max(0.005, np.random.normal(*profile["market_noise"]))
        stale_shift = profile["stale_shift"]
        nowcast = profile["nowcast"]
        bias_type = profile["market_bias"]
        bias_str = profile["market_bias_str"]

        # Generate reality
        true_temp = np.random.normal(true_mean, true_std)

        # Generate bands
        band_center = round(true_mean / BAND_WIDTH) * BAND_WIDTH
        bands = generate_bands(band_center)
        true_probs = compute_probs(bands, true_mean, true_std)

        # Market pricing
        if bias_type == "stale":
            stale_probs = compute_probs(bands, true_mean + stale_shift, true_std)
            market_probs = add_noise(stale_probs, market_noise, "none", 0.0)
        else:
            market_probs = add_noise(true_probs, market_noise, bias_type, bias_str)

        # Build model distribution
        model_mean = true_mean + model_mean_err
        model_std = true_std * model_std_err
        members = []
        weights = []
        for _ in range(50):
            if random.random() < 0.85:
                v = np.random.normal(model_mean, model_std)
            else:
                v = model_mean + model_std * np.random.standard_t(df=10)
            members.append(float(v))
            weights.append(random.choice([2.5, 1.5, 1.0, 0.7, 0.7]))

        dist = TemperatureDistribution(
            city="SIM", mean=model_mean, std=model_std,
            member_values=members, member_weights=weights,
            sources={"sim": members}, forecast_date=date.today(),
            bias_correction_f=nowcast * stale_shift * 0.7 if nowcast else 0.0,
        )

        # Find and trade edges
        day_pnl = 0.0
        for i, (bmin, bmax, label) in enumerate(bands):
            model_prob = dist.probability_for_band(bmin, bmax)
            market_prob = market_probs[i]
            edge = model_prob - market_prob

            band_pos = classify_band(bmin, bmax, dist.mean, dist.std)
            threshold = get_threshold(band_pos, dist.confidence,
                                      dist.bias_correction_f)

            if abs(edge) < threshold:
                continue
            if model_prob < 0.02 and market_prob < 0.02:
                continue
            if bmin is not None and bmax is not None:
                if (bmax - bmin) <= 3.0 and market_prob > 0.50:
                    continue

            bet_size = kelly_contracts(model_prob, market_prob, edge, bankroll)
            if bet_size < MIN_BET:
                continue

            total_signals += 1

            # Settle
            if edge > 0:
                yes = settle(true_temp, bmin, bmax)
                gross = (1.0 - market_prob) * bet_size if yes else -market_prob * bet_size
            else:
                yes = settle(true_temp, bmin, bmax)
                gross = market_prob * bet_size if not yes else -(1.0 - market_prob) * bet_size

            fee = KALSHI_FEE * 2 * bet_size
            net = gross - fee
            day_pnl += net
            total_bets += 1
            if net > 0:
                total_wins += 1

        bankroll += day_pnl
        bankroll = max(0.0, bankroll)
        curve.append(bankroll)
        daily_pnl.append(day_pnl)

        if bankroll > peak:
            peak = bankroll
        dd = peak - bankroll
        if dd > max_dd:
            max_dd = dd

    return {
        "final": bankroll,
        "curve": curve,
        "daily_pnl": daily_pnl,
        "total_bets": total_bets,
        "total_signals": total_signals,
        "total_wins": total_wins,
        "win_rate": total_wins / total_bets if total_bets else 0,
        "max_drawdown": max_dd,
        "peak": peak,
        "busted": busted,
        "profit": bankroll - starting_bankroll,
        "return_pct": (bankroll - starting_bankroll) / starting_bankroll * 100,
    }


def run_scenario_set(name, profile_overrides=None, n_sims=NUM_SIMULATIONS):
    """Run N full-year simulations and collect statistics."""
    results = []

    if profile_overrides:
        saved = {}
        for key, val in profile_overrides.items():
            if key in DAY_PROFILES:
                saved[key] = DAY_PROFILES[key].copy()
                DAY_PROFILES[key].update(val)

    for i in range(n_sims):
        np.random.seed(SEED + i)
        random.seed(SEED + i)
        r = simulate_one_year(STARTING_BANKROLL)
        results.append(r)

    if profile_overrides:
        for key, val in saved.items():
            DAY_PROFILES[key] = val

    finals = [r["final"] for r in results]
    profits = [r["profit"] for r in results]
    returns = [r["return_pct"] for r in results]
    max_dds = [r["max_drawdown"] for r in results]
    bets = [r["total_bets"] for r in results]
    win_rates = [r["win_rate"] for r in results]
    busted = sum(1 for r in results if r["busted"])

    return {
        "name": name,
        "n_sims": n_sims,
        "results": results,
        "final_median": float(np.median(finals)),
        "final_mean": float(np.mean(finals)),
        "final_p10": float(np.percentile(finals, 10)),
        "final_p25": float(np.percentile(finals, 25)),
        "final_p75": float(np.percentile(finals, 75)),
        "final_p90": float(np.percentile(finals, 90)),
        "profit_median": float(np.median(profits)),
        "return_median": float(np.median(returns)),
        "return_mean": float(np.mean(returns)),
        "max_dd_median": float(np.median(max_dds)),
        "max_dd_p90": float(np.percentile(max_dds, 90)),
        "bets_median": float(np.median(bets)),
        "win_rate_median": float(np.median(win_rates)),
        "bust_rate": busted / n_sims * 100,
        "profitable_pct": sum(1 for p in profits if p > 0) / n_sims * 100,
    }


# ─── Output ─────────────────────────────────────────────────────────────────

def print_header(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def print_scenario(s):
    print(f"\n  {s['name']}")
    print(f"  {'─' * 60}")
    print(f"  Starting bankroll:     ${STARTING_BANKROLL:,.0f}")
    print(f"  Simulations:           {s['n_sims']}")
    print(f"  Trading days:          {TRADING_DAYS}")
    print()
    print(f"  ENDING BANKROLL:")
    print(f"    Median:              ${s['final_median']:,.0f}")
    print(f"    Mean:                ${s['final_mean']:,.0f}")
    print(f"    10th percentile:     ${s['final_p10']:,.0f}   (bad luck)")
    print(f"    25th percentile:     ${s['final_p25']:,.0f}")
    print(f"    75th percentile:     ${s['final_p75']:,.0f}")
    print(f"    90th percentile:     ${s['final_p90']:,.0f}   (good luck)")
    print()
    print(f"  RETURNS:")
    print(f"    Median return:       {s['return_median']:+.1f}%")
    print(f"    Mean return:         {s['return_mean']:+.1f}%")
    print(f"    % of sims profitable:{s['profitable_pct']:.0f}%")
    print(f"    Bust rate (<$10):    {s['bust_rate']:.1f}%")
    print()
    print(f"  RISK:")
    print(f"    Median max drawdown: ${s['max_dd_median']:,.0f}")
    print(f"    90th pctl drawdown:  ${s['max_dd_p90']:,.0f}")
    print()
    print(f"  ACTIVITY:")
    print(f"    Median bets/year:    {s['bets_median']:.0f}")
    print(f"    Bets/day (avg):      {s['bets_median'] / TRADING_DAYS:.1f}")
    print(f"    Median win rate:     {s['win_rate_median']:.1%}")


def print_monthly_curve(results):
    """Show a sample month-by-month bankroll trajectory."""
    # Pick the median outcome
    finals = [r["final"] for r in results]
    sorted_idx = np.argsort(finals)
    median_idx = sorted_idx[len(sorted_idx) // 2]
    curve = results[median_idx]["curve"]

    print(f"\n  MEDIAN PATH — Month-by-month bankroll:")
    print(f"  {'Month':<12s} {'Bankroll':>10s} {'Change':>10s}")
    print(f"  {'─'*12} {'─'*10} {'─'*10}")
    for month in range(13):
        day = min(month * 30, len(curve) - 1)
        val = curve[day]
        if month == 0:
            change = 0
        else:
            prev_day = min((month - 1) * 30, len(curve) - 1)
            change = curve[day] - curve[prev_day]
        label = f"Month {month}" if month > 0 else "Start"
        print(f"  {label:<12s} ${val:>9,.0f} ${change:>+9,.0f}")


def main():
    print_header("MARK JOHNSON — FULL YEAR BANKROLL PROJECTION")
    print(f"  Starting bankroll: ${STARTING_BANKROLL:,.0f}")
    print(f"  {NUM_SIMULATIONS} Monte Carlo paths x {TRADING_DAYS} trading days")
    print(f"  Kelly fraction: {KELLY_FRACTION:.0%} | Max bet: {KELLY_MAX_PCT:.0%} of bankroll")
    print(f"  Day profiles: 35% efficient, 20% slight edge, 10% nowcast,")
    print(f"                5% panic, 15% model-worse, 10% high-unc, 5% no-trade")

    # ── Scenario 1: BASE CASE ────────────────────────────────────────────
    print_header("SCENARIO 1: BASE CASE (realistic mix)")
    base = run_scenario_set("BASE CASE")
    print_scenario(base)
    print_monthly_curve(base["results"])

    # ── Scenario 2: CONSERVATIVE ─────────────────────────────────────────
    # Assume worse conditions: more efficient days, fewer nowcast edges
    print_header("SCENARIO 2: CONSERVATIVE (market is smarter than you think)")
    conservative = run_scenario_set("CONSERVATIVE", {
        "efficient": {"weight": 0.45},
        "slight_model_edge": {"weight": 0.15},
        "nowcast_edge": {"weight": 0.05},
        "panic_market": {"weight": 0.03},
        "model_worse": {"weight": 0.20},
        "high_uncertainty": {"weight": 0.07},
        "no_signal": {"weight": 0.05},
    })
    print_scenario(conservative)

    # ── Scenario 3: OPTIMISTIC ───────────────────────────────────────────
    # More nowcast edges, more panic markets, fewer model-worse days
    print_header("SCENARIO 3: OPTIMISTIC (you're actually good at this)")
    optimistic = run_scenario_set("OPTIMISTIC", {
        "efficient": {"weight": 0.25},
        "slight_model_edge": {"weight": 0.25},
        "nowcast_edge": {"weight": 0.15},
        "panic_market": {"weight": 0.08},
        "model_worse": {"weight": 0.10},
        "high_uncertainty": {"weight": 0.10},
        "no_signal": {"weight": 0.07},
    })
    print_scenario(optimistic)

    # ── Scenario 4: NOWCAST ONLY ─────────────────────────────────────────
    # Only trade when you have a METAR nowcast edge — skip everything else
    print_header("SCENARIO 4: NOWCAST ONLY (disciplined — only trade same-day METAR signals)")
    nowcast_only = run_scenario_set("NOWCAST ONLY", {
        "efficient": {"weight": 0.0},
        "slight_model_edge": {"weight": 0.0},
        "nowcast_edge": {"weight": 0.15},
        "panic_market": {"weight": 0.0},
        "model_worse": {"weight": 0.0},
        "high_uncertainty": {"weight": 0.0},
        "no_signal": {"weight": 0.85},
    })
    print_scenario(nowcast_only)

    # ── Summary ──────────────────────────────────────────────────────────
    print_header("SUMMARY: WHERE IS YOUR $1,000 AT YEAR END?")

    print(f"\n  {'Scenario':<40s} {'Median $':>9s} {'Return':>8s} {'Profitable':>11s} {'Bust':>6s} {'Max DD':>8s}")
    print(f"  {'─'*40} {'─'*9} {'─'*8} {'─'*11} {'─'*6} {'─'*8}")
    for s in [conservative, base, optimistic, nowcast_only]:
        print(f"  {s['name']:<40s} ${s['final_median']:>8,.0f} {s['return_median']:>+7.1f}% "
              f"{s['profitable_pct']:>10.0f}% {s['bust_rate']:>5.1f}% ${s['max_dd_median']:>7,.0f}")

    print(f"""
  BOTTOM LINE:

  With $1,000 and realistic assumptions:

  - BASE CASE: You'll most likely end the year around ${base['final_median']:,.0f}
    ({base['return_median']:+.0f}% return). {base['profitable_pct']:.0f}% chance of being profitable.

  - WORST CASE (conservative): ~${conservative['final_p10']:,.0f} at 10th percentile.
    You could lose ${(1 - conservative['final_p10']/1000)*100:.0f}% of your bankroll.

  - BEST CASE (optimistic): ~${optimistic['final_p90']:,.0f} at 90th percentile.

  - SAFEST PATH: Only trade nowcast/METAR signals.
    Median ${nowcast_only['final_median']:,.0f}, {nowcast_only['profitable_pct']:.0f}% profitable,
    but only ~{nowcast_only['bets_median']:.0f} bets/year.

  CRITICAL CAVEATS:
  - This assumes you can actually execute at mid-market. Real fills
    will be worse, especially on illiquid contracts.
  - No slippage modeled. Kalshi books are thin — a $50 order can
    move the market 2-5 cents on some contracts.
  - The "nowcast edge" frequency (10% of days) is an estimate.
    Could be higher or lower depending on season and city.
  - Kelly sizing amplifies both gains AND losses. If the model
    has a systematic bias you haven't caught, Kelly will size
    into it aggressively and you'll draw down fast.
""")


if __name__ == "__main__":
    main()
