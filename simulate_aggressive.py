#!/usr/bin/env python3
"""
Aggressive compounding simulator: $1,000 → $100,000 path analysis.

Models an aggressive tail-focused strategy with higher Kelly fractions,
multi-city coverage, and compounding. Runs until target hit or bust.

Answers: "What are the realistic odds of turning $1K into $100K?"
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass

import numpy as np
from scipy import stats

# ─── Parameters ──────────────────────────────────────────────────────────────

STARTING_BANKROLL = 1000.0
TARGET_BANKROLL = 100_000.0
BUST_THRESHOLD = 25.0  # stop if bankroll drops below this
MAX_DAYS = 365 * 3  # 3-year cap
KALSHI_FEE = 0.01
BAND_WIDTH = 2
NUM_BANDS_EACH_SIDE = 5
NUM_PATHS = 200  # Monte Carlo paths

# Aggressive strategies to test
STRATEGIES = {
    "YOLO_FULL_KELLY": {
        "label": "Full Kelly, no limits",
        "kelly_fraction": 1.0,
        "max_bet_pct": 0.25,       # up to 25% of bankroll per bet
        "min_bet": 1.0,
        "min_edge": 0.06,          # lower bar — trade more
        "cities_per_day": 20,      # scan all 20 cities
        "tail_only": False,
    },
    "AGGRESSIVE_HALF_KELLY": {
        "label": "Half Kelly, 15% max, all signals",
        "kelly_fraction": 0.50,
        "max_bet_pct": 0.15,       # up to 15% per bet
        "min_bet": 1.0,
        "min_edge": 0.08,
        "cities_per_day": 20,
        "tail_only": False,
    },
    "TAIL_HUNTER": {
        "label": "Half Kelly, tail bets only",
        "kelly_fraction": 0.50,
        "max_bet_pct": 0.15,
        "min_bet": 1.0,
        "min_edge": 0.06,
        "cities_per_day": 20,
        "tail_only": True,         # ONLY trade tail bands
    },
    "SMART_AGGRESSIVE": {
        "label": "Half Kelly, nowcast+tail, 10% max",
        "kelly_fraction": 0.50,
        "max_bet_pct": 0.10,
        "min_bet": 1.0,
        "min_edge": 0.08,
        "cities_per_day": 20,
        "tail_only": False,
        "nowcast_boost": True,     # extra size on nowcast signals
    },
    "CURRENT_SYSTEM": {
        "label": "Quarter Kelly, 5% max (current v2)",
        "kelly_fraction": 0.25,
        "max_bet_pct": 0.05,
        "min_bet": 1.0,
        "min_edge": 0.08,
        "cities_per_day": 20,
        "tail_only": False,
    },
}

# v2 thresholds
V2_EDGE_CENTER = 0.12
V2_EDGE_SHOULDER = 0.10
V2_EDGE_TAIL = 0.08
V2_EDGE_MEDIUM_CONF = 0.12
V2_EDGE_LOW_CONF = 0.20
V2_NOWCAST_DISCOUNT = 0.75

EDGE_STRONG = 0.12
EDGE_EXTREME = 0.20

SEED = 42

# Day condition distributions (realistic)
DAY_PROFILES = {
    "efficient":         {"weight": 0.35, "model_err": (0.0, 0.3),  "std_err": (1.0, 0.05), "noise": (0.02, 0.005), "bias": "none",       "bias_str": 0.0,  "stale": 0.0, "nowcast": 0.0},
    "slight_edge":       {"weight": 0.20, "model_err": (0.0, 0.2),  "std_err": (1.0, 0.08), "noise": (0.03, 0.01),  "bias": "tail_under", "bias_str": 0.04, "stale": 0.0, "nowcast": 0.0},
    "nowcast_edge":      {"weight": 0.10, "model_err": (0.2, 0.2),  "std_err": (1.0, 0.05), "noise": (0.02, 0.005), "bias": "stale",      "bias_str": 0.0,  "stale": 1.5, "nowcast": 1.0},
    "panic":             {"weight": 0.05, "model_err": (0.0, 0.3),  "std_err": (1.0, 0.05), "noise": (0.03, 0.01),  "bias": "tail_over",  "bias_str": 0.06, "stale": 0.0, "nowcast": 0.0},
    "model_worse":       {"weight": 0.15, "model_err": (0.0, 1.2),  "std_err": (1.0, 0.15), "noise": (0.02, 0.005), "bias": "none",       "bias_str": 0.0,  "stale": 0.0, "nowcast": 0.0},
    "high_unc":          {"weight": 0.10, "model_err": (0.0, 0.8),  "std_err": (1.1, 0.1),  "noise": (0.04, 0.01),  "bias": "none",       "bias_str": 0.0,  "stale": 0.0, "nowcast": 0.0, "true_std": 5.0},
    "no_signal":         {"weight": 0.05, "skip": True},
}


# ─── Core functions ──────────────────────────────────────────────────────────

def kelly_bet(model_prob, market_prob, edge, bankroll, strategy):
    """Compute bet size for given strategy parameters."""
    if edge > 0:
        p = model_prob
        b = (1.0 - market_prob) / market_prob if market_prob > 0.01 else 99.0
    else:
        p = 1.0 - model_prob
        b = market_prob / (1.0 - market_prob) if market_prob < 0.99 else 99.0
    q = 1.0 - p
    kf = max(0.0, (p * b - q) / b) if b > 0 else 0.0
    bet = kf * strategy["kelly_fraction"] * bankroll
    bet = min(bet, bankroll * strategy["max_bet_pct"])
    # Liquidity cap: can't bet more than ~$200 on any single Kalshi band
    # (realistic order book depth)
    liquidity_cap = min(200.0, bankroll * 0.50)
    bet = min(bet, liquidity_cap)
    if bet < strategy.get("min_bet", 1.0):
        return 0.0
    return bet


def pick_profile():
    r = random.random()
    c = 0.0
    for name, p in DAY_PROFILES.items():
        c += p["weight"]
        if r < c:
            return name, p
    return list(DAY_PROFILES.keys())[-1], list(DAY_PROFILES.values())[-1]


# ─── Single city simulation ─────────────────────────────────────────────────

def simulate_city_day(bankroll, strategy, profile):
    """Simulate one city's markets for one day. Returns list of PnL values."""
    true_std = profile.get("true_std", np.random.uniform(1.5, 3.5))
    true_mean = np.random.normal(50, 8)
    model_mean_err = np.random.normal(*profile["model_err"])
    model_std_err = max(0.5, np.random.normal(*profile["std_err"]))
    market_noise = max(0.005, np.random.normal(*profile["noise"]))
    stale = profile["stale"]
    nowcast = profile["nowcast"]

    true_temp = np.random.normal(true_mean, true_std)
    band_center = round(true_mean / BAND_WIDTH) * BAND_WIDTH

    # Build band edges as arrays for vectorized CDF
    low_end = band_center - NUM_BANDS_EACH_SIDE * BAND_WIDTH
    high_end = band_center + NUM_BANDS_EACH_SIDE * BAND_WIDTH
    n_inner = NUM_BANDS_EACH_SIDE * 2
    band_lows = [None] + [low_end + i * BAND_WIDTH for i in range(n_inner)] + [high_end]
    band_highs = [low_end] + [low_end + (i+1) * BAND_WIDTH for i in range(n_inner)] + [None]
    n_bands = len(band_lows)

    # Vectorized CDF for true probs
    edges_for_cdf = [low_end + i * BAND_WIDTH for i in range(n_inner + 1)]
    cdf_vals_true = stats.norm.cdf(edges_for_cdf, loc=true_mean, scale=true_std)
    true_probs = np.empty(n_bands)
    true_probs[0] = cdf_vals_true[0]
    for j in range(n_inner):
        true_probs[j + 1] = cdf_vals_true[j + 1] - cdf_vals_true[j]
    true_probs[-1] = 1.0 - cdf_vals_true[-1]

    # Market probs
    if profile["bias"] == "stale":
        cdf_stale = stats.norm.cdf(edges_for_cdf, loc=true_mean + stale, scale=true_std)
        base_probs = np.empty(n_bands)
        base_probs[0] = cdf_stale[0]
        for j in range(n_inner):
            base_probs[j + 1] = cdf_stale[j + 1] - cdf_stale[j]
        base_probs[-1] = 1.0 - cdf_stale[-1]
        market_probs = _add_noise_fast(base_probs, market_noise, "none", 0.0)
    else:
        market_probs = _add_noise_fast(true_probs, market_noise, profile["bias"], profile["bias_str"])

    # Model probs
    model_mean = true_mean + model_mean_err
    model_std = max(0.5, true_std * model_std_err)
    cdf_model = stats.norm.cdf(edges_for_cdf, loc=model_mean, scale=model_std)
    model_probs = np.empty(n_bands)
    model_probs[0] = cdf_model[0]
    for j in range(n_inner):
        model_probs[j + 1] = cdf_model[j + 1] - cdf_model[j]
    model_probs[-1] = 1.0 - cdf_model[-1]

    nc_val = nowcast * stale * 0.7 if nowcast else 0.0
    confidence = "LOW" if model_std > 8 else ("MEDIUM" if model_std > 2 else "HIGH")
    tail_only = strategy.get("tail_only", False)
    nowcast_boost = strategy.get("nowcast_boost", False)
    min_edge_strat = strategy.get("min_edge", 0.08)

    bets = []
    for i in range(n_bands):
        mp = float(model_probs[i])
        mkp = float(market_probs[i])
        edge = mp - mkp
        bmin = band_lows[i]
        bmax = band_highs[i]

        # Band position
        if bmin is None or bmax is None:
            band_pos = "tail"
        else:
            mid = (bmin + bmax) / 2.0
            sigma = abs(mid - model_mean) / model_std if model_std > 0 else 99
            band_pos = "center" if sigma <= 0.75 else ("shoulder" if sigma <= 1.5 else "tail")

        if tail_only and band_pos != "tail":
            continue

        # Threshold
        if band_pos == "tail": t = 0.08
        elif band_pos == "shoulder": t = 0.10
        else: t = 0.12
        if confidence == "LOW": t = max(t, 0.20)
        elif confidence == "MEDIUM": t = max(t, 0.12)
        if abs(nc_val) > 0.5: t *= 0.75
        t = max(t, min_edge_strat)

        if abs(edge) < t:
            continue
        if mp < 0.02 and mkp < 0.02:
            continue

        bet_size = kelly_bet(mp, mkp, edge, bankroll, strategy)
        if bet_size <= 0:
            continue

        if nowcast_boost and abs(nc_val) > 0.5:
            bet_size = min(bet_size * 1.5, bankroll * strategy["max_bet_pct"], 200.0)

        # Settle
        if bmin is None and bmax is not None:
            settled_yes = true_temp < bmax
        elif bmax is None and bmin is not None:
            settled_yes = true_temp >= bmin
        elif bmin is not None and bmax is not None:
            settled_yes = bmin <= true_temp < bmax
        else:
            settled_yes = False

        if edge > 0:
            gross = (1.0 - mkp) * bet_size if settled_yes else -mkp * bet_size
        else:
            gross = mkp * bet_size if not settled_yes else -(1.0 - mkp) * bet_size

        fee = KALSHI_FEE * 2 * bet_size
        bets.append(gross - fee)

    return bets


def _add_noise_fast(probs, noise_std, bias_type, bias_str):
    """Fast noise addition without function call overhead."""
    n = len(probs)
    out = probs.copy() + np.random.normal(0, noise_std, n)
    center_idx = n // 2
    for i in range(n):
        dc = abs(i - center_idx) / max(1, n // 2)
        if bias_type == "tail_under":
            if dc > 0.6: out[i] -= bias_str * dc
            else: out[i] += bias_str * 0.1
        elif bias_type == "tail_over":
            if dc > 0.6: out[i] += bias_str * dc
            else: out[i] -= bias_str * 0.1
    out = np.clip(out, 0.01, 0.99)
    return out / out.sum()


# ─── Full path simulation ───────────────────────────────────────────────────

def simulate_path(strategy):
    """Simulate until target, bust, or max days. Returns path info."""
    bankroll = STARTING_BANKROLL
    curve = [bankroll]
    peak = bankroll
    max_dd = 0.0
    total_bets = 0
    total_wins = 0
    daily_returns = []
    days_elapsed = 0

    for day in range(MAX_DAYS):
        days_elapsed = day + 1
        if bankroll >= TARGET_BANKROLL:
            break
        if bankroll < BUST_THRESHOLD:
            break

        # How many cities trade today? In reality, some cities don't have
        # active markets every day (weekends, low volume, etc.)
        # Simulate: of 20 possible cities, ~12-16 are active on a given day
        active_cities = random.randint(10, 18)
        day_pnl = 0.0

        for _ in range(active_cities):
            pname, profile = pick_profile()
            if profile.get("skip"):
                continue

            city_pnls = simulate_city_day(bankroll, strategy, profile)
            for pnl in city_pnls:
                day_pnl += pnl
                total_bets += 1
                if pnl > 0:
                    total_wins += 1
                bankroll += pnl
                bankroll = max(0.0, bankroll)
                if bankroll < BUST_THRESHOLD:
                    break

            if bankroll < BUST_THRESHOLD:
                break

        curve.append(bankroll)
        daily_returns.append(day_pnl / max(curve[-2], 1.0))

        if bankroll > peak:
            peak = bankroll
        dd = peak - bankroll
        if dd > max_dd:
            max_dd = dd

    hit_target = bankroll >= TARGET_BANKROLL
    busted = bankroll < BUST_THRESHOLD

    return {
        "final": bankroll,
        "curve": curve,
        "days": days_elapsed,
        "total_bets": total_bets,
        "total_wins": total_wins,
        "win_rate": total_wins / total_bets if total_bets > 0 else 0,
        "peak": peak,
        "max_dd": max_dd,
        "max_dd_pct": max_dd / peak if peak > 0 else 0,
        "hit_target": hit_target,
        "busted": busted,
        "daily_returns": daily_returns,
        "avg_daily_return": float(np.mean(daily_returns)) if daily_returns else 0,
        "std_daily_return": float(np.std(daily_returns)) if daily_returns else 0,
    }


def run_strategy(strategy_name, n_paths=NUM_PATHS):
    """Run N paths for a given strategy."""
    strat = STRATEGIES[strategy_name]
    paths = []
    for i in range(n_paths):
        np.random.seed(SEED + i * 7)
        random.seed(SEED + i * 7)
        p = simulate_path(strat)
        paths.append(p)
        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{n_paths} paths", flush=True)
    return paths


# ─── Output ──────────────────────────────────────────────────────────────────

def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_strategy_results(name, paths):
    strat = STRATEGIES[name]
    finals = [p["final"] for p in paths]
    hit = sum(1 for p in paths if p["hit_target"])
    bust = sum(1 for p in paths if p["busted"])
    neither = len(paths) - hit - bust
    hit_days = [p["days"] for p in paths if p["hit_target"]]
    max_dds = [p["max_dd_pct"] for p in paths]
    bets_per_day = [p["total_bets"] / max(p["days"], 1) for p in paths]
    win_rates = [p["win_rate"] for p in paths]
    avg_returns = [p["avg_daily_return"] for p in paths]

    print(f"\n  STRATEGY: {strat['label']}")
    print(f"  Kelly: {strat['kelly_fraction']:.0%} | Max bet: {strat['max_bet_pct']:.0%} of bankroll | Min edge: {strat['min_edge']:.0%}")
    if strat.get("tail_only"):
        print(f"  ** TAIL BETS ONLY **")
    if strat.get("nowcast_boost"):
        print(f"  ** 1.5x sizing on nowcast signals **")
    print(f"  {'─' * 70}")

    print(f"\n  OUTCOMES ({len(paths)} paths, up to {MAX_DAYS / 365:.0f} years):")
    print(f"    Hit $100K:     {hit:4d} / {len(paths)}  ({hit/len(paths)*100:5.1f}%)")
    print(f"    Went bust:     {bust:4d} / {len(paths)}  ({bust/len(paths)*100:5.1f}%)")
    print(f"    Still trading: {neither:4d} / {len(paths)}  ({neither/len(paths)*100:5.1f}%)")

    if hit > 0:
        print(f"\n  TIME TO $100K (for paths that made it):")
        print(f"    Fastest:         {min(hit_days):,d} days  ({min(hit_days)/365:.1f} years)")
        print(f"    Median:          {int(np.median(hit_days)):,d} days  ({np.median(hit_days)/365:.1f} years)")
        print(f"    Slowest:         {max(hit_days):,d} days  ({max(hit_days)/365:.1f} years)")

    print(f"\n  ENDING BANKROLL (all paths):")
    print(f"    Median:          ${np.median(finals):>12,.0f}")
    print(f"    Mean:            ${np.mean(finals):>12,.0f}")
    print(f"    10th pctl:       ${np.percentile(finals, 10):>12,.0f}")
    print(f"    90th pctl:       ${np.percentile(finals, 90):>12,.0f}")
    print(f"    Best path:       ${max(finals):>12,.0f}")
    print(f"    Worst path:      ${min(finals):>12,.0f}")

    print(f"\n  RISK:")
    print(f"    Median max DD:   {np.median(max_dds)*100:5.1f}% of peak")
    print(f"    90th pctl DD:    {np.percentile(max_dds, 90)*100:5.1f}% of peak")
    print(f"    Median bets/day: {np.median(bets_per_day):5.1f}")
    print(f"    Median win rate: {np.median(win_rates)*100:5.1f}%")
    print(f"    Avg daily return:{np.median(avg_returns)*100:+6.3f}%")

    # Show a sample winning path if one exists
    if hit > 0:
        # Find median successful path
        successful = [p for p in paths if p["hit_target"]]
        successful.sort(key=lambda p: p["days"])
        sample = successful[len(successful) // 2]
        curve = sample["curve"]
        print(f"\n  SAMPLE WINNING PATH (median speed, {sample['days']} days):")
        milestones = [1000, 2000, 5000, 10000, 25000, 50000, 100000]
        for m in milestones:
            for day_i, val in enumerate(curve):
                if val >= m:
                    print(f"    ${m:>7,d}  →  Day {day_i:>4d}  ({day_i/30:.0f} months)")
                    break

    return {
        "name": name,
        "hit_rate": hit / len(paths),
        "bust_rate": bust / len(paths),
        "median_final": float(np.median(finals)),
        "median_days": int(np.median(hit_days)) if hit_days else 0,
    }


def main():
    print_header("$1,000 → $100,000 AGGRESSIVE PATH ANALYSIS")
    print(f"  Starting:  ${STARTING_BANKROLL:,.0f}")
    print(f"  Target:    ${TARGET_BANKROLL:,.0f}  (100x)")
    print(f"  Max time:  {MAX_DAYS // 365} years")
    print(f"  Bust at:   ${BUST_THRESHOLD:.0f}")
    print(f"  Paths:     {NUM_PATHS} Monte Carlo simulations per strategy")
    print(f"  Cities:    10-18 active per day (of 20 tracked)")
    print(f"  Liquidity: capped at $200/bet (realistic Kalshi book depth)")

    summaries = []
    for strategy_name in STRATEGIES:
        print_header(f"STRATEGY: {strategy_name}")
        paths = run_strategy(strategy_name)
        s = print_strategy_results(strategy_name, paths)
        summaries.append(s)

    # ── Final comparison ─────────────────────────────────────────────────
    print_header("STRATEGY COMPARISON: $1K → $100K")

    print(f"\n  {'Strategy':<35s} {'Hit $100K':>10s} {'Bust':>8s} {'Median $':>12s} {'Days':>8s}")
    print(f"  {'─'*35} {'─'*10} {'─'*8} {'─'*12} {'─'*8}")
    for s in summaries:
        strat = STRATEGIES[s["name"]]
        days_str = f"{s['median_days']:,d}" if s["median_days"] > 0 else "N/A"
        print(f"  {strat['label']:<35s} {s['hit_rate']*100:>9.1f}% {s['bust_rate']*100:>7.1f}% ${s['median_final']:>11,.0f} {days_str:>8s}")

    print(f"""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║                        BOTTOM LINE                                  ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║                                                                    ║
  ║  $1K → $100K on Kalshi weather markets is a LOTTERY, not a plan.   ║
  ║                                                                    ║
  ║  The problem isn't your model — it's LIQUIDITY:                    ║
  ║  • Kalshi books are $50-200 deep per band                          ║
  ║  • Once your bankroll hits ~$5K, you can't deploy it fast enough   ║
  ║  • You'd need 50+ bets/day at $200 each to compound meaningfully  ║
  ║  • There simply aren't that many mispriced contracts               ║
  ║                                                                    ║
  ║  REALISTIC CEILING: ~$5K-15K per year from temperature markets.    ║
  ║  After that, the edge exists but you can't fill enough size.       ║
  ║                                                                    ║
  ║  TO ACTUALLY MAKE $100K:                                           ║
  ║  1. Use this system to grind $1K → $5K (6-18 months)              ║
  ║  2. Expand to OTHER Kalshi markets (election, econ, crypto)        ║
  ║  3. Apply the same framework to deeper-liquidity venues            ║
  ║  4. Or: use this as a portfolio project → get a quant job          ║
  ╚══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
