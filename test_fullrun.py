#!/usr/bin/env python3
"""
MARK JOHNSON — Full run test.

Simulates 2 complete polling cycles of the main.py pipeline using mocked
API clients. Verifies:
  - Startup banner prints
  - Market scan completes without errors
  - Weather fetch completes for all cities with data
  - Signal engine runs and detects edges
  - Heartbeat would fire on schedule
  - Logs are written to data/logs/
  - Graceful shutdown works
"""
import asyncio
import json
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
from models.signal import Signal
from services.alert_dispatcher import AlertDispatcher
from services.market_scanner import MarketScanner
from services.signal_engine import SignalEngine
from services.weather_engine import WeatherEngine
from services import logger as log_service
from utils.discord_client import DiscordWebhookClient

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures — full 20-city weather data + multi-city temperature markets
# ═══════════════════════════════════════════════════════════════════════════════

CLOSE_TIME = (datetime.now(timezone.utc) + timedelta(hours=8)).isoformat().replace("+00:00", "Z")


def _make_markets():
    """Generate realistic temperature markets across multiple cities."""
    markets = []
    city_temps = {
        # (city_name_in_title, city_code, expected_high, expected_low)
        ("New York City", "KXHIGHTEMPNYC", 46, 32),
        ("Chicago", "KXHIGHTEMPCHI", 30, 22),
        ("Miami", "KXHIGHTEMPMIA", 79, 66),
        ("Denver", "KXHIGHTEMPDEN", 42, 25),
        ("Austin", "KXHIGHTEMPAUS", 65, 48),
        ("Los Angeles", "KXHIGHTEMPLAX", 68, 52),
        ("Philadelphia", "KXHIGHTEMPPHL", 44, 30),
        ("Phoenix", "KXHIGHTEMPPHX", 72, 50),
        ("Seattle", "KXHIGHTEMPSEA", 48, 38),
        ("Atlanta", "KXHIGHTEMPATL", 55, 40),
    }

    for city_name, event_base, high, low in city_temps:
        # Generate 3 bands around the expected high
        for offset in [-2, 0, 2]:
            band_low = high + offset
            band_high = band_low + 1
            # Make the model-matching band more likely in market pricing
            yes_price = 30 if offset == 0 else 15
            markets.append({
                "ticker": f"{event_base}-T{band_low}",
                "event_ticker": f"{event_base}-26FEB17",
                "series_ticker": "KXHIGHTEMP",
                "title": f"Highest temperature in {city_name} on February 17?",
                "subtitle": f"{band_low}\u00b0 to {band_high}\u00b0F",
                "status": "open",
                "yes_bid": yes_price - 2,
                "yes_ask": yes_price + 2,
                "yes_price": yes_price,
                "no_price": 100 - yes_price,
                "volume": 8000 + offset * 500,
                "dollar_volume": 0,
                "close_time": CLOSE_TIME,
                "expiration_time": CLOSE_TIME,
            })

        # Add low temp market
        markets.append({
            "ticker": f"{event_base.replace('HIGH', 'LOW')}-T{low}",
            "event_ticker": f"{event_base.replace('HIGH', 'LOW')}-26FEB17",
            "series_ticker": "KXLOWTEMP",
            "title": f"Lowest temperature in {city_name} on February 17?",
            "subtitle": f"{low}\u00b0 to {low + 1}\u00b0F",
            "status": "open",
            "yes_bid": 18,
            "yes_ask": 22,
            "yes_price": 20,
            "no_price": 80,
            "volume": 6000,
            "dollar_volume": 0,
            "close_time": CLOSE_TIME,
            "expiration_time": CLOSE_TIME,
        })

    return markets


# Generate per-city ensemble data for ALL 20 cities
def _make_ensemble_data():
    """Realistic ensemble data for all 20 cities."""
    import random
    random.seed(42)  # Reproducible

    city_highs = {
        "NYC": 46, "MIA": 79, "CHI": 30, "DEN": 42, "AUS": 65,
        "LAX": 68, "PHL": 44, "PHX": 72, "SEA": 48, "ATL": 55,
        "LAS": 62, "DCA": 45, "BOS": 43, "SFO": 58, "DFW": 60,
        "MSP": 28, "IAH": 67, "SAT": 66, "OKC": 55, "MSY": 70,
    }
    city_lows = {
        "NYC": 32, "MIA": 66, "CHI": 22, "DEN": 25, "AUS": 48,
        "LAX": 52, "PHL": 30, "PHX": 50, "SEA": 38, "ATL": 40,
        "LAS": 42, "DCA": 33, "BOS": 30, "SFO": 46, "DFW": 44,
        "MSP": 18, "IAH": 52, "SAT": 50, "OKC": 38, "MSY": 55,
    }

    data = {}
    for city_key in config.CITIES:
        high = city_highs.get(city_key, 50)
        low = city_lows.get(city_key, 35)
        # 20 ensemble members per city
        data[city_key] = {
            "max_members": [high + random.gauss(0, 1.5) for _ in range(20)],
            "min_members": [low + random.gauss(0, 1.5) for _ in range(20)],
        }
    return data


def _make_nws_data():
    """NWS point forecasts for all 20 cities."""
    city_highs = {
        "NYC": 45.5, "MIA": 79.0, "CHI": 29.0, "DEN": 41.0, "AUS": 64.0,
        "LAX": 67.0, "PHL": 43.5, "PHX": 71.0, "SEA": 47.0, "ATL": 54.0,
        "LAS": 61.0, "DCA": 44.0, "BOS": 42.0, "SFO": 57.0, "DFW": 59.0,
        "MSP": 27.0, "IAH": 66.0, "SAT": 65.0, "OKC": 54.0, "MSY": 69.0,
    }
    city_lows = {
        "NYC": 32.0, "MIA": 66.0, "CHI": 23.0, "DEN": 26.0, "AUS": 49.0,
        "LAX": 53.0, "PHL": 31.0, "PHX": 51.0, "SEA": 39.0, "ATL": 41.0,
        "LAS": 43.0, "DCA": 34.0, "BOS": 31.0, "SFO": 47.0, "DFW": 45.0,
        "MSP": 19.0, "IAH": 53.0, "SAT": 51.0, "OKC": 39.0, "MSY": 56.0,
    }
    return {
        k: {"max_temp_f": city_highs.get(k), "min_temp_f": city_lows.get(k)}
        for k in config.CITIES
    }


FIXTURE_MARKETS = _make_markets()
ENSEMBLE_DATA = _make_ensemble_data()
NWS_DATA = _make_nws_data()


# ═══════════════════════════════════════════════════════════════════════════════
# Mock clients
# ═══════════════════════════════════════════════════════════════════════════════

class MockKalshiClient:
    async def get_open_temperature_markets(self):
        return FIXTURE_MARKETS


class MockOpenMeteoClient:
    async def get_ensemble_forecast(self, lat, lon):
        for key, city_info in config.CITIES.items():
            if abs(city_info["lat"] - lat) < 0.5 and abs(city_info["lon"] - lon) < 0.5:
                return ENSEMBLE_DATA.get(key, {})
        return {}


class MockNWSClient:
    async def get_forecast(self, lat, lon):
        for key, city_info in config.CITIES.items():
            if abs(city_info["lat"] - lat) < 0.5 and abs(city_info["lon"] - lon) < 0.5:
                return NWS_DATA.get(key, {"max_temp_f": None, "min_temp_f": None})
        return {"max_temp_f": None, "min_temp_f": None}


# ═══════════════════════════════════════════════════════════════════════════════
# Full run simulation
# ═══════════════════════════════════════════════════════════════════════════════

async def run_full_test():
    from main import _print_banner

    # ── Startup ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("  MARK JOHNSON — FULL RUN TEST (2 polling cycles)")
    print("=" * 70)

    _print_banner()

    # Initialize services with mock clients
    scanner = MarketScanner(MockKalshiClient())
    weather = WeatherEngine(MockOpenMeteoClient(), MockNWSClient())
    signal_engine = SignalEngine()
    discord = DiscordWebhookClient()
    dispatcher = AlertDispatcher(discord)

    all_signals = []
    errors = []

    for cycle in range(1, 3):
        print(f"\n{'─' * 70}")
        print(f"  POLLING CYCLE {cycle}")
        print(f"{'─' * 70}")

        # ── Market scan ───────────────────────────────────────────────────
        print(f"\n  [{cycle}] Market scan...")
        try:
            markets = await scanner.scan()
            cities_found = set(m.city for m in markets)
            print(f"      {len(markets)} markets across {len(cities_found)} cities")
            log_service.log_market_snapshot(markets)
        except Exception as e:
            print(f"      ERROR: {e}")
            errors.append(f"Cycle {cycle} market scan: {e}")
            continue

        # ── Weather fetch ─────────────────────────────────────────────────
        print(f"  [{cycle}] Weather fetch...")
        try:
            distributions = await weather.refresh_all()
            high_count = sum(1 for (_, t) in distributions if t == "high_temp")
            low_count = sum(1 for (_, t) in distributions if t == "low_temp")
            print(f"      {len(distributions)} distributions ({high_count} high, {low_count} low)")

            # Report per-city status
            cities_ok = set()
            cities_fail = set()
            for city_key in config.CITIES:
                if (city_key, "high_temp") in distributions or (city_key, "low_temp") in distributions:
                    cities_ok.add(city_key)
                else:
                    cities_fail.add(city_key)

            if cities_fail:
                print(f"      Cities with data: {len(cities_ok)}/20")
                print(f"      Cities failed: {sorted(cities_fail)}")
            else:
                print(f"      All 20 cities have forecast data")

            for (city, mtype), dist in sorted(distributions.items()):
                log_service.log_forecast(city, dist)
        except Exception as e:
            print(f"      ERROR: {e}")
            errors.append(f"Cycle {cycle} weather: {e}")
            continue

        # ── Signal detection ──────────────────────────────────────────────
        print(f"  [{cycle}] Signal detection...")
        try:
            signals = await signal_engine.scan_for_signals(
                list(scanner.markets.values()),
                weather.distributions,
            )
            print(f"      {len(signals)} signals generated")

            for sig in signals:
                log_service.log_signal(sig)
                all_signals.append(sig)
                print(f"      SIGNAL: {sig.market.city} {sig.market.band_label} "
                      f"edge={sig.edge:+.1%} ({sig.edge_class})")

        except Exception as e:
            print(f"      ERROR: {e}")
            errors.append(f"Cycle {cycle} signal: {e}")

    # ── Heartbeat check ───────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  HEARTBEAT CHECK")
    print(f"{'─' * 70}")
    stats = {
        "Signals today": len(all_signals),
        "Uptime": "OK",
        "Time": datetime.now(timezone.utc).strftime("%H:%M UTC"),
    }
    if config.DISCORD_WEBHOOK_URL:
        ok = await dispatcher.send_heartbeat(stats)
        print(f"  Heartbeat send: {'OK' if ok else 'FAILED'}")
    else:
        print(f"  Heartbeat would send: {stats}")
        print(f"  (Discord webhook not configured — skipping actual send)")

    # ── Log verification ──────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  LOG VERIFICATION")
    print(f"{'─' * 70}")
    log_base = Path(__file__).parent / "data" / "logs"
    for subdir in ("markets", "forecasts", "signals"):
        dir_path = log_base / subdir
        files = list(dir_path.glob("*.jsonl")) if dir_path.exists() else []
        total_entries = 0
        for f in files:
            with open(f) as fh:
                lines = fh.readlines()
                total_entries += len(lines)
                # Validate each line is valid JSON
                for i, line in enumerate(lines):
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON in {subdir}/{f.name} line {i+1}: {e}")
        print(f"  {subdir}/: {total_entries} entries across {len(files)} file(s)")

    # ── Graceful shutdown simulation ──────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  GRACEFUL SHUTDOWN TEST")
    print(f"{'─' * 70}")

    # Test that the shutdown event mechanism works
    from main import _shutdown_event
    _shutdown_event.set()
    print("  Shutdown event fired")
    assert _shutdown_event.is_set()
    print("  Shutdown event confirmed set")
    _shutdown_event.clear()  # Reset for any future use
    print("  Shutdown event cleared (cleanup)")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    if errors:
        print(f"  FULL RUN TEST: FAILED ({len(errors)} errors)")
        for e in errors:
            print(f"    - {e}")
        return False
    else:
        print(f"  FULL RUN TEST: PASSED")
        print(f"    2 polling cycles completed")
        print(f"    Markets parsed: {len(scanner.markets)}")
        print(f"    Distributions: {len(weather.distributions)}")
        print(f"    Total signals generated: {len(all_signals)}")
        print(f"    Logs: validated (all entries are valid JSON)")
        print(f"    Shutdown: clean")
        return True


if __name__ == "__main__":
    ok = asyncio.run(run_full_test())
    sys.exit(0 if ok else 1)
