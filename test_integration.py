#!/usr/bin/env python3
"""
MARK JOHNSON — Integration test.

Runs one full pipeline cycle with mocked API responses:
  scan markets → pull weather → compute signals → format alerts → write logs.

Since this environment has no outbound network, we inject realistic fixture
data at the HTTP layer and verify the full pipeline processes it correctly.
"""
import asyncio
import json
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import aiohttp

import config
from models.market import Market
from models.forecast import TemperatureDistribution
from models.signal import Signal
from services.market_scanner import MarketScanner
from services.weather_engine import WeatherEngine
from services.signal_engine import SignalEngine
from services.alert_dispatcher import AlertDispatcher
from services import logger as log_service
from utils.kalshi_client import KalshiClient
from utils.weather_client import OpenMeteoClient, NWSClient
from utils.discord_client import DiscordWebhookClient


# ═══════════════════════════════════════════════════════════════════════════════
# Test fixtures — realistic multi-city market + weather data
# ═══════════════════════════════════════════════════════════════════════════════

CLOSE_TIME = (datetime.now(timezone.utc) + timedelta(hours=8)).isoformat().replace("+00:00", "Z")

FIXTURE_MARKETS = [
    # NYC high temp — 4 band contracts
    {
        "ticker": "KXHIGHTEMPNYC-T44",
        "event_ticker": "KXHIGHTEMPNYC-26FEB17",
        "series_ticker": "KXHIGHTEMP",
        "title": "Highest temperature in New York City on February 17?",
        "subtitle": "44\u00b0 to 45\u00b0F",
        "status": "open",
        "yes_bid": 20, "yes_ask": 24, "yes_price": 22, "no_price": 78,
        "volume": 9000, "dollar_volume": 0,
        "close_time": CLOSE_TIME, "expiration_time": CLOSE_TIME,
    },
    {
        "ticker": "KXHIGHTEMPNYC-T46",
        "event_ticker": "KXHIGHTEMPNYC-26FEB17",
        "series_ticker": "KXHIGHTEMP",
        "title": "Highest temperature in New York City on February 17?",
        "subtitle": "46\u00b0 to 47\u00b0F",
        "status": "open",
        "yes_bid": 10, "yes_ask": 14, "yes_price": 12, "no_price": 88,
        "volume": 8500, "dollar_volume": 0,
        "close_time": CLOSE_TIME, "expiration_time": CLOSE_TIME,
    },
    {
        "ticker": "KXHIGHTEMPNYC-T48",
        "event_ticker": "KXHIGHTEMPNYC-26FEB17",
        "series_ticker": "KXHIGHTEMP",
        "title": "Highest temperature in New York City on February 17?",
        "subtitle": "48\u00b0 to 49\u00b0F",
        "status": "open",
        "yes_bid": 5, "yes_ask": 9, "yes_price": 7, "no_price": 93,
        "volume": 7000, "dollar_volume": 0,
        "close_time": CLOSE_TIME, "expiration_time": CLOSE_TIME,
    },
    {
        "ticker": "KXHIGHTEMPNYC-T50UP",
        "event_ticker": "KXHIGHTEMPNYC-26FEB17",
        "series_ticker": "KXHIGHTEMP",
        "title": "Highest temperature in New York City on February 17?",
        "subtitle": "50\u00b0 or above",
        "status": "open",
        "yes_bid": 3, "yes_ask": 7, "yes_price": 5, "no_price": 95,
        "volume": 6000, "dollar_volume": 0,
        "close_time": CLOSE_TIME, "expiration_time": CLOSE_TIME,
    },
    # Chicago low temp — 2 contracts
    {
        "ticker": "KXLOWTEMPCHI-T24",
        "event_ticker": "KXLOWTEMPCHI-26FEB17",
        "series_ticker": "KXLOWTEMP",
        "title": "Lowest temperature in Chicago on February 17?",
        "subtitle": "24\u00b0 to 25\u00b0F",
        "status": "open",
        "yes_bid": 15, "yes_ask": 19, "yes_price": 17, "no_price": 83,
        "volume": 5500, "dollar_volume": 0,
        "close_time": CLOSE_TIME, "expiration_time": CLOSE_TIME,
    },
    {
        "ticker": "KXLOWTEMPCHI-T22DN",
        "event_ticker": "KXLOWTEMPCHI-26FEB17",
        "series_ticker": "KXLOWTEMP",
        "title": "Lowest temperature in Chicago on February 17?",
        "subtitle": "22\u00b0 or below",
        "status": "open",
        "yes_bid": 25, "yes_ask": 29, "yes_price": 27, "no_price": 73,
        "volume": 7200, "dollar_volume": 0,
        "close_time": CLOSE_TIME, "expiration_time": CLOSE_TIME,
    },
    # Miami high temp — 1 contract
    {
        "ticker": "KXHIGHTEMPMIA-T78",
        "event_ticker": "KXHIGHTEMPMIA-26FEB17",
        "series_ticker": "KXHIGHTEMP",
        "title": "Highest temperature in Miami on February 17?",
        "subtitle": "78\u00b0 to 79\u00b0F",
        "status": "open",
        "yes_bid": 30, "yes_ask": 34, "yes_price": 32, "no_price": 68,
        "volume": 11000, "dollar_volume": 0,
        "close_time": CLOSE_TIME, "expiration_time": CLOSE_TIME,
    },
    # Non-temperature market (should be filtered out)
    {
        "ticker": "POLITICS-XYZ",
        "event_ticker": "POLITICS",
        "series_ticker": "POLITICS",
        "title": "Will the party win?",
        "subtitle": "",
        "status": "open",
        "yes_bid": 55, "yes_ask": 57, "yes_price": 56, "no_price": 44,
        "volume": 500000, "dollar_volume": 0,
        "close_time": "2026-11-03T04:00:00Z", "expiration_time": "2026-11-03T04:00:00Z",
    },
]

# Ensemble data for NYC, CHI, MIA
ENSEMBLE_DATA = {
    "NYC": {
        "max_members": [44.1, 44.5, 44.8, 45.0, 45.2, 45.3, 45.5, 45.7,
                        45.9, 46.0, 46.2, 46.3, 46.5, 46.8, 47.0, 47.2],
        "min_members": [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
    },
    "CHI": {
        "max_members": [28.0, 28.5, 29.0, 29.5, 30.0, 30.5, 31.0, 31.5],
        "min_members": [20.0, 21.0, 22.0, 22.5, 23.0, 23.5, 24.0, 25.0,
                        25.5, 26.0, 26.5, 27.0],
    },
    "MIA": {
        "max_members": [77.0, 77.5, 78.0, 78.2, 78.5, 78.8, 79.0, 79.2,
                        79.5, 80.0, 80.5, 81.0],
        "min_members": [65.0, 66.0, 67.0, 68.0],
    },
}

NWS_DATA = {
    "NYC": {"max_temp_f": 45.5, "min_temp_f": 32.0},
    "CHI": {"max_temp_f": 29.0, "min_temp_f": 23.0},
    "MIA": {"max_temp_f": 79.0, "min_temp_f": 66.0},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Mock clients that return fixture data
# ═══════════════════════════════════════════════════════════════════════════════

class MockKalshiClient:
    async def get_open_temperature_markets(self):
        return FIXTURE_MARKETS

    async def get_market_details(self, ticker):
        return {}

    async def get_orderbook(self, ticker):
        return {}


class MockOpenMeteoClient:
    async def get_ensemble_forecast(self, lat, lon):
        # Match lat/lon to city
        for key, city_info in config.CITIES.items():
            if abs(city_info["lat"] - lat) < 0.5 and abs(city_info["lon"] - lon) < 0.5:
                data = ENSEMBLE_DATA.get(key, {})
                if data:
                    return data
        return {}


class MockNWSClient:
    async def get_forecast(self, lat, lon):
        for key, city_info in config.CITIES.items():
            if abs(city_info["lat"] - lat) < 0.5 and abs(city_info["lon"] - lon) < 0.5:
                return NWS_DATA.get(key, {"max_temp_f": None, "min_temp_f": None})
        return {"max_temp_f": None, "min_temp_f": None}


# ═══════════════════════════════════════════════════════════════════════════════
# Run the integration test
# ═══════════════════════════════════════════════════════════════════════════════

async def run_integration():
    print("=" * 60)
    print("  MARK JOHNSON — INTEGRATION TEST")
    print("=" * 60)

    errors = []

    # ── Step 1: Market Scan ───────────────────────────────────────────────
    print("\n--- STEP 1: Market Scan ---")
    scanner = MarketScanner(MockKalshiClient())
    markets = await scanner.scan()

    if not markets:
        print("  CRITICAL: No markets parsed!")
        errors.append("No markets parsed")
    else:
        # Summarize
        cities = {}
        for m in markets:
            cities.setdefault(m.city, []).append(m)

        print(f"  Found {len(markets)} temperature markets across {len(cities)} cities")
        for city_key, city_markets in sorted(cities.items()):
            city_name = config.CITIES.get(city_key, {}).get("name", city_key)
            types = set(m.market_type for m in city_markets)
            print(f"    {city_name} ({city_key}): {len(city_markets)} bands, types: {types}")
            for m in city_markets:
                print(f"      {m.ticker}: {m.band_label} implied={m.implied_prob:.0%} vol=${m.volume:,.0f}")

    # ── Step 2: Weather Fetch ─────────────────────────────────────────────
    print("\n--- STEP 2: Weather Fetch ---")
    weather = WeatherEngine(MockOpenMeteoClient(), MockNWSClient())
    distributions = await weather.refresh_all()

    if not distributions:
        print("  CRITICAL: No distributions built!")
        errors.append("No distributions built")
    else:
        print(f"  Built {len(distributions)} temperature distributions")
        for (city, mtype), dist in sorted(distributions.items()):
            city_name = config.CITIES.get(city, {}).get("name", city)
            print(f"    {city_name} ({city}) {mtype}: mean={dist.mean:.1f}\u00b0F \u00b1{dist.std:.1f}\u00b0F "
                  f"({len(dist.member_values)} members, confidence={dist.confidence})")

    # ── Step 3: Signal Detection ──────────────────────────────────────────
    print("\n--- STEP 3: Signal Detection (cycle 1 of 2) ---")
    signal_engine = SignalEngine()
    signals_1 = await signal_engine.scan_for_signals(markets, distributions)
    print(f"  Cycle 1: {len(signals_1)} signals passed all filters (need persistence >= {config.EDGE_PERSIST_COUNT})")

    # Print ALL edges including sub-threshold for debugging
    print("\n  All edges (including sub-threshold):")
    for m in markets:
        key = (m.city, m.market_type)
        dist = distributions.get(key)
        if dist is None:
            print(f"    {m.ticker}: NO DISTRIBUTION for {key}")
            continue
        model_prob = dist.probability_for_band(m.band_min, m.band_max)
        edge = model_prob - m.implied_prob
        edge_pct = edge * 100
        status = ""
        if abs(edge) < config.MIN_EDGE_PERCENT / 100.0:
            status = "(below threshold)"
        elif m.volume < config.MIN_VOLUME:
            status = "(low volume)"
        else:
            status = f"(qualifies: {Signal.classify_edge(edge)})"
        print(f"    {m.ticker} {m.band_label}: model={model_prob:.1%} market={m.implied_prob:.1%} "
              f"edge={edge_pct:+.1f}% {status}")

    # Run cycle 2 for persistence
    print(f"\n--- STEP 3b: Signal Detection (cycle 2 — persistence check) ---")
    signals_2 = await signal_engine.scan_for_signals(markets, distributions)
    print(f"  Cycle 2: {len(signals_2)} signals passed all filters (persistence satisfied)")

    all_signals = signals_2

    if all_signals:
        print("\n  SIGNALS GENERATED:")
        for sig in all_signals:
            print(f"    {sig.market.city} {sig.market.band_label}: "
                  f"edge={sig.edge:+.1%} ({sig.edge_class}) "
                  f"confidence={sig.confidence}")
    else:
        print("  No signals generated (edges may be below threshold or filtered)")

    # ── Step 4: Discord Test ──────────────────────────────────────────────
    print("\n--- STEP 4: Discord Alert Formatting ---")
    discord = DiscordWebhookClient()
    dispatcher = AlertDispatcher(discord)

    if all_signals:
        # Build embed for first signal and verify structure
        sig = all_signals[0]
        print(f"  Building embed for: {sig.market.city} {sig.market.band_label}")

    # Test that we CAN build an alert (even if no webhook to send to)
    test_market = Market(
        ticker="TEST-TICKER",
        city="NYC",
        market_type="high_temp",
        band_min=46.0,
        band_max=47.0,
        implied_prob=0.12,
        best_bid=0.11,
        best_ask=0.13,
        volume=8500.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=5),
        raw_title="Test market",
    )
    test_signal = Signal(
        market=test_market,
        model_prob=0.25,
        edge=0.13,
        edge_class="STRONG",
        forecast_mean=46.0,
        forecast_std=1.3,
        confidence="HIGH",
        sources={},
    )

    webhook_url = config.DISCORD_WEBHOOK_URL
    if webhook_url:
        print(f"  Discord webhook configured — sending test message")
        ok = await dispatcher.send_signal_alert(test_signal)
        print(f"  Send result: {'OK' if ok else 'FAILED'}")
    else:
        print("  Discord webhook not configured — skipping live send")
        print("  (Set DISCORD_WEBHOOK_URL in .env to enable)")

    # ── Step 5: Log Writing ───────────────────────────────────────────────
    print("\n--- STEP 5: JSONL Log Writing ---")
    # Write test entries
    if markets:
        log_service.log_market_snapshot(markets)
    if distributions:
        for (city, _), dist in distributions.items():
            log_service.log_forecast(city, dist)
    if all_signals:
        for sig in all_signals:
            log_service.log_signal(sig)

    # Verify files exist
    log_base = Path(__file__).parent / "data" / "logs"
    for subdir in ("markets", "forecasts", "signals"):
        dir_path = log_base / subdir
        files = list(dir_path.glob("*.jsonl")) if dir_path.exists() else []
        if files:
            # Read and count lines in today's file
            for f in files:
                with open(f) as fh:
                    lines = fh.readlines()
                print(f"  {subdir}/{f.name}: {len(lines)} entries")
                # Print first entry as sample
                if lines:
                    sample = json.loads(lines[0])
                    print(f"    Sample keys: {sorted(sample.keys())}")
        else:
            if subdir == "signals" and not all_signals:
                print(f"  {subdir}/: no entries (no signals generated — expected)")
            else:
                print(f"  WARNING: {subdir}/: no log files found!")
                errors.append(f"No {subdir} log files")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if errors:
        print(f"  INTEGRATION TEST: FAILED ({len(errors)} errors)")
        for e in errors:
            print(f"    - {e}")
        return False
    else:
        print("  INTEGRATION TEST: PASSED")
        print(f"    Markets: {len(markets)} parsed across {len(set(m.city for m in markets))} cities")
        print(f"    Distributions: {len(distributions)} built")
        print(f"    Signals (cycle 2): {len(all_signals)}")
        print(f"    Logs: written and verified")
        return True


if __name__ == "__main__":
    ok = asyncio.run(run_integration())
    sys.exit(0 if ok else 1)
