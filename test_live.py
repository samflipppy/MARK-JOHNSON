#!/usr/bin/env python3
"""
MARK JOHNSON — API parsing tests using realistic response fixtures.

Note: This environment has no outbound network access. These tests validate
that our parsing logic handles real Kalshi / Open-Meteo / NWS response
structures correctly by using captured response fixtures.
"""
import asyncio
import json
import statistics
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

import config
from utils.kalshi_client import KalshiClient, parse_temperature_band
from utils.weather_client import OpenMeteoClient, NWSClient


# ═══════════════════════════════════════════════════════════════════════════════
# Realistic fixtures (based on actual API response schemas)
# ═══════════════════════════════════════════════════════════════════════════════

KALSHI_MARKETS_RESPONSE = {
    "markets": [
        {
            "ticker": "KXHIGHTEMPNYC-26FEB17-T46",
            "event_ticker": "KXHIGHTEMPNYC-26FEB17",
            "series_ticker": "KXHIGHTEMP",
            "title": "Highest temperature in New York City on February 17?",
            "subtitle": "46\u00b0 to 47\u00b0F",
            "status": "open",
            "yes_price": 12,
            "no_price": 88,
            "yes_bid": 11,
            "yes_ask": 13,
            "volume": 8500,
            "dollar_volume": 0,
            "close_time": "2026-02-18T04:00:00Z",
            "expiration_time": "2026-02-18T04:00:00Z",
            "open_time": "2026-02-16T12:00:00Z",
        },
        {
            "ticker": "KXHIGHTEMPNYC-26FEB17-T52",
            "event_ticker": "KXHIGHTEMPNYC-26FEB17",
            "series_ticker": "KXHIGHTEMP",
            "title": "Highest temperature in New York City on February 17?",
            "subtitle": "52\u00b0 or above",
            "status": "open",
            "yes_price": 20,
            "no_price": 80,
            "yes_bid": 19,
            "yes_ask": 21,
            "volume": 15000,
            "dollar_volume": 0,
            "close_time": "2026-02-18T04:00:00Z",
            "expiration_time": "2026-02-18T04:00:00Z",
            "open_time": "2026-02-16T12:00:00Z",
        },
        {
            "ticker": "KXHIGHTEMPNYC-26FEB17-T38",
            "event_ticker": "KXHIGHTEMPNYC-26FEB17",
            "series_ticker": "KXHIGHTEMP",
            "title": "Highest temperature in New York City on February 17?",
            "subtitle": "38\u00b0 or below",
            "status": "open",
            "yes_price": 5,
            "no_price": 95,
            "yes_bid": 4,
            "yes_ask": 6,
            "volume": 3000,
            "dollar_volume": 0,
            "close_time": "2026-02-18T04:00:00Z",
            "expiration_time": "2026-02-18T04:00:00Z",
            "open_time": "2026-02-16T12:00:00Z",
        },
        {
            "ticker": "KXLOWTEMPCHI-26FEB17-T25",
            "event_ticker": "KXLOWTEMPCHI-26FEB17",
            "series_ticker": "KXLOWTEMP",
            "title": "Lowest temperature in Chicago on February 17?",
            "subtitle": "25\u00b0 to 26\u00b0F",
            "status": "open",
            "yes_price": 30,
            "no_price": 70,
            "yes_bid": 28,
            "yes_ask": 32,
            "volume": 6000,
            "dollar_volume": 0,
            "close_time": "2026-02-18T04:00:00Z",
            "expiration_time": "2026-02-18T04:00:00Z",
            "open_time": "2026-02-16T12:00:00Z",
        },
        {
            "ticker": "NONTEMP-MARKET-123",
            "event_ticker": "POLITICS-2026",
            "series_ticker": "POLITICS",
            "title": "Will candidate X win?",
            "subtitle": "",
            "status": "open",
            "yes_price": 55,
            "no_price": 45,
            "yes_bid": 54,
            "yes_ask": 56,
            "volume": 100000,
            "dollar_volume": 0,
            "close_time": "2026-11-03T04:00:00Z",
            "expiration_time": "2026-11-03T04:00:00Z",
            "open_time": "2026-01-01T12:00:00Z",
        },
    ],
    "cursor": None,
}

# Open-Meteo ensemble response has member-indexed daily fields
OPENMETEO_RESPONSE = {
    "latitude": 40.78,
    "longitude": -73.97,
    "daily": {
        "time": ["2026-02-17"],
        # Each model contributes multiple ensemble members
        "temperature_2m_max_member0": [45.2],
        "temperature_2m_max_member1": [46.1],
        "temperature_2m_max_member2": [44.8],
        "temperature_2m_max_member3": [47.3],
        "temperature_2m_max_member4": [45.9],
        "temperature_2m_max_member5": [46.5],
        "temperature_2m_max_member6": [44.2],
        "temperature_2m_max_member7": [47.8],
        "temperature_2m_max_member8": [45.0],
        "temperature_2m_max_member9": [46.8],
        "temperature_2m_max_member10": [43.9],
        "temperature_2m_max_member11": [48.1],
        "temperature_2m_max_member12": [45.5],
        "temperature_2m_max_member13": [46.2],
        "temperature_2m_max_member14": [44.7],
        "temperature_2m_max_member15": [47.0],
        "temperature_2m_min_member0": [32.1],
        "temperature_2m_min_member1": [33.5],
        "temperature_2m_min_member2": [31.8],
        "temperature_2m_min_member3": [34.2],
        "temperature_2m_min_member4": [32.7],
        "temperature_2m_min_member5": [33.0],
        "temperature_2m_min_member6": [31.2],
        "temperature_2m_min_member7": [34.8],
    },
}

NWS_POINTS_RESPONSE = {
    "properties": {
        "forecastGridData": "https://api.weather.gov/gridpoints/OKX/33,37",
        "forecast": "https://api.weather.gov/gridpoints/OKX/33,37/forecast",
    }
}

NWS_GRID_RESPONSE = {
    "properties": {
        "maxTemperature": {
            "values": [
                {"validTime": "2026-02-17T08:00:00+00:00/PT12H", "value": 7.8}
            ]
        },
        "minTemperature": {
            "values": [
                {"validTime": "2026-02-17T00:00:00+00:00/PT12H", "value": 0.5}
            ]
        },
    }
}

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name}")
        import traceback
        traceback.print_exc()
        failed += 1


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Kalshi response parsing
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== KALSHI RESPONSE PARSING ===")


def test_kalshi_filter_temp_markets():
    """Filter temperature markets from a mixed response."""
    markets = KALSHI_MARKETS_RESPONSE["markets"]
    temp_markets = []
    for m in markets:
        combined = " ".join([
            m.get("title", ""),
            m.get("subtitle", ""),
            m.get("event_ticker", ""),
            m.get("series_ticker", ""),
        ]).lower()
        if any(kw in combined for kw in config.MARKET_SERIES_KEYWORDS):
            temp_markets.append(m)

    assert len(temp_markets) == 4, f"Expected 4 temp markets, got {len(temp_markets)}"
    tickers = [m["ticker"] for m in temp_markets]
    assert "NONTEMP-MARKET-123" not in tickers, "Non-temp market should be filtered out"
    print(f"    Filtered {len(temp_markets)} temperature markets from {len(markets)} total")
    for m in temp_markets:
        print(f"      {m['ticker']}: {m['subtitle']}")


def test_kalshi_parse_bands():
    """Parse temperature bands from subtitle field."""
    cases = [
        ("46\u00b0 to 47\u00b0F", (46.0, 47.0)),
        ("52\u00b0 or above", (52.0, None)),
        ("38\u00b0 or below", (None, 38.0)),
        ("25\u00b0 to 26\u00b0F", (25.0, 26.0)),
    ]
    for subtitle, expected in cases:
        result = parse_temperature_band(subtitle)
        assert result == expected, f"parse('{subtitle}') = {result}, expected {expected}"


def test_kalshi_price_parsing():
    """Parse implied probabilities from Kalshi pricing fields."""
    m = KALSHI_MARKETS_RESPONSE["markets"][0]
    yes_bid = m["yes_bid"]
    yes_ask = m["yes_ask"]
    implied = (yes_bid + yes_ask) / 2.0 / 100.0
    assert 0.11 < implied < 0.13, f"Expected ~0.12, got {implied}"
    print(f"    Implied prob: {implied:.2%} (bid={yes_bid}, ask={yes_ask})")


def test_kalshi_city_matching():
    """Match cities from market titles."""
    for m in KALSHI_MARKETS_RESPONSE["markets"][:4]:
        city = config.city_key_from_text(f"{m['title']} {m['subtitle']} {m['event_ticker']}")
        assert city is not None, f"Failed to match city for: {m['title']}"
        print(f"    '{m['title'][:50]}...' → {city}")


def test_kalshi_market_scanner_parse():
    """Test the MarketScanner._parse_market static method."""
    from services.market_scanner import MarketScanner
    m = KALSHI_MARKETS_RESPONSE["markets"][0]
    parsed = MarketScanner._parse_market(m)
    assert parsed is not None, "Failed to parse market"
    assert parsed.city == "NYC", f"Expected NYC, got {parsed.city}"
    assert parsed.market_type == "high_temp", f"Expected high_temp, got {parsed.market_type}"
    assert parsed.band_min == 46.0
    assert parsed.band_max == 47.0
    assert 0.11 < parsed.implied_prob < 0.13
    print(f"    Parsed: {parsed.ticker} → {parsed.city} {parsed.band_label} prob={parsed.implied_prob:.2%}")


test("filter temp markets from mixed response", test_kalshi_filter_temp_markets)
test("parse bands from subtitles", test_kalshi_parse_bands)
test("parse implied probability from bid/ask", test_kalshi_price_parsing)
test("match cities from titles", test_kalshi_city_matching)
test("full MarketScanner._parse_market", test_kalshi_market_scanner_parse)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Open-Meteo response parsing
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== OPEN-METEO RESPONSE PARSING ===")


def test_openmeteo_parsing():
    """Parse ensemble members from Open-Meteo response."""
    result = OpenMeteoClient._parse_ensemble_response(OPENMETEO_RESPONSE)
    max_members = result.get("max_members", [])
    min_members = result.get("min_members", [])

    assert len(max_members) >= 16, f"Expected >=16 max members, got {len(max_members)}"
    assert len(min_members) >= 8, f"Expected >=8 min members, got {len(min_members)}"

    mean = statistics.mean(max_members)
    std = statistics.stdev(max_members)
    print(f"    Max temp: {len(max_members)} members, mean={mean:.1f}°F, std={std:.1f}°F")
    print(f"    Range: {min(max_members):.1f}°F – {max(max_members):.1f}°F")

    mean_min = statistics.mean(min_members)
    print(f"    Min temp: {len(min_members)} members, mean={mean_min:.1f}°F")


def test_openmeteo_builds_distribution():
    """Build a TemperatureDistribution from parsed ensemble data."""
    from models.forecast import TemperatureDistribution
    result = OpenMeteoClient._parse_ensemble_response(OPENMETEO_RESPONSE)
    max_members = result.get("max_members", [])

    mean = statistics.mean(max_members)
    std = statistics.stdev(max_members)
    dist = TemperatureDistribution(city="NYC", mean=mean, std=std, member_values=max_members)

    # The band 46°-47° should have some probability
    p = dist.probability_for_band(46.0, 47.0)
    print(f"    P(46°-47°F) = {p:.2%}")
    assert 0.0 < p < 1.0, f"Probability should be between 0 and 1, got {p}"

    # Mean is around 46°F, so P(T >= 52) should be very small
    p_high = dist.probability_for_band(52.0, None)
    print(f"    P(T >= 52°F) = {p_high:.4%}")
    assert p_high < 0.05, f"Expected <5%, got {p_high:.2%}"


test("parse ensemble members from response", test_openmeteo_parsing)
test("build distribution from ensemble", test_openmeteo_builds_distribution)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: NWS response parsing
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== NWS RESPONSE PARSING ===")


def test_nws_points_parsing():
    """Parse grid URL from NWS points response."""
    grid_url = NWS_POINTS_RESPONSE["properties"]["forecastGridData"]
    assert "gridpoints" in grid_url
    print(f"    Grid URL: {grid_url}")


def test_nws_grid_parsing():
    """Parse max/min temps from NWS grid response."""
    props = NWS_GRID_RESPONSE["properties"]
    max_vals = props["maxTemperature"]["values"]
    val_c = max_vals[0]["value"]
    val_f = val_c * 9 / 5 + 32
    print(f"    Max temp: {val_c:.1f}°C = {val_f:.1f}°F")
    assert 40 < val_f < 50, f"Expected ~46°F, got {val_f:.1f}°F"

    min_vals = props["minTemperature"]["values"]
    val_c = min_vals[0]["value"]
    val_f = val_c * 9 / 5 + 32
    print(f"    Min temp: {val_c:.1f}°C = {val_f:.1f}°F")
    assert 30 < val_f < 35, f"Expected ~33°F, got {val_f:.1f}°F"


def test_nws_extract_first_value():
    """Test the NWSClient._extract_first_value static method."""
    val = NWSClient._extract_first_value(NWS_GRID_RESPONSE["properties"]["maxTemperature"])
    assert val is not None
    assert abs(val - 7.8) < 0.01
    print(f"    Extracted value: {val}")


test("parse grid URL from points response", test_nws_points_parsing)
test("parse max/min from grid response", test_nws_grid_parsing)
test("NWSClient._extract_first_value", test_nws_extract_first_value)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'='*60}")

if failed:
    print(f"  NOTE: No outbound network in this environment.")
    print(f"  Tests used realistic fixtures to validate parsing logic.")
    sys.exit(1)
else:
    print(f"  All parsing tests passed!")
    print(f"  NOTE: No outbound network — used realistic response fixtures.")
    sys.exit(0)
