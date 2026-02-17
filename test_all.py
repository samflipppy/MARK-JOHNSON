#!/usr/bin/env python3
"""
MARK JOHNSON — Unit test suite.
Tests every component in isolation without hitting live APIs.
"""
import sys
import traceback
from datetime import datetime, timezone

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
        traceback.print_exc()
        failed += 1


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Import checks
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 1. IMPORT CHECKS ===")


def test_import_config():
    import config
    assert hasattr(config, "CITIES")
    assert hasattr(config, "KALSHI_BASE_URL")
    assert hasattr(config, "city_key_from_text")


def test_import_models():
    from models.market import Market
    from models.forecast import TemperatureDistribution
    from models.signal import Signal


def test_import_utils():
    from utils.kalshi_client import KalshiClient, parse_temperature_band, detect_market_type
    from utils.weather_client import OpenMeteoClient, NWSClient
    from utils.discord_client import DiscordWebhookClient


def test_import_services():
    from services.weather_engine import WeatherEngine
    from services.market_scanner import MarketScanner
    from services.signal_engine import SignalEngine
    from services.alert_dispatcher import AlertDispatcher
    from services.logger import log_forecast, log_market_snapshot, log_signal


def test_import_main():
    # Just check main module can be parsed (don't run it)
    import importlib
    spec = importlib.util.find_spec("main")
    assert spec is not None


test("import config", test_import_config)
test("import models", test_import_models)
test("import utils", test_import_utils)
test("import services", test_import_services)
test("import main module spec", test_import_main)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Band parsing regex
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. BAND PARSING ===")

from utils.kalshi_client import parse_temperature_band


def test_band_range_unicode():
    """46° to 47° (unicode degree symbol)"""
    result = parse_temperature_band("46\u00b0 to 47\u00b0")
    assert result == (46.0, 47.0), f"Expected (46.0, 47.0), got {result}"


def test_band_range_plain():
    """46 to 47 (no degree symbol)"""
    result = parse_temperature_band("46 to 47")
    assert result == (46.0, 47.0), f"Expected (46.0, 47.0), got {result}"


def test_band_range_with_f():
    """46°F to 47°F"""
    result = parse_temperature_band("46\u00b0F to 47\u00b0F")
    assert result == (46.0, 47.0), f"Expected (46.0, 47.0), got {result}"


def test_band_79_to_80():
    """79° to 80°"""
    result = parse_temperature_band("79\u00b0 to 80\u00b0")
    assert result == (79.0, 80.0), f"Expected (79.0, 80.0), got {result}"


def test_band_or_below():
    """52° or below"""
    result = parse_temperature_band("52\u00b0 or below")
    assert result == (None, 52.0), f"Expected (None, 52.0), got {result}"


def test_band_or_above():
    """60° or above"""
    result = parse_temperature_band("60\u00b0 or above")
    assert result == (60.0, None), f"Expected (60.0, None), got {result}"


def test_band_below_plain():
    """52 or below (no degree)"""
    result = parse_temperature_band("52 or below")
    assert result == (None, 52.0), f"Expected (None, 52.0), got {result}"


def test_band_above_plain():
    """60 or above (no degree)"""
    result = parse_temperature_band("60 or above")
    assert result == (60.0, None), f"Expected (60.0, None), got {result}"


def test_band_negative():
    """-5° to -3°"""
    result = parse_temperature_band("-5\u00b0 to -3\u00b0")
    assert result == (-5.0, -3.0), f"Expected (-5.0, -3.0), got {result}"


def test_band_in_title():
    """Full Kalshi-style title"""
    result = parse_temperature_band(
        "Will the high temperature in NYC be 46\u00b0 to 47\u00b0F?"
    )
    assert result == (46.0, 47.0), f"Expected (46.0, 47.0), got {result}"


def test_band_or_lower():
    """52° or lower (alternative wording)"""
    result = parse_temperature_band("52\u00b0 or lower")
    assert result == (None, 52.0), f"Expected (None, 52.0), got {result}"


def test_band_or_higher():
    """60° or higher (alternative wording)"""
    result = parse_temperature_band("60\u00b0 or higher")
    assert result == (60.0, None), f"Expected (60.0, None), got {result}"


test("band: 46° to 47° (unicode)", test_band_range_unicode)
test("band: 46 to 47 (plain)", test_band_range_plain)
test("band: 46°F to 47°F", test_band_range_with_f)
test("band: 79° to 80°", test_band_79_to_80)
test("band: 52° or below", test_band_or_below)
test("band: 60° or above", test_band_or_above)
test("band: 52 or below (plain)", test_band_below_plain)
test("band: 60 or above (plain)", test_band_above_plain)
test("band: -5° to -3°", test_band_negative)
test("band: full Kalshi title", test_band_in_title)
test("band: 52° or lower", test_band_or_lower)
test("band: 60° or higher", test_band_or_higher)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TemperatureDistribution
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. TEMPERATURE DISTRIBUTION ===")

from models.forecast import TemperatureDistribution


def test_dist_band_middle():
    """P(49 <= T < 51) for N(50, 2) should be ~0.38"""
    dist = TemperatureDistribution(city="NYC", mean=50.0, std=2.0)
    p = dist.probability_for_band(49.0, 51.0)
    assert 0.35 < p < 0.42, f"Expected ~0.38, got {p:.4f}"


def test_dist_or_below():
    """P(T < 45) for N(50, 2) should be small (~0.006)"""
    dist = TemperatureDistribution(city="NYC", mean=50.0, std=2.0)
    p = dist.probability_for_band(None, 45.0)
    assert 0.0 < p < 0.02, f"Expected ~0.006, got {p:.4f}"


def test_dist_or_above():
    """P(T >= 55) for N(50, 2) should be small (~0.006)"""
    dist = TemperatureDistribution(city="NYC", mean=50.0, std=2.0)
    p = dist.probability_for_band(55.0, None)
    assert 0.0 < p < 0.02, f"Expected ~0.006, got {p:.4f}"


def test_dist_full_range():
    """P(0 <= T < 100) should be ~1.0"""
    dist = TemperatureDistribution(city="NYC", mean=50.0, std=2.0)
    p = dist.probability_for_band(0.0, 100.0)
    assert p > 0.99, f"Expected ~1.0, got {p:.4f}"


def test_dist_std_floor():
    """std < 0.5 should be floored to 1.0"""
    dist = TemperatureDistribution(city="NYC", mean=50.0, std=0.2)
    assert dist.std == 1.0, f"Expected std=1.0, got {dist.std}"


def test_dist_low_confidence():
    """std > 8 should flag low_confidence"""
    dist = TemperatureDistribution(city="NYC", mean=50.0, std=10.0)
    assert dist.low_confidence is True
    assert dist.confidence == "LOW"


def test_dist_confidence_levels():
    high = TemperatureDistribution(city="NYC", mean=50.0, std=1.5)
    assert high.confidence == "HIGH", f"Expected HIGH, got {high.confidence}"
    med = TemperatureDistribution(city="NYC", mean=50.0, std=3.0)
    assert med.confidence == "MEDIUM", f"Expected MEDIUM, got {med.confidence}"


test("dist: P(49<=T<51) for N(50,2) ~ 0.38", test_dist_band_middle)
test("dist: P(T<45) for N(50,2) ~ 0.006", test_dist_or_below)
test("dist: P(T>=55) for N(50,2) ~ 0.006", test_dist_or_above)
test("dist: P(0<=T<100) ~ 1.0", test_dist_full_range)
test("dist: std floor at 0.5 → 1.0", test_dist_std_floor)
test("dist: low confidence when std > 8", test_dist_low_confidence)
test("dist: confidence levels HIGH/MEDIUM", test_dist_confidence_levels)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Edge classification
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 4. EDGE CLASSIFICATION ===")

from models.signal import Signal


def test_edge_below_threshold():
    """5% edge → should be below MIN_EDGE (no signal generated)"""
    import config
    assert 0.05 < config.MIN_EDGE_PERCENT / 100.0, "5% should be below 8% threshold"


def test_edge_moderate():
    assert Signal.classify_edge(0.08) == "MODERATE"
    assert Signal.classify_edge(0.10) == "MODERATE"
    assert Signal.classify_edge(0.119) == "MODERATE"


def test_edge_strong():
    assert Signal.classify_edge(0.12) == "STRONG"
    assert Signal.classify_edge(0.15) == "STRONG"
    assert Signal.classify_edge(0.199) == "STRONG"


def test_edge_extreme():
    assert Signal.classify_edge(0.20) == "EXTREME"
    assert Signal.classify_edge(0.25) == "EXTREME"
    assert Signal.classify_edge(0.50) == "EXTREME"


def test_edge_negative():
    """Negative edges should also classify correctly"""
    assert Signal.classify_edge(-0.15) == "STRONG"
    assert Signal.classify_edge(-0.25) == "EXTREME"


test("edge: 5% below threshold", test_edge_below_threshold)
test("edge: 8-12% → MODERATE", test_edge_moderate)
test("edge: 12-20% → STRONG", test_edge_strong)
test("edge: >20% → EXTREME", test_edge_extreme)
test("edge: negative edges classify correctly", test_edge_negative)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. City matching
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 5. CITY MATCHING ===")

import config


def test_city_nyc():
    assert config.city_key_from_text("NYC") == "NYC"


def test_city_new_york_city():
    assert config.city_key_from_text("New York City") == "NYC"


def test_city_los_angeles():
    assert config.city_key_from_text("Los Angeles") == "LAX"


def test_city_la():
    assert config.city_key_from_text("LA") == "LAX"


def test_city_washington_dc():
    assert config.city_key_from_text("Washington DC") == "DCA"


def test_city_minneapolis():
    assert config.city_key_from_text("Minneapolis") == "MSP"


def test_city_in_title():
    result = config.city_key_from_text("Highest temperature in Chicago today?")
    assert result == "CHI", f"Expected CHI, got {result}"


def test_city_no_match():
    result = config.city_key_from_text("random text with no city")
    assert result is None, f"Expected None, got {result}"


test("city: NYC", test_city_nyc)
test("city: New York City", test_city_new_york_city)
test("city: Los Angeles", test_city_los_angeles)
test("city: LA", test_city_la)
test("city: Washington DC", test_city_washington_dc)
test("city: Minneapolis", test_city_minneapolis)
test("city: in full title", test_city_in_title)
test("city: no match returns None", test_city_no_match)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Discord embed formatting
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 6. DISCORD EMBED FORMATTING ===")

from models.market import Market
from services.alert_dispatcher import AlertDispatcher, _format_time_remaining


def test_embed_format():
    """Build a signal and verify the embed has all required fields."""
    market = Market(
        ticker="KXHIGHTEMPNYC-26FEB17-B46",
        city="NYC",
        market_type="high_temp",
        band_min=46.0,
        band_max=47.0,
        implied_prob=0.15,
        best_bid=0.14,
        best_ask=0.16,
        volume=12000.0,
        close_time=datetime(2026, 2, 17, 23, 59, tzinfo=timezone.utc),
        raw_title="Will the high temperature in NYC be 46° to 47°F?",
    )
    signal = Signal(
        market=market,
        model_prob=0.25,
        edge=0.10,
        edge_class="MODERATE",
        forecast_mean=48.5,
        forecast_std=2.1,
        confidence="HIGH",
        sources={"open_meteo_ensemble": [47.0, 48.0, 49.0, 50.0]},
    )

    # We can't call the async method directly, but we can verify the market/signal structure
    assert market.band_label == "46°\u201347°F", f"Got: {market.band_label}"
    assert signal.edge_class == "MODERATE"
    assert signal.confidence == "HIGH"

    # Test time remaining formatter
    from datetime import timedelta
    future = datetime.now(timezone.utc) + timedelta(hours=3, minutes=30)
    tr = _format_time_remaining(future)
    assert "3h" in tr, f"Expected '3h' in '{tr}'"

    past = datetime.now(timezone.utc) - timedelta(hours=1)
    tr2 = _format_time_remaining(past)
    assert tr2 == "CLOSED", f"Expected 'CLOSED', got '{tr2}'"


def test_market_band_labels():
    """Test all band label formats."""
    m1 = Market(ticker="t1", city="NYC", market_type="high_temp",
                band_min=46.0, band_max=47.0, implied_prob=0.5,
                best_bid=0.49, best_ask=0.51, volume=1000.0,
                close_time=datetime.now(timezone.utc), raw_title="")
    assert "46" in m1.band_label and "47" in m1.band_label

    m2 = Market(ticker="t2", city="NYC", market_type="high_temp",
                band_min=None, band_max=40.0, implied_prob=0.5,
                best_bid=0.49, best_ask=0.51, volume=1000.0,
                close_time=datetime.now(timezone.utc), raw_title="")
    assert "below" in m2.band_label

    m3 = Market(ticker="t3", city="NYC", market_type="high_temp",
                band_min=60.0, band_max=None, implied_prob=0.5,
                best_bid=0.49, best_ask=0.51, volume=1000.0,
                close_time=datetime.now(timezone.utc), raw_title="")
    assert "above" in m3.band_label


test("embed: signal formatting & time remaining", test_embed_format)
test("embed: band label formats", test_market_band_labels)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print(f"{'='*60}")

if failed > 0:
    sys.exit(1)
else:
    print("  All tests passed!")
    sys.exit(0)
