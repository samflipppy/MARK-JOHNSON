import os
from dotenv import load_dotenv

load_dotenv()

# ── Kalshi ────────────────────────────────────────────────────────────────────
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
MARKET_POLL_INTERVAL_SECONDS = 180  # 3 minutes
MARKET_SERIES_KEYWORDS = [
    "temperature",
    "temp",
    "high temp",
    "low temp",
    "highest temperature",
    "lowest temperature",
]

# Kalshi series tickers are city-specific (not one global series).
# Discovered from https://kalshi.com/category/climate/daily-temperature
# Older cities use KXHIGH{city}, newer ones use KXHIGHT{city}.
# Low temp series all use KXLOWT{city}.
# We query each series_ticker individually to avoid paginating all markets.

# Rate-limiting for external APIs
OPENMETEO_MAX_CONCURRENT = 5  # max simultaneous Open-Meteo requests
OPENMETEO_REQUEST_DELAY = 0.3  # seconds between batches
FORECAST_DAYS = 2  # number of days to fetch from Open-Meteo (today + tomorrow)
NWS_BLEND_WEIGHT = 0.15  # weight given to NWS point forecast when blending with ensemble

# ── Weather ───────────────────────────────────────────────────────────────────
OPENMETEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
NWS_API_BASE = "https://api.weather.gov"
WEATHER_POLL_INTERVAL_SECONDS = 600  # 10 minutes

# ── Signal thresholds ─────────────────────────────────────────────────────────
MIN_EDGE_PERCENT = 8.0  # minimum edge to alert
MIN_VOLUME = 5000  # minimum market volume in dollars
MIN_TIME_TO_CLOSE_MINUTES = 60  # ignore markets closing within this window
EDGE_PERSIST_COUNT = 2  # edge must persist across N polling cycles
MAX_ENSEMBLE_SPREAD_F = 4.0  # suppress if ensemble spread exceeds this

# ── Alert ─────────────────────────────────────────────────────────────────────
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
ALERT_COOLDOWN_MINUTES = 15  # max 1 alert per city per this interval

# ── City configuration ────────────────────────────────────────────────────────
# city_key → { lat, lon, station (ICAO), name (for display / market matching) }
CITIES = {
    "NYC": {
        "lat": 40.7829, "lon": -73.9654, "station": "KNYC",
        "name": "New York City", "tz": "America/New_York",
        "kalshi_high": "KXHIGHNY", "kalshi_low": "KXLOWTNYC",
    },
    "MIA": {
        "lat": 25.7959, "lon": -80.2870, "station": "KMIA",
        "name": "Miami", "tz": "America/New_York",
        "kalshi_high": "KXHIGHMIA", "kalshi_low": "KXLOWTMIA",
    },
    "CHI": {
        "lat": 41.9742, "lon": -87.9073, "station": "KORD",
        "name": "Chicago", "tz": "America/Chicago",
        "kalshi_high": "KXHIGHCHI", "kalshi_low": "KXLOWTCHI",
    },
    "DEN": {
        "lat": 39.8561, "lon": -104.6737, "station": "KDEN",
        "name": "Denver", "tz": "America/Denver",
        "kalshi_high": "KXHIGHDEN", "kalshi_low": "KXLOWTDEN",
    },
    "AUS": {
        "lat": 30.1945, "lon": -97.6699, "station": "KAUS",
        "name": "Austin", "tz": "America/Chicago",
        "kalshi_high": "KXHIGHAUS", "kalshi_low": "KXLOWTAUS",
    },
    "LAX": {
        "lat": 34.0207, "lon": -118.6919, "station": "KCQT",
        "name": "Los Angeles", "tz": "America/Los_Angeles",
        "kalshi_high": "KXHIGHLAX", "kalshi_low": "KXLOWTLAX",
    },
    "PHL": {
        "lat": 39.8721, "lon": -75.2411, "station": "KPHL",
        "name": "Philadelphia", "tz": "America/New_York",
        "kalshi_high": "KXHIGHPHIL", "kalshi_low": "KXLOWTPHIL",
    },
    "PHX": {
        "lat": 33.4373, "lon": -112.0078, "station": "KPHX",
        "name": "Phoenix", "tz": "America/Phoenix",
        "kalshi_high": "KXHIGHTPHX", "kalshi_low": "KXLOWTPHX",
    },
    "SEA": {
        "lat": 47.4502, "lon": -122.3088, "station": "KSEA",
        "name": "Seattle", "tz": "America/Los_Angeles",
        "kalshi_high": "KXHIGHTSEA", "kalshi_low": "KXLOWTSEA",
    },
    "ATL": {
        "lat": 33.6407, "lon": -84.4277, "station": "KATL",
        "name": "Atlanta", "tz": "America/New_York",
        "kalshi_high": "KXHIGHTATL", "kalshi_low": "KXLOWTATL",
    },
    "LAS": {
        "lat": 36.0840, "lon": -115.1537, "station": "KLAS",
        "name": "Las Vegas", "tz": "America/Los_Angeles",
        "kalshi_high": "KXHIGHTLV", "kalshi_low": "KXLOWTLV",
    },
    "DCA": {
        "lat": 38.8512, "lon": -77.0402, "station": "KDCA",
        "name": "Washington DC", "tz": "America/New_York",
        "kalshi_high": "KXHIGHTDC", "kalshi_low": "KXLOWTDC",
    },
    "BOS": {
        "lat": 42.3656, "lon": -71.0096, "station": "KBOS",
        "name": "Boston", "tz": "America/New_York",
        "kalshi_high": "KXHIGHTBOS", "kalshi_low": "KXLOWTBOS",
    },
    "SFO": {
        "lat": 37.6213, "lon": -122.3790, "station": "KSFO",
        "name": "San Francisco", "tz": "America/Los_Angeles",
        "kalshi_high": "KXHIGHTSFO", "kalshi_low": "KXLOWTSFO",
    },
    "DFW": {
        "lat": 32.8998, "lon": -97.0403, "station": "KDFW",
        "name": "Dallas", "tz": "America/Chicago",
        "kalshi_high": "KXHIGHTDAL", "kalshi_low": "KXLOWTDAL",
    },
    "MSP": {
        "lat": 44.8848, "lon": -93.2223, "station": "KMSP",
        "name": "Minneapolis", "tz": "America/Chicago",
        "kalshi_high": "KXHIGHTMIN", "kalshi_low": "KXLOWTMIN",
    },
    "IAH": {
        "lat": 29.9902, "lon": -95.3368, "station": "KIAH",
        "name": "Houston", "tz": "America/Chicago",
        "kalshi_high": "KXHIGHTHOU", "kalshi_low": "KXLOWTHOU",
    },
    "SAT": {
        "lat": 29.5337, "lon": -98.4698, "station": "KSAT",
        "name": "San Antonio", "tz": "America/Chicago",
        "kalshi_high": "KXHIGHTSA", "kalshi_low": "KXLOWTSA",
    },
    "OKC": {
        "lat": 35.3931, "lon": -97.6007, "station": "KOKC",
        "name": "Oklahoma City", "tz": "America/Chicago",
        "kalshi_high": "KXHIGHTOKC", "kalshi_low": "KXLOWTOKC",
    },
    "MSY": {
        "lat": 29.9934, "lon": -90.2580, "station": "KMSY",
        "name": "New Orleans", "tz": "America/Chicago",
        "kalshi_high": "KXHIGHTNO", "kalshi_low": "KXLOWTNO",
    },
}

# Build reverse-lookup helpers (city name → city key) for market title parsing.
# Include common aliases so we can match Kalshi titles like "New York City" or "NYC".
_CITY_NAME_TO_KEY: dict[str, str] = {}
for _key, _info in CITIES.items():
    _CITY_NAME_TO_KEY[_info["name"].lower()] = _key
    _CITY_NAME_TO_KEY[_key.lower()] = _key

# Extra aliases that may appear in Kalshi market titles
_EXTRA_ALIASES: dict[str, str] = {
    "new york": "NYC",
    "nyc": "NYC",
    "los angeles": "LAX",
    "la": "LAX",
    "san fran": "SFO",
    "san francisco": "SFO",
    "dc": "DCA",
    "washington": "DCA",
    "washington d.c.": "DCA",
    "washington dc": "DCA",
    "dallas": "DFW",
    "dallas-fort worth": "DFW",
    "dallas fort worth": "DFW",
    "houston": "IAH",
    "san antonio": "SAT",
    "oklahoma city": "OKC",
    "new orleans": "MSY",
    "minneapolis": "MSP",
    "boston": "BOS",
    "philadelphia": "PHL",
    "philly": "PHL",
    "phoenix": "PHX",
    "seattle": "SEA",
    "atlanta": "ATL",
    "las vegas": "LAS",
    "vegas": "LAS",
    "miami": "MIA",
    "chicago": "CHI",
    "denver": "DEN",
    "austin": "AUS",
}
for _alias, _key in _EXTRA_ALIASES.items():
    _CITY_NAME_TO_KEY[_alias.lower()] = _key


def city_key_from_text(text: str) -> str | None:
    """Return the CITIES key that best matches *text*, or None."""
    lowered = text.lower()
    # Try longest match first to avoid "New York" matching before "New York City"
    for name in sorted(_CITY_NAME_TO_KEY, key=len, reverse=True):
        if name in lowered:
            return _CITY_NAME_TO_KEY[name]
    return None
