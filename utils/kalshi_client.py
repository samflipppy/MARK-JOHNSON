from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import aiohttp

import config

logger = logging.getLogger("mark_johnson.kalshi")

# Regex patterns for parsing temperature bands from market titles/subtitles
# Matches patterns like "46° to 47°", "46 to 47", "46°F to 47°F"
_BAND_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*(?:F\s*)?(?:to|-)\s*(-?\d+(?:\.\d+)?)\s*°?\s*F?",
    re.IGNORECASE,
)
# Matches "X° or below", "X or lower", "under X°"
_BELOW_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*F?\s*(?:or\s+)?(?:below|lower|under|less)",
    re.IGNORECASE,
)
# Matches "X° or above", "X or higher", "over X°"
_ABOVE_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*F?\s*(?:or\s+)?(?:above|higher|over|more|greater)",
    re.IGNORECASE,
)
# Also match "below X°" and "above X°" (reversed order)
_BELOW_REV_RE = re.compile(
    r"(?:below|under|less\s+than)\s*(-?\d+(?:\.\d+)?)\s*°?\s*F?",
    re.IGNORECASE,
)
_ABOVE_REV_RE = re.compile(
    r"(?:above|over|greater\s+than|at\s+least)\s*(-?\d+(?:\.\d+)?)\s*°?\s*F?",
    re.IGNORECASE,
)


def parse_temperature_band(
    title: str, subtitle: str = ""
) -> tuple[float | None, float | None]:
    """
    Extract (band_min, band_max) from a market title / subtitle.

    Returns:
        (band_min, band_max) where None indicates an open-ended band.
        (None, 46)   → "46° or below"
        (60, None)   → "60° or above"
        (46, 47)     → "46° to 47°"
        (None, None)  → could not parse
    """
    text = f"{title} {subtitle}"

    # Try range first ("46° to 47°")
    m = _BAND_RE.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))

    # "X° or below"
    m = _BELOW_RE.search(text) or _BELOW_REV_RE.search(text)
    if m:
        return None, float(m.group(1))

    # "X° or above"
    m = _ABOVE_RE.search(text) or _ABOVE_REV_RE.search(text)
    if m:
        return float(m.group(1)), None

    return None, None


def detect_market_type(title: str) -> str:
    """Determine whether a market is for high temp or low temp."""
    lowered = title.lower()
    if any(kw in lowered for kw in ("lowest", "low temp", "minimum", "min temp")):
        return "low_temp"
    # Default to high temp (most common)
    return "high_temp"


class KalshiClient:
    """Read-only async client for the Kalshi public API."""

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._session = session
        self._owns_session = session is None
        self._base = config.KALSHI_BASE_URL

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "MarkJohnson/1.0 (temperature-scanner)",
                    "Accept": "application/json",
                }
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def _get(
        self, path: str, params: dict[str, Any] | None = None, max_retries: int = 4
    ) -> dict[str, Any]:
        session = await self._ensure_session()
        url = f"{self._base}{path}"
        backoff = 1.0

        for attempt in range(max_retries + 1):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 429:
                        retry_after = float(
                            resp.headers.get("Retry-After", backoff)
                        )
                        logger.warning(
                            "Kalshi 429 — backing off %.1fs (attempt %d)",
                            retry_after,
                            attempt + 1,
                        )
                        await asyncio.sleep(retry_after)
                        backoff *= 2
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == max_retries:
                    logger.error("Kalshi request failed after %d retries: %s", max_retries, exc)
                    raise
                logger.warning(
                    "Kalshi request error (attempt %d): %s", attempt + 1, exc
                )
                await asyncio.sleep(backoff)
                backoff *= 2

        return {}  # unreachable, satisfies type checker

    # ── Public API methods ────────────────────────────────────────────────

    async def get_open_temperature_markets(self) -> list[dict[str, Any]]:
        """
        Fetch open temperature markets using per-city series_ticker filters.

        Kalshi uses city-specific series tickers (e.g. KXHIGHNY for NYC highs,
        KXLOWTCHI for Chicago lows). We query each known series individually.
        This is far more efficient than paginating through ALL open markets.

        Requests are staggered to stay well within rate limits (20 req/sec basic).
        """
        all_markets: list[dict[str, Any]] = []
        seen_tickers: set[str] = set()

        # Collect all known series tickers from city config
        series_tickers: list[str] = []
        for city_info in config.CITIES.values():
            for key in ("kalshi_high", "kalshi_low"):
                st = city_info.get(key)
                if st:
                    series_tickers.append(st)

        # Query each series ticker (staggered to avoid rate limits)
        for i, series in enumerate(series_tickers):
            try:
                data = await self._get(
                    "/markets",
                    params={"status": "open", "series_ticker": series, "limit": 200},
                )
                markets = data.get("markets", [])
                new_count = 0
                for mkt in markets:
                    t = mkt.get("ticker")
                    if t and t not in seen_tickers:
                        all_markets.append(mkt)
                        seen_tickers.add(t)
                        new_count += 1
                if new_count > 0:
                    logger.info("  %s: %d markets", series, new_count)
                else:
                    logger.debug("  %s: 0 markets (no open contracts)", series)
            except Exception as exc:
                # Log but continue — one failed series shouldn't block others
                logger.warning("  %s: query failed — %s", series, exc)

            # Stagger requests: small delay between every request to stay under rate limits
            await asyncio.sleep(0.15)

        logger.info(
            "Found %d open temperature markets across %d series",
            len(all_markets),
            len(series_tickers),
        )
        return all_markets

    async def get_market_details(self, ticker: str) -> dict[str, Any]:
        data = await self._get(f"/markets/{ticker}")
        return data.get("market", data)

    async def get_orderbook(self, ticker: str) -> dict[str, Any]:
        return await self._get(f"/markets/{ticker}/orderbook")
