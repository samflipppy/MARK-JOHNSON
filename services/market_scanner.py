from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import config
from models.market import Market
from utils.kalshi_client import (
    KalshiClient,
    detect_market_type,
    parse_temperature_band,
    parse_ticker_date,
)

logger = logging.getLogger("mark_johnson.market_scanner")


class MarketScanner:
    """Polls Kalshi for open temperature markets and parses them."""

    def __init__(self, kalshi: KalshiClient) -> None:
        self._kalshi = kalshi
        # Current market state keyed by ticker
        self.markets: dict[str, Market] = {}

    async def scan(self) -> list[Market]:
        """Fetch all open temperature markets, parse, and return them."""
        raw_markets = await self._kalshi.get_open_temperature_markets()
        parsed: list[Market] = []

        for raw in raw_markets:
            try:
                market = self._parse_market(raw)
                if market:
                    parsed.append(market)
                    self.markets[market.ticker] = market
            except Exception as exc:
                logger.error(
                    "Failed to parse market %s: %s",
                    raw.get("ticker", "unknown"),
                    exc,
                )

        # Remove stale tickers (no longer open)
        current_tickers = {m.ticker for m in parsed}
        stale = [t for t in self.markets if t not in current_tickers]
        for t in stale:
            del self.markets[t]

        logger.info(
            "Market scan complete — %d temperature markets parsed", len(parsed)
        )
        return parsed

    @staticmethod
    def _parse_market(raw: dict[str, Any]) -> Market | None:
        """Parse a raw Kalshi market dict into a Market dataclass."""
        title = raw.get("title", "")
        subtitle = raw.get("subtitle", "")
        event_ticker = raw.get("event_ticker", "")
        ticker = raw.get("ticker", "")

        # Match city
        combined_text = f"{title} {subtitle} {event_ticker}"
        city_key = config.city_key_from_text(combined_text)
        if not city_key:
            logger.debug("Could not match city for market: %s", title)
            return None

        # Detect market type (high vs low temp)
        market_type = detect_market_type(combined_text)

        # Parse temperature band
        band_min, band_max = parse_temperature_band(title, subtitle)
        if band_min is None and band_max is None:
            logger.debug("Could not parse temperature band: %s", title)
            return None

        # Debug log for band parsing verification
        logger.debug(
            "Parsed %s: title='%s' → band=(%s, %s) city=%s type=%s",
            ticker, title,
            band_min, band_max,
            city_key, market_type,
        )

        # Extract pricing
        yes_price = raw.get("yes_price", 0) or 0
        no_price = raw.get("no_price", 0) or 0
        yes_bid = raw.get("yes_bid", 0) or 0
        yes_ask = raw.get("yes_ask", 0) or 0

        # Kalshi prices are in cents (0-100 scale representing probability)
        # Convert to 0-1 scale
        if yes_bid > 0 and yes_ask > 0:
            implied_prob = (yes_bid + yes_ask) / 2.0 / 100.0
            best_bid = yes_bid / 100.0
            best_ask = yes_ask / 100.0
        elif yes_price > 0:
            implied_prob = yes_price / 100.0
            best_bid = implied_prob
            best_ask = implied_prob
        else:
            # Use last trade or skip
            last_price = raw.get("last_price", 0) or 0
            if last_price > 0:
                implied_prob = last_price / 100.0
            else:
                implied_prob = 0.5  # unknown
            best_bid = implied_prob
            best_ask = implied_prob

        # Volume (Kalshi reports volume in number of contracts; approximate $ volume)
        volume = float(raw.get("volume", 0) or 0)
        # Some endpoints report dollar_volume directly
        dollar_volume = float(raw.get("dollar_volume", 0) or 0)
        if dollar_volume > 0:
            volume = dollar_volume

        # Close time
        close_time_str = raw.get("close_time") or raw.get("expiration_time", "")
        try:
            close_time = datetime.fromisoformat(
                close_time_str.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            close_time = datetime.now(timezone.utc)

        # Parse settlement date from ticker (e.g. KXHIGHNY-26FEB17-T44 → 2026-02-17)
        market_date = parse_ticker_date(ticker)

        return Market(
            ticker=ticker,
            city=city_key,
            market_type=market_type,
            band_min=band_min,
            band_max=band_max,
            implied_prob=implied_prob,
            best_bid=best_bid,
            best_ask=best_ask,
            volume=volume,
            close_time=close_time,
            raw_title=title,
            event_ticker=event_ticker,
            market_date=market_date,
        )
