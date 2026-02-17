from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import config
from models.forecast import TemperatureDistribution
from models.market import Market
from models.signal import Signal

logger = logging.getLogger("mark_johnson.signal_engine")


class SignalEngine:
    """Compares model probabilities to market-implied probabilities to find edges."""

    def __init__(self) -> None:
        # Persistence tracking: ticker → consecutive edge count
        self._edge_counts: dict[str, int] = {}
        # Cooldown tracking: city_key → last alert timestamp (monotonic)
        self._last_alert_time: dict[str, float] = {}

    async def scan_for_signals(
        self,
        markets: list[Market],
        distributions: dict[tuple[str, str, str], TemperatureDistribution],
    ) -> list[Signal]:
        """Compare each market against the forecast distribution and return signals."""
        signals: list[Signal] = []
        seen_tickers: set[str] = set()

        for market in markets:
            seen_tickers.add(market.ticker)

            # Look up matching distribution by (city, type, date)
            date_str = market.market_date.isoformat() if market.market_date else ""
            dist = distributions.get((market.city, market.market_type, date_str))
            if dist is None:
                continue

            # Compute model probability for this band
            model_prob = dist.probability_for_band(market.band_min, market.band_max)
            edge = model_prob - market.implied_prob

            # Apply filters
            if not self._passes_filters(market, dist, edge):
                # Reset persistence if edge has disappeared
                self._edge_counts.pop(market.ticker, None)
                continue

            # Check persistence
            self._edge_counts[market.ticker] = (
                self._edge_counts.get(market.ticker, 0) + 1
            )
            if self._edge_counts[market.ticker] < config.EDGE_PERSIST_COUNT:
                logger.debug(
                    "%s edge=%.1f%% — persistence %d/%d",
                    market.ticker,
                    edge * 100,
                    self._edge_counts[market.ticker],
                    config.EDGE_PERSIST_COUNT,
                )
                continue

            # Check cooldown
            if self._is_on_cooldown(market.city):
                logger.debug(
                    "%s on cooldown — skipping alert for %s",
                    market.city,
                    market.ticker,
                )
                continue

            edge_class = Signal.classify_edge(edge)

            signal = Signal(
                market=market,
                model_prob=model_prob,
                edge=edge,
                edge_class=edge_class,
                forecast_mean=dist.mean,
                forecast_std=dist.std,
                confidence=dist.confidence,
                sources=dist.sources,
            )

            if edge_class == "EXTREME":
                logger.warning(
                    "EXTREME edge %.1f%% on %s — potential data error?",
                    edge * 100,
                    market.ticker,
                )

            signals.append(signal)
            self._last_alert_time[market.city] = time.monotonic()

            logger.info(
                "SIGNAL: %s %s | model=%.1f%% market=%.1f%% edge=%.1f%% (%s)",
                market.city,
                market.band_label,
                model_prob * 100,
                market.implied_prob * 100,
                edge * 100,
                edge_class,
            )

        # Clean up persistence for tickers no longer in the market list
        stale = [t for t in self._edge_counts if t not in seen_tickers]
        for t in stale:
            del self._edge_counts[t]

        return signals

    @staticmethod
    def _passes_filters(
        market: Market,
        dist: TemperatureDistribution,
        edge: float,
    ) -> bool:
        """Apply all signal filters. Returns True if the signal should proceed."""
        # Minimum edge
        if abs(edge) < config.MIN_EDGE_PERCENT / 100.0:
            return False

        # Minimum volume
        if market.volume < config.MIN_VOLUME:
            return False

        # Minimum time to close
        now = datetime.now(timezone.utc)
        minutes_to_close = (market.close_time - now).total_seconds() / 60.0
        if minutes_to_close < config.MIN_TIME_TO_CLOSE_MINUTES:
            return False

        # Ensemble spread check (model confidence)
        if dist.std > config.MAX_ENSEMBLE_SPREAD_F:
            logger.debug(
                "Suppressed %s — ensemble spread %.1f°F exceeds threshold",
                market.ticker,
                dist.std,
            )
            return False

        return True

    def _is_on_cooldown(self, city_key: str) -> bool:
        """Check whether the given city is still in alert cooldown."""
        last = self._last_alert_time.get(city_key)
        if last is None:
            return False
        elapsed_minutes = (time.monotonic() - last) / 60.0
        return elapsed_minutes < config.ALERT_COOLDOWN_MINUTES
