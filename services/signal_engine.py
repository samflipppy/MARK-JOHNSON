from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone

import config
from models.forecast import TemperatureDistribution
from models.market import Market
from models.signal import Signal

logger = logging.getLogger("mark_johnson.signal_engine")


class SignalEngine:
    """Compares model probabilities to market-implied probabilities to find edges.

    Simulation-driven improvements (v2):
      - Tiered edge thresholds by band position (tail < shoulder < center)
      - Bid-ask spread filter (don't trade wide books)
      - Confidence-gated thresholds (MEDIUM/LOW need bigger edges)
      - Nowcast-aware boosting (lower threshold when METAR confirms drift)
      - Kelly criterion position sizing
    """

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

            # Determine band position for tiered thresholds
            band_pos = self._classify_band_position(market, dist)

            # Compute the effective edge threshold for this specific signal
            effective_threshold = self._effective_edge_threshold(
                band_pos, dist.confidence, dist.bias_correction_f,
            )

            # Apply filters
            if not self._passes_filters(market, dist, edge, effective_threshold):
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

            # Compute Kelly criterion position sizing
            kelly_contracts = self._kelly_size(model_prob, market.implied_prob, edge)

            signal = Signal(
                market=market,
                model_prob=model_prob,
                edge=edge,
                edge_class=edge_class,
                forecast_mean=dist.mean,
                forecast_std=dist.std,
                confidence=dist.confidence,
                sources=dist.sources,
                band_position=band_pos,
                effective_threshold=effective_threshold,
                kelly_contracts=kelly_contracts,
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
                "SIGNAL: %s %s | model=%.1f%% market=%.1f%% edge=%.1f%% (%s) "
                "band=%s conf=%s kelly=%.1f threshold=%.1f%%",
                market.city,
                market.band_label,
                model_prob * 100,
                market.implied_prob * 100,
                edge * 100,
                edge_class,
                band_pos,
                dist.confidence,
                kelly_contracts,
                effective_threshold * 100,
            )

        # Clean up persistence for tickers no longer in the market list
        stale = [t for t in self._edge_counts if t not in seen_tickers]
        for t in stale:
            del self._edge_counts[t]

        return signals

    @staticmethod
    def _classify_band_position(
        market: Market, dist: TemperatureDistribution
    ) -> str:
        """Classify a market's band as 'tail', 'shoulder', or 'center'
        relative to the forecast distribution.

        This drives tiered edge thresholds — tails need smaller edges
        because our KDE captures tail probabilities better than the market.
        """
        # Open-ended bands are always tails
        if market.band_min is None or market.band_max is None:
            return "tail"

        band_mid = (market.band_min + market.band_max) / 2.0
        distance_from_mean = abs(band_mid - dist.mean)

        # Classify based on distance in units of std
        if dist.std > 0:
            sigma_distance = distance_from_mean / dist.std
        else:
            sigma_distance = distance_from_mean / 2.0

        if sigma_distance <= 0.75:
            return "center"
        elif sigma_distance <= 1.5:
            return "shoulder"
        else:
            return "tail"

    @staticmethod
    def _effective_edge_threshold(
        band_position: str,
        confidence: str,
        nowcast_correction: float,
    ) -> float:
        """Compute the dynamic edge threshold based on band position,
        confidence, and whether nowcasting is active.

        This is the key simulation-driven insight: different situations
        warrant different thresholds.
        """
        # Base threshold by band position
        if band_position == "tail":
            threshold = config.MIN_EDGE_TAIL_PERCENT / 100.0
        elif band_position == "shoulder":
            threshold = config.MIN_EDGE_SHOULDER_PERCENT / 100.0
        else:
            threshold = config.MIN_EDGE_CENTER_PERCENT / 100.0

        # Override with confidence-gated threshold if stricter
        if confidence == "LOW":
            conf_threshold = config.MIN_EDGE_LOW_CONFIDENCE_PERCENT / 100.0
            threshold = max(threshold, conf_threshold)
        elif confidence == "MEDIUM":
            conf_threshold = config.MIN_EDGE_MEDIUM_CONFIDENCE_PERCENT / 100.0
            threshold = max(threshold, conf_threshold)

        # Discount when nowcasting correction is active (higher conviction)
        if abs(nowcast_correction) > 0.5:
            threshold *= config.NOWCAST_ACTIVE_EDGE_DISCOUNT

        return threshold

    @staticmethod
    def _passes_filters(
        market: Market,
        dist: TemperatureDistribution,
        edge: float,
        effective_threshold: float,
    ) -> bool:
        """Apply all signal filters. Returns True if the signal should proceed."""
        # Dynamic edge threshold (replaces flat MIN_EDGE_PERCENT)
        if abs(edge) < effective_threshold:
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

        # ── Bid-ask spread filter ──────────────────────────────────────
        # Simulation showed fees destroy marginal edges.
        # Skip wide books where execution cost eats the edge.
        spread = market.best_ask - market.best_bid
        if spread > config.MAX_BID_ASK_SPREAD:
            logger.debug(
                "Suppressed %s — bid-ask spread $%.2f exceeds $%.2f threshold",
                market.ticker,
                spread,
                config.MAX_BID_ASK_SPREAD,
            )
            return False

        # ── Narrow-band sanity check ───────────────────────────────────
        # A single 1-2°F narrow band should NEVER have >50% implied prob.
        # If it does, the market is almost certainly cumulative (e.g. "56°F
        # or below") being mis-parsed as a narrow range ("56-57°F").
        # Suppress these to avoid false EXTREME signals.
        if market.band_min is not None and market.band_max is not None:
            band_width = market.band_max - market.band_min
            if band_width <= 3.0 and market.implied_prob > 0.50:
                logger.warning(
                    "SUPPRESSED %s — narrow %.0f°F band '%s' at %.0f%% implied "
                    "prob is suspicious (likely cumulative market mis-parsed as "
                    "narrow band). Raw title: '%s'",
                    market.ticker, band_width, market.band_label,
                    market.implied_prob * 100, market.raw_title,
                )
                return False

        # ── Skip dust-level probability bands ──────────────────────────
        # When both model and market agree the probability is tiny (<2%),
        # any percentage edge is misleading (e.g. 1% vs 3% = "200% edge")
        if dist.probability_for_band(market.band_min, market.band_max) < 0.02:
            if market.implied_prob < 0.02:
                return False

        return True

    @staticmethod
    def _kelly_size(
        model_prob: float,
        implied_prob: float,
        edge: float,
    ) -> float:
        """Compute recommended position size using fractional Kelly criterion.

        Kelly fraction f* = (p * b - q) / b
        where p = model probability of winning, b = payout odds, q = 1-p

        We use quarter-Kelly for safety — full Kelly has enormous variance.
        """
        if edge > 0:
            # BUY YES: win (1 - implied_prob), lose implied_prob
            p = model_prob
            b = (1.0 - implied_prob) / implied_prob if implied_prob > 0.01 else 99.0
        else:
            # BUY NO: win implied_prob, lose (1 - implied_prob)
            p = 1.0 - model_prob
            b = implied_prob / (1.0 - implied_prob) if implied_prob < 0.99 else 99.0

        q = 1.0 - p
        kelly_full = (p * b - q) / b if b > 0 else 0.0
        kelly_full = max(0.0, kelly_full)  # never negative (never bet if -EV)

        kelly_frac = kelly_full * config.KELLY_FRACTION
        contracts = kelly_frac * config.BANKROLL

        return min(contracts, config.KELLY_MAX_CONTRACTS)

    def _is_on_cooldown(self, city_key: str) -> bool:
        """Check whether the given city is still in alert cooldown."""
        last = self._last_alert_time.get(city_key)
        if last is None:
            return False
        elapsed_minutes = (time.monotonic() - last) / 60.0
        return elapsed_minutes < config.ALERT_COOLDOWN_MINUTES
