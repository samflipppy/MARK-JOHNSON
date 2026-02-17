"""
Model Output Statistics (MOS) bias tracker.

Tracks forecast-vs-actual temperature errors per city and per model.
Persists a running exponentially-weighted bias to disk (JSON).
The bias is then used to correct future forecasts.

This is the same technique the NWS uses in their MOS system:
  - Record what each model predicted
  - Compare to actual observed temperature
  - Maintain a running bias estimate
  - Apply the bias as a correction to future forecasts

The bias adapts over time as model performance changes seasonally.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger("mark_johnson.bias_tracker")

# Default file location for bias data
BIAS_FILE = Path(__file__).parent.parent / "data" / "bias_history.json"


class BiasTracker:
    """
    Track and persist forecast bias per city/model for MOS corrections.

    Bias is stored as an exponentially-weighted moving average (EWMA)
    so that recent performance is weighted more heavily than old data,
    allowing the tracker to adapt to seasonal model drift.
    """

    def __init__(self, filepath: Path | str = BIAS_FILE) -> None:
        self._filepath = Path(filepath)
        # In-memory bias data
        # Structure: {city: {market_type: {model_or_aggregate: BiasRecord}}}
        self._data: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        self._load()

    def _load(self) -> None:
        """Load bias history from disk."""
        if self._filepath.exists():
            try:
                with open(self._filepath, "r") as f:
                    self._data = json.load(f)
                logger.info(
                    "Loaded bias history from %s (%d cities)",
                    self._filepath,
                    len(self._data),
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load bias history: %s", exc)
                self._data = {}
        else:
            logger.info("No bias history file found — starting fresh")

    def _save(self) -> None:
        """Persist bias data to disk."""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._filepath, "w") as f:
                json.dump(self._data, f, indent=2)
        except OSError as exc:
            logger.error("Could not save bias history: %s", exc)

    def record_verification(
        self,
        city_key: str,
        market_type: str,  # "high_temp" or "low_temp"
        forecast_mean: float,
        actual_temp: float,
        model_biases: dict[str, float] | None = None,
        forecast_date: str = "",
    ) -> None:
        """
        Record a forecast verification (forecast vs actual).

        Parameters:
            city_key:       City identifier
            market_type:    "high_temp" or "low_temp"
            forecast_mean:  What the blended forecast predicted (°F)
            actual_temp:    What actually happened (°F)
            model_biases:   Per-model errors {model_name: forecast - actual}
            forecast_date:  ISO date string for logging
        """
        error = forecast_mean - actual_temp  # positive = too warm

        # Update aggregate bias for this city + market type
        city_data = self._data.setdefault(city_key, {})
        type_data = city_data.setdefault(market_type, {})
        self._update_ewma(type_data, "aggregate", error)

        # Update per-model biases
        if model_biases:
            for model_name, model_error in model_biases.items():
                self._update_ewma(type_data, model_name, model_error)

        self._save()

        logger.info(
            "VERIFICATION %s %s [%s]: forecast=%.1f°F actual=%.1f°F "
            "error=%+.1f°F running_bias=%+.2f°F",
            city_key, market_type, forecast_date,
            forecast_mean, actual_temp, error,
            self.get_bias(city_key, market_type),
        )

    @staticmethod
    def _update_ewma(
        type_data: dict[str, dict[str, Any]],
        key: str,
        new_error: float,
        alpha: float | None = None,
    ) -> None:
        """
        Update the exponentially-weighted moving average for a bias record.

        Uses config.BIAS_EWMA_ALPHA (default 0.25) for faster convergence.
        At alpha=0.25, ~50% of the weight is on the last ~2.4 observations,
        allowing rapid adaptation to model drift while still smoothing noise.
        """
        if alpha is None:
            alpha = getattr(config, "BIAS_EWMA_ALPHA", 0.25)
        record = type_data.get(key, {})
        old_bias = record.get("bias", 0.0)
        n = record.get("n_samples", 0)

        if n == 0:
            new_bias = new_error
        else:
            new_bias = alpha * new_error + (1 - alpha) * old_bias

        # Also track MAE for confidence estimation
        old_mae = record.get("mae", abs(new_error))
        new_mae = alpha * abs(new_error) + (1 - alpha) * old_mae

        type_data[key] = {
            "bias": round(new_bias, 3),
            "mae": round(new_mae, 3),
            "n_samples": n + 1,
            "last_error": round(new_error, 2),
            "last_updated": datetime.utcnow().isoformat(),
        }

    def get_bias(
        self,
        city_key: str,
        market_type: str,
        model_name: str = "aggregate",
    ) -> float:
        """
        Get the current running bias for a city/type/model.

        Returns 0.0 if no data available.
        Positive bias = model runs warm → should subtract from forecast.
        """
        try:
            record = self._data[city_key][market_type][model_name]
            n = record.get("n_samples", 0)
            if n < config.BIAS_MIN_SAMPLES:
                return 0.0  # not enough data to be confident
            return record.get("bias", 0.0)
        except KeyError:
            return 0.0

    def get_model_bias(self, city_key: str, market_type: str, model_name: str) -> float:
        """Get bias for a specific model. Returns 0.0 if insufficient data."""
        return self.get_bias(city_key, market_type, model_name)

    def get_mae(
        self,
        city_key: str,
        market_type: str,
        model_name: str = "aggregate",
    ) -> float:
        """Get the running Mean Absolute Error for confidence estimation."""
        try:
            record = self._data[city_key][market_type][model_name]
            return record.get("mae", 3.0)  # default 3°F MAE
        except KeyError:
            return 3.0

    def get_spread_inflation(
        self, city_key: str, market_type: str
    ) -> float:
        """
        Compute an EMOS-style spread inflation factor from historical MAE.

        If the model's MAE is larger than the typical ensemble spread,
        the spread should be inflated to account for real-world uncertainty.

        Returns a multiplier (>1.0 means inflate spread).
        """
        mae = self.get_mae(city_key, market_type)
        # Typical ensemble spread is ~2°F.  If MAE > 2, we need more spread.
        # inflation = max(1.0, MAE / typical_spread)
        typical_spread = 2.0
        inflation = max(1.0, mae / typical_spread)
        return min(inflation, config.EMOS_MAX_INFLATION)

    def get_all_biases(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return the full bias dataset for debugging/inspection."""
        return self._data

    async def verify_from_metar(
        self,
        metar_client: Any,
        distributions: dict[tuple[str, str, str], Any],
    ) -> int:
        """
        Check if any of yesterday's forecasts can be verified using today's METAR.

        For high_temp markets, we compare the forecasted max to the actual
        observed max from the previous day's METAR observations.

        Returns the number of verifications performed.
        """
        from datetime import timedelta
        yesterday = (datetime.utcnow() - timedelta(days=1)).date()
        yesterday_str = yesterday.isoformat()
        verified = 0

        for (city_key, market_type, date_str), dist in distributions.items():
            if date_str != yesterday_str:
                continue

            station = config.CITIES.get(city_key, {}).get("station", "")
            if not station:
                continue

            # For now, use the latest METAR as a proxy for the actual temp.
            # A proper implementation would fetch historical METARs for the
            # entire previous day and extract the actual max/min.
            # TODO: Replace with historical METAR query for actual max/min
            obs = await metar_client.get_observation(station)
            if obs is None:
                continue

            # We only have the current temp, not yesterday's max/min.
            # Skip this for now — proper implementation needs a historical API.
            # This is a placeholder for the verification pipeline.

        return verified
