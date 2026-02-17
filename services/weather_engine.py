"""
Weather Engine — multi-source forecast fusion with physics-based corrections.

Pipeline:
  1. Fetch ensemble forecasts from Open-Meteo (5 global models, ~143 members)
  2. Fetch NWS deterministic forecast
  3. Fetch real-time METAR surface observations
  4. Apply model-specific skill weighting (ECMWF > GFS > ICON > ...)
  5. Blend NWS into ensemble with configurable weight
  6. Apply MOS bias correction from historical verification data
  7. Anchor to climatological prior (Bayesian blend with NOAA normals)
  8. Calibrate ensemble spread (EMOS variance inflation)
  9. Compute nowcasting bias correction via diurnal cycle physics
  10. Shift entire ensemble by combined bias corrections
  11. Fit Kernel Density Estimation on weighted, corrected members
  12. Output a TemperatureDistribution for each city/type/date
"""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

import config
from models.forecast import TemperatureDistribution
from services.bias_tracker import BiasTracker
from services.climatology import (
    bayesian_climo_blend,
    compute_climo_anomaly,
    emos_spread_calibration,
    get_climate_normal,
)
from utils.weather_client import METARClient, NWSClient, OpenMeteoClient

logger = logging.getLogger("mark_johnson.weather_engine")


# ── Diurnal cycle model ──────────────────────────────────────────────────────

def _diurnal_fraction(hour_local: float, forecast_min: float, forecast_max: float) -> float:
    """
    Estimate expected temperature at a given local hour using a truncated
    sinusoidal diurnal cycle model.

    Physics: Solar radiation drives a roughly sinusoidal temperature curve.
    - Minimum occurs near sunrise (~6 AM local).
    - Maximum occurs ~2–3 hours after solar noon (~3 PM local).
    - Warming phase (sunrise→afternoon) is shorter than cooling phase.

    Returns the expected temperature in °F.
    """
    MIN_HOUR = 6.0   # approximate sunrise / daily minimum
    MAX_HOUR = 15.0  # approximate daily maximum (~3 PM)

    amplitude = (forecast_max - forecast_min) / 2.0
    midpoint = (forecast_max + forecast_min) / 2.0

    if MIN_HOUR <= hour_local <= MAX_HOUR:
        # Warming phase: half-cosine from min to max
        progress = (hour_local - MIN_HOUR) / (MAX_HOUR - MIN_HOUR)
        return midpoint - amplitude * math.cos(progress * math.pi)
    else:
        # Cooling phase: half-cosine from max toward next min
        if hour_local > MAX_HOUR:
            hours_past_max = hour_local - MAX_HOUR
        else:
            hours_past_max = (24.0 - MAX_HOUR) + hour_local
        cooling_duration = 24.0 - (MAX_HOUR - MIN_HOUR)  # ~15 hours
        progress = hours_past_max / cooling_duration
        return midpoint + amplitude * math.cos(progress * math.pi)


def _compute_nowcast_bias(
    obs_temp_f: float,
    forecast_min: float,
    forecast_max: float,
    local_hour: float,
) -> float:
    """
    Compare the METAR observation to the diurnal model's expected temperature
    at this hour.  Returns the bias (obs - expected) that can be used to
    correct the forecast distribution.

    Positive bias = reality is warmer than the model expected.
    """
    expected = _diurnal_fraction(local_hour, forecast_min, forecast_max)
    bias = obs_temp_f - expected
    return bias


def _radiation_cooling_adjustment(cloud_cover: str, wind_kt: float) -> float:
    """
    Estimate additional overnight cooling or reduced daytime heating
    based on cloud cover and wind speed (physical radiative transfer).

    - Clear skies + calm winds → strong radiation cooling (lower min)
    - Overcast + windy → minimal radiation cooling
    - Overcast daytime → reduced heating (lower max)

    Returns an adjustment factor (multiplier) for the bias correction weight.
    Higher = more confident in the bias correction.
    """
    # Cloud cover suppresses radiative effects (both cooling and heating)
    cover_factors = {
        "SKC": 1.0,  # clear — full radiative effect
        "CLR": 1.0,
        "FEW": 0.9,
        "SCT": 0.7,  # scattered — moderate suppression
        "BKN": 0.4,  # broken — significant suppression
        "OVC": 0.2,  # overcast — minimal radiative effect
    }
    cloud_factor = cover_factors.get(cloud_cover, 0.5)

    # Wind mixing reduces surface temperature extremes
    # Calm winds allow stronger inversions and radiation effects
    if wind_kt < 5:
        wind_factor = 1.0
    elif wind_kt < 10:
        wind_factor = 0.8
    elif wind_kt < 20:
        wind_factor = 0.5
    else:
        wind_factor = 0.3

    return cloud_factor * wind_factor


# ── Main engine ──────────────────────────────────────────────────────────────

class WeatherEngine:
    """
    Pulls forecasts from multiple sources, applies physics-based corrections,
    and builds probability distributions using KDE.
    """

    def __init__(
        self,
        openmeteo: OpenMeteoClient,
        nws: NWSClient,
        metar: METARClient | None = None,
        bias_tracker: BiasTracker | None = None,
    ) -> None:
        self._openmeteo = openmeteo
        self._nws = nws
        self._metar = metar
        self._bias_tracker = bias_tracker
        # Latest distributions keyed by (city_key, market_type, date_str)
        self.distributions: dict[tuple[str, str, str], TemperatureDistribution] = {}

    async def refresh_all(self) -> dict[tuple[str, str, str], TemperatureDistribution]:
        """Pull forecasts for every configured city and build distributions."""
        tasks = [
            self._build_distributions(city_key)
            for city_key in config.CITIES
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for city_key, result in zip(config.CITIES, results):
            if isinstance(result, Exception):
                logger.error("Failed to build distribution for %s: %s", city_key, result)
                continue
            if result:
                for market_type, dist in result:
                    date_str = dist.forecast_date.isoformat() if dist.forecast_date else ""
                    self.distributions[(dist.city, market_type, date_str)] = dist

        logger.info(
            "Weather engine refreshed — %d distributions available",
            len(self.distributions),
        )
        return self.distributions

    async def _build_distributions(
        self, city_key: str
    ) -> list[tuple[str, TemperatureDistribution]]:
        """Build high and low temp distributions for a single city, per date."""
        city = config.CITIES[city_key]
        lat, lon = city["lat"], city["lon"]
        tz_name = city.get("tz", "UTC")
        station = city.get("station", "")

        # Pull all three sources concurrently
        async def _no_metar() -> None:
            return None

        coros: list = [
            self._openmeteo.get_ensemble_forecast(lat, lon, timezone=tz_name),
            self._nws.get_forecast(lat, lon, timezone=tz_name),
        ]
        if self._metar and config.METAR_ENABLED and station:
            coros.append(self._metar.get_observation(station))
        else:
            coros.append(_no_metar())

        results = await asyncio.gather(*coros, return_exceptions=True)

        ensemble_by_date = results[0] if not isinstance(results[0], Exception) else {}
        if isinstance(results[0], Exception):
            logger.error("Open-Meteo failed for %s: %s", city_key, results[0])

        nws_by_date = results[1] if not isinstance(results[1], Exception) else {}
        if isinstance(results[1], Exception):
            logger.error("NWS failed for %s: %s", city_key, results[1])

        metar_obs = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None

        # Collect all dates from both sources
        all_dates = set(ensemble_by_date.keys()) | set(nws_by_date.keys())

        distributions: list[tuple[str, TemperatureDistribution]] = []

        for date_str in sorted(all_dates):
            ensemble_data = ensemble_by_date.get(date_str, {})
            nws_data = nws_by_date.get(date_str, {})

            forecast_date = None
            try:
                forecast_date = date.fromisoformat(date_str)
            except (ValueError, TypeError):
                pass

            # Build high-temp distribution
            high_dist = self._fit_distribution(
                city_key, "high_temp", ensemble_data, nws_data,
                temp_field="max", forecast_date=forecast_date,
                metar_obs=metar_obs, tz_name=tz_name,
                bias_tracker=self._bias_tracker,
            )
            if high_dist:
                distributions.append(("high_temp", high_dist))

            # Build low-temp distribution
            low_dist = self._fit_distribution(
                city_key, "low_temp", ensemble_data, nws_data,
                temp_field="min", forecast_date=forecast_date,
                metar_obs=metar_obs, tz_name=tz_name,
                bias_tracker=self._bias_tracker,
            )
            if low_dist:
                distributions.append(("low_temp", low_dist))

        return distributions

    @staticmethod
    def _fit_distribution(
        city_key: str,
        market_type: str,
        ensemble_data: dict[str, Any],
        nws_data: dict[str, Any],
        temp_field: str,  # "max" or "min"
        forecast_date: date | None = None,
        metar_obs: dict | None = None,
        tz_name: str = "UTC",
        bias_tracker: BiasTracker | None = None,
    ) -> TemperatureDistribution | None:
        """
        Build a distribution through a 9-step pipeline:
        1. Collect model-weighted ensemble members
        2. Blend NWS point forecast
        3. Apply MOS historical bias correction
        4. Anchor to climatological prior (Bayesian)
        5. Calibrate ensemble spread (EMOS)
        6. Compute nowcasting bias via diurnal physics
        7. Shift members by combined corrections
        8. Fit KDE on the corrected, weighted members
        9. Log diagnostics
        """

        sources: dict[str, list[float]] = {}
        member_values: list[float] = []
        member_weights: list[float] = []

        # ── Step 1: Collect ensemble members with model-specific weights ─────
        raw_members = ensemble_data.get(f"{temp_field}_members", [])
        model_labels = ensemble_data.get(f"{temp_field}_model_labels", [])

        for i, val in enumerate(raw_members):
            model_name = model_labels[i] if i < len(model_labels) else "unknown"
            weight = config.MODEL_WEIGHTS.get(model_name, config.MODEL_WEIGHT_DEFAULT)
            member_values.append(val)
            member_weights.append(weight)

        if raw_members:
            sources["open_meteo_ensemble"] = raw_members

        # Store per-model data for logging
        for key, vals in ensemble_data.items():
            if key.startswith(f"{temp_field}_") and key not in (
                f"{temp_field}_members", f"{temp_field}_model_labels"
            ):
                sources[key] = vals

        # ── Step 2: Get NWS point forecast ───────────────────────────────────
        nws_field = f"{temp_field}_temp_f"
        nws_val = nws_data.get(nws_field) if isinstance(nws_data, dict) else None
        if nws_val is not None:
            sources["nws"] = [nws_val]

        if not member_values and nws_val is None:
            logger.warning("No forecast data for %s (%s)", city_key, market_type)
            return None

        # ── Step 3: Compute weighted ensemble statistics ─────────────────────
        if member_values:
            w = np.array(member_weights, dtype=np.float64)
            v = np.array(member_values, dtype=np.float64)
            w_norm = w / w.sum()
            ensemble_mean = float(np.average(v, weights=w_norm))
            ensemble_std = float(np.sqrt(np.average((v - ensemble_mean) ** 2, weights=w_norm)))
            if ensemble_std < 0.01:
                ensemble_std = 1.0  # degenerate
        else:
            ensemble_mean = None
            ensemble_std = None

        # ── Step 4: Blend NWS into the mean ──────────────────────────────────
        if ensemble_mean is not None and nws_val is not None:
            nws_w = config.NWS_BLEND_WEIGHT
            blended_mean = (1 - nws_w) * ensemble_mean + nws_w * nws_val
            blended_std = ensemble_std if ensemble_std is not None else 2.0

            delta = abs(ensemble_mean - nws_val)
            if delta > 5.0:
                logger.warning(
                    "%s %s: NWS (%.1f°F) and ensemble (%.1f°F) disagree by %.1f°F",
                    city_key, market_type, nws_val, ensemble_mean, delta,
                )
        elif ensemble_mean is not None:
            blended_mean = ensemble_mean
            blended_std = ensemble_std if ensemble_std is not None else 2.0
        else:
            blended_mean = nws_val
            blended_std = 3.0

        # ── Step 5: MOS bias correction (historical model error) ─────────────
        mos_correction = 0.0
        if bias_tracker is not None and config.BIAS_TRACKER_ENABLED:
            mos_bias = bias_tracker.get_bias(city_key, market_type)
            if abs(mos_bias) > 0.05:
                mos_correction = -mos_bias  # subtract warm bias, add cold bias
                mos_correction = max(
                    -config.BIAS_MAX_CORRECTION_F,
                    min(config.BIAS_MAX_CORRECTION_F, mos_correction),
                )
                logger.info(
                    "%s %s MOS: historical bias=%+.2f°F → correction=%+.2f°F",
                    city_key, market_type, mos_bias, mos_correction,
                )

        # ── Step 6: Climatological prior (Bayesian anchoring) ────────────────
        climo_correction = 0.0
        if forecast_date is not None and config.CLIMO_PRIOR_WEIGHT > 0:
            climo_field = "high" if market_type == "high_temp" else "low"
            normal = get_climate_normal(city_key, forecast_date.month, climo_field)

            if normal is not None:
                climo_mean, climo_std = normal

                # Check for extreme anomaly
                anomaly = compute_climo_anomaly(
                    blended_mean + mos_correction,
                    city_key, forecast_date.month, climo_field,
                )
                if anomaly is not None and abs(anomaly) > config.CLIMO_ANOMALY_WARN_SIGMA:
                    logger.warning(
                        "%s %s: forecast %.1f°F is %.1fσ from climo normal %.1f°F",
                        city_key, market_type,
                        blended_mean + mos_correction, anomaly, climo_mean,
                    )

                # Bayesian blend: pull toward climatology
                pre_climo = blended_mean + mos_correction
                post_mean, post_std = bayesian_climo_blend(
                    pre_climo, blended_std,
                    climo_mean, climo_std,
                    config.CLIMO_PRIOR_WEIGHT,
                )
                climo_correction = post_mean - pre_climo

                if abs(climo_correction) > 0.1:
                    logger.debug(
                        "%s %s climo anchor: %.1f°F → %.1f°F (pull %+.1f°F toward %.1f°F)",
                        city_key, market_type, pre_climo, post_mean,
                        climo_correction, climo_mean,
                    )

        # ── Step 7: EMOS spread calibration ──────────────────────────────────
        if blended_std is not None:
            # Base inflation from config
            calibrated_std = emos_spread_calibration(
                blended_std, config.EMOS_INFLATION_FACTOR
            )
            # Additional inflation from historical MAE (if available)
            if bias_tracker is not None:
                tracker_inflation = bias_tracker.get_spread_inflation(city_key, market_type)
                if tracker_inflation > 1.0:
                    calibrated_std = emos_spread_calibration(
                        calibrated_std, tracker_inflation / config.EMOS_INFLATION_FACTOR
                    )
                    logger.debug(
                        "%s %s EMOS: spread %.2f → %.2f (tracker inflation=%.2f)",
                        city_key, market_type, blended_std, calibrated_std, tracker_inflation,
                    )
            blended_std = calibrated_std

        # ── Step 8: Nowcasting bias correction (diurnal cycle physics) ───────
        nowcast_correction = 0.0
        obs_temp = None
        cloud_cover = ""

        if metar_obs is not None and forecast_date is not None:
            today = datetime.now(tz=ZoneInfo(tz_name)).date()

            # Only apply nowcasting to today's forecast
            if forecast_date == today:
                obs_temp = metar_obs.get("temp_f")
                cloud_cover = metar_obs.get("cloud_cover", "")
                wind_kt = metar_obs.get("wind_kt", 0)

                if obs_temp is not None and nws_val is not None:
                    local_now = datetime.now(tz=ZoneInfo(tz_name))
                    local_hour = local_now.hour + local_now.minute / 60.0

                    nws_data_dict = nws_data if isinstance(nws_data, dict) else {}
                    forecast_max = nws_data_dict.get("max_temp_f", blended_mean + 5)
                    forecast_min = nws_data_dict.get("min_temp_f", blended_mean - 5)

                    if forecast_max is not None and forecast_min is not None:
                        raw_bias = _compute_nowcast_bias(
                            obs_temp, forecast_min, forecast_max, local_hour
                        )
                        rad_factor = _radiation_cooling_adjustment(cloud_cover, wind_kt)

                        correction = (
                            raw_bias
                            * config.NOWCAST_CORRECTION_WEIGHT
                            * rad_factor
                        )
                        correction = max(
                            -config.NOWCAST_MAX_CORRECTION_F,
                            min(config.NOWCAST_MAX_CORRECTION_F, correction),
                        )

                        if abs(correction) > 0.1:
                            nowcast_correction = correction
                            logger.info(
                                "%s %s nowcast: obs=%.1f°F expected=%.1f°F "
                                "raw_bias=%+.1f°F correction=%+.1f°F "
                                "(clouds=%s wind=%.0fkt rad_factor=%.2f)",
                                city_key, market_type, obs_temp,
                                _diurnal_fraction(local_hour, forecast_min, forecast_max),
                                raw_bias, correction,
                                cloud_cover, wind_kt, rad_factor,
                            )

        # ── Step 9: Apply all corrections ────────────────────────────────────
        total_correction = mos_correction + climo_correction + nowcast_correction
        if total_correction != 0.0 and member_values:
            member_values = [v + total_correction for v in member_values]
            blended_mean += total_correction

        # ── Step 10: Build the distribution (KDE or Gaussian) ────────────────
        dist = TemperatureDistribution(
            city=city_key,
            mean=blended_mean,
            std=blended_std,
            member_values=member_values,
            member_weights=member_weights,
            sources=sources,
            forecast_date=forecast_date,
            bias_correction_f=total_correction,
            cloud_cover=cloud_cover,
            observation_temp_f=obs_temp,
        )

        date_tag = forecast_date.isoformat() if forecast_date else "?"
        corrections = []
        if abs(mos_correction) > 0.05:
            corrections.append(f"MOS={mos_correction:+.1f}")
        if abs(climo_correction) > 0.05:
            corrections.append(f"climo={climo_correction:+.1f}")
        if abs(nowcast_correction) > 0.05:
            corrections.append(f"nowcast={nowcast_correction:+.1f}")
        correction_str = f" [{', '.join(corrections)}]" if corrections else ""

        logger.info(
            "%s %s [%s]: mean=%.1f°F ±%.1f°F (%d members, %s, confidence=%s%s)",
            city_key, market_type, date_tag,
            dist.mean, dist.std,
            len(dist.member_values),
            dist.distribution_type,
            dist.confidence,
            correction_str,
        )

        return dist

    def get_distribution(
        self, city_key: str, market_type: str, date_str: str = ""
    ) -> TemperatureDistribution | None:
        """Retrieve the latest distribution for a city + market type + date."""
        return self.distributions.get((city_key, market_type, date_str))
