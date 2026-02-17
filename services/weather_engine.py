from __future__ import annotations

import asyncio
import logging
import statistics
from datetime import date
from typing import Any

import config
from models.forecast import TemperatureDistribution
from utils.weather_client import NWSClient, OpenMeteoClient

logger = logging.getLogger("mark_johnson.weather_engine")


class WeatherEngine:
    """Pulls forecasts from multiple sources and builds temperature distributions."""

    def __init__(
        self,
        openmeteo: OpenMeteoClient,
        nws: NWSClient,
    ) -> None:
        self._openmeteo = openmeteo
        self._nws = nws
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

        # Pull both sources concurrently — both now return date-keyed data
        ensemble_by_date, nws_by_date = await asyncio.gather(
            self._openmeteo.get_ensemble_forecast(lat, lon, timezone=tz_name),
            self._nws.get_forecast(lat, lon, timezone=tz_name),
            return_exceptions=True,
        )

        if isinstance(ensemble_by_date, Exception):
            logger.error("Open-Meteo failed for %s: %s", city_key, ensemble_by_date)
            ensemble_by_date = {}
        if isinstance(nws_by_date, Exception):
            logger.error("NWS failed for %s: %s", city_key, nws_by_date)
            nws_by_date = {}

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
            )
            if high_dist:
                distributions.append(("high_temp", high_dist))

            # Build low-temp distribution
            low_dist = self._fit_distribution(
                city_key, "low_temp", ensemble_data, nws_data,
                temp_field="min", forecast_date=forecast_date,
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
    ) -> TemperatureDistribution | None:
        """
        Combine ensemble members + NWS point forecast into a distribution.

        NWS is blended into the mean at NWS_BLEND_WEIGHT (default 15%) rather
        than being a single data point lost among 143 ensemble members.
        """

        sources: dict[str, list[float]] = {}

        # Collect ensemble members
        member_key = f"{temp_field}_members"
        members = ensemble_data.get(member_key, [])
        if members:
            sources["open_meteo_ensemble"] = members

        # Collect per-model data for logging
        for key, vals in ensemble_data.items():
            if key.startswith(f"{temp_field}_") and key != member_key:
                sources[key] = vals

        # Get NWS point forecast
        nws_field = f"{temp_field}_temp_f"
        nws_val = nws_data.get(nws_field) if isinstance(nws_data, dict) else None
        if nws_val is not None:
            sources["nws"] = [nws_val]

        if not members and nws_val is None:
            logger.warning("No forecast data for %s (%s)", city_key, market_type)
            return None

        # Compute ensemble statistics
        ensemble_mean = statistics.mean(members) if members else None
        ensemble_std = (
            statistics.stdev(members) if members and len(members) > 1 else None
        )

        # Blend NWS into ensemble mean (NWS is high-quality but single-valued)
        if ensemble_mean is not None and nws_val is not None:
            w = config.NWS_BLEND_WEIGHT
            blended_mean = (1 - w) * ensemble_mean + w * nws_val
            blended_std = ensemble_std if ensemble_std is not None else 2.0

            # Cross-validation: warn if NWS and ensemble disagree significantly
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
            # NWS only — no ensemble data
            blended_mean = nws_val
            blended_std = 3.0  # conservative uncertainty for single-source forecast

        dist = TemperatureDistribution(
            city=city_key,
            mean=blended_mean,
            std=blended_std,
            member_values=members if members else ([nws_val] if nws_val is not None else []),
            sources=sources,
            forecast_date=forecast_date,
        )

        date_tag = forecast_date.isoformat() if forecast_date else "?"
        logger.info(
            "%s %s [%s]: mean=%.1f°F ±%.1f°F (%d members, confidence=%s)",
            city_key,
            market_type,
            date_tag,
            dist.mean,
            dist.std,
            len(dist.member_values),
            dist.confidence,
        )

        return dist

    def get_distribution(
        self, city_key: str, market_type: str, date_str: str = ""
    ) -> TemperatureDistribution | None:
        """Retrieve the latest distribution for a city + market type + date."""
        return self.distributions.get((city_key, market_type, date_str))
