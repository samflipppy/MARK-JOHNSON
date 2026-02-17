from __future__ import annotations

import asyncio
import logging
import statistics
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
        # Latest distributions keyed by (city_key, "high_temp"/"low_temp")
        self.distributions: dict[tuple[str, str], TemperatureDistribution] = {}

    async def refresh_all(self) -> dict[tuple[str, str], TemperatureDistribution]:
        """Pull forecasts for every configured city and build distributions."""
        tasks = [
            self._build_distribution(city_key)
            for city_key in config.CITIES
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for city_key, result in zip(config.CITIES, results):
            if isinstance(result, Exception):
                logger.error("Failed to build distribution for %s: %s", city_key, result)
                continue
            if result:
                for market_type, dist in result:
                    self.distributions[(dist.city, market_type)] = dist

        logger.info(
            "Weather engine refreshed — %d distributions available",
            len(self.distributions),
        )
        return self.distributions

    async def _build_distribution(
        self, city_key: str
    ) -> list[tuple[str, TemperatureDistribution]]:
        """Build high and low temp distributions for a single city."""
        city = config.CITIES[city_key]
        lat, lon = city["lat"], city["lon"]

        # Pull both sources concurrently
        ensemble_data, nws_data = await asyncio.gather(
            self._openmeteo.get_ensemble_forecast(lat, lon),
            self._nws.get_forecast(lat, lon),
            return_exceptions=True,
        )

        if isinstance(ensemble_data, Exception):
            logger.error("Open-Meteo failed for %s: %s", city_key, ensemble_data)
            ensemble_data = {}
        if isinstance(nws_data, Exception):
            logger.error("NWS failed for %s: %s", city_key, nws_data)
            nws_data = {}

        distributions: list[tuple[str, TemperatureDistribution]] = []

        # Build high-temp distribution
        high_dist = self._fit_distribution(
            city_key, "high_temp", ensemble_data, nws_data, temp_field="max"
        )
        if high_dist:
            distributions.append(("high_temp", high_dist))

        # Build low-temp distribution
        low_dist = self._fit_distribution(
            city_key, "low_temp", ensemble_data, nws_data, temp_field="min"
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
    ) -> TemperatureDistribution | None:
        """Combine ensemble members + NWS point forecast into a distribution."""

        all_values: list[float] = []
        sources: dict[str, list[float]] = {}

        # Collect ensemble members
        member_key = f"{temp_field}_members"
        members = ensemble_data.get(member_key, [])
        if members:
            all_values.extend(members)
            sources["open_meteo_ensemble"] = members

        # Collect per-model data for logging
        for key, vals in ensemble_data.items():
            if key.startswith(f"{temp_field}_") and key != member_key:
                sources[key] = vals

        # Add NWS point forecast
        nws_field = f"{temp_field}_temp_f"
        nws_val = nws_data.get(nws_field) if isinstance(nws_data, dict) else None
        if nws_val is not None:
            all_values.append(nws_val)
            sources["nws"] = [nws_val]

        if not all_values:
            logger.warning("No forecast data for %s (%s)", city_key, market_type)
            return None

        mean = statistics.mean(all_values)
        std = statistics.stdev(all_values) if len(all_values) > 1 else 1.0

        dist = TemperatureDistribution(
            city=city_key,
            mean=mean,
            std=std,
            member_values=all_values,
            sources=sources,
        )

        logger.info(
            "%s %s: mean=%.1f°F ±%.1f°F (%d members, confidence=%s)",
            city_key,
            market_type,
            dist.mean,
            dist.std,
            len(all_values),
            dist.confidence,
        )

        return dist

    def get_distribution(
        self, city_key: str, market_type: str
    ) -> TemperatureDistribution | None:
        """Retrieve the latest distribution for a city + market type."""
        return self.distributions.get((city_key, market_type))
