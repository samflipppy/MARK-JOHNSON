from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

import config

logger = logging.getLogger("mark_johnson.weather")

# NWS grid URL cache: (lat, lon) → forecast grid data URL
_nws_grid_cache: dict[tuple[float, float], str] = {}


class OpenMeteoClient:
    """Async client for the Open-Meteo ensemble forecast API."""

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._session = session
        self._owns_session = session is None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "MarkJohnson/1.0 (temperature-scanner)"}
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def get_ensemble_forecast(
        self, lat: float, lon: float
    ) -> dict[str, list[float]]:
        """
        Fetch ensemble forecast members for the given location.

        Returns dict mapping model name → list of temperature values (°F)
        for today's max and min temps across all ensemble members.
        """
        session = await self._ensure_session()
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min",
            "temperature_unit": "fahrenheit",
            "models": "icon_seamless,gfs_seamless,ecmwf_ifs025,gem_global,meteofrance_seamless",
            "forecast_days": 1,
        }

        try:
            async with session.get(
                config.OPENMETEO_ENSEMBLE_URL, params=params
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error("Open-Meteo request failed: %s", exc)
            return {}

        return self._parse_ensemble_response(data)

    @staticmethod
    def _parse_ensemble_response(data: dict[str, Any]) -> dict[str, list[float]]:
        """
        Parse the Open-Meteo ensemble JSON into {model_name: [temp_values]}.

        The API returns data per model, each with multiple ensemble members.
        The daily key contains arrays of values per member for each day.
        """
        result: dict[str, list[float]] = {}

        # The response nests data under each model name
        # Two possible structures: flat daily or per-model
        daily = data.get("daily", {})

        if not daily:
            return result

        # Open-Meteo ensemble returns member data like:
        # "daily": { "temperature_2m_max_member0": [...], "temperature_2m_max_member1": [...], ... }
        # or under model-specific keys.
        # Collect all temperature max/min values from all member keys.
        for key, values in daily.items():
            if not isinstance(values, list):
                continue
            if "temperature_2m_max" in key:
                model_key = key.replace("temperature_2m_max", "max").strip("_")
                if not model_key:
                    model_key = "max_default"
                vals = [v for v in values if v is not None]
                if vals:
                    result.setdefault("max_members", []).extend(vals)
                    result[f"max_{model_key}"] = vals
            elif "temperature_2m_min" in key:
                model_key = key.replace("temperature_2m_min", "min").strip("_")
                if not model_key:
                    model_key = "min_default"
                vals = [v for v in values if v is not None]
                if vals:
                    result.setdefault("min_members", []).extend(vals)
                    result[f"min_{model_key}"] = vals

        # If the response is a simple array (single model, no member suffix)
        for field_name in ("temperature_2m_max", "temperature_2m_min"):
            if field_name in daily:
                vals = daily[field_name]
                if isinstance(vals, list):
                    clean = [v for v in vals if v is not None]
                    if clean:
                        bucket = "max_members" if "max" in field_name else "min_members"
                        result.setdefault(bucket, []).extend(clean)

        return result


class NWSClient:
    """Async client for the National Weather Service API."""

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._session = session
        self._owns_session = session is None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "MarkJohnson/1.0 (contact@markjohnson.app)",
                    "Accept": "application/geo+json",
                }
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def _get_grid_url(self, lat: float, lon: float) -> str | None:
        """Get the forecast grid data URL for the given coordinates (cached)."""
        cache_key = (round(lat, 4), round(lon, 4))
        if cache_key in _nws_grid_cache:
            return _nws_grid_cache[cache_key]

        session = await self._ensure_session()
        url = f"{config.NWS_API_BASE}/points/{lat:.4f},{lon:.4f}"

        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning("NWS points request failed: %d", resp.status)
                    return None
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error("NWS points request error: %s", exc)
            return None

        grid_url = data.get("properties", {}).get("forecastGridData")
        if grid_url:
            _nws_grid_cache[cache_key] = grid_url
        return grid_url

    async def get_forecast(
        self, lat: float, lon: float
    ) -> dict[str, float | None]:
        """
        Fetch the NWS grid forecast and extract today's max/min temps.

        Returns {"max_temp_f": float|None, "min_temp_f": float|None}.
        """
        grid_url = await self._get_grid_url(lat, lon)
        if not grid_url:
            return {"max_temp_f": None, "min_temp_f": None}

        session = await self._ensure_session()

        try:
            async with session.get(grid_url) as resp:
                if resp.status != 200:
                    logger.warning("NWS grid request failed: %d", resp.status)
                    return {"max_temp_f": None, "min_temp_f": None}
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error("NWS grid request error: %s", exc)
            return {"max_temp_f": None, "min_temp_f": None}

        props = data.get("properties", {})
        max_temp = self._extract_first_value(props.get("maxTemperature", {}))
        min_temp = self._extract_first_value(props.get("minTemperature", {}))

        # NWS grid data is in Celsius — convert to Fahrenheit
        max_f = (max_temp * 9 / 5 + 32) if max_temp is not None else None
        min_f = (min_temp * 9 / 5 + 32) if min_temp is not None else None

        return {"max_temp_f": max_f, "min_temp_f": min_f}

    @staticmethod
    def _extract_first_value(field: dict[str, Any]) -> float | None:
        """Extract the first numeric value from an NWS gridded data field."""
        values = field.get("values", [])
        if values and isinstance(values, list):
            val = values[0].get("value")
            if val is not None:
                return float(val)
        return None
