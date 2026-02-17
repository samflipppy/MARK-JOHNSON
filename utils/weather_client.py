from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp

import config

logger = logging.getLogger("mark_johnson.weather")

# NWS grid URL cache: (lat, lon) → forecast grid data URL
_nws_grid_cache: dict[tuple[float, float], str] = {}

# Duration parsing for NWS validTime (e.g. "PT14H", "P1D")
_DURATION_H_RE = re.compile(r"(\d+)H")
_DURATION_D_RE = re.compile(r"(\d+)D")


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

    # Class-level semaphore to limit concurrent Open-Meteo requests
    _semaphore: asyncio.Semaphore | None = None

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        if cls._semaphore is None:
            cls._semaphore = asyncio.Semaphore(config.OPENMETEO_MAX_CONCURRENT)
        return cls._semaphore

    async def get_ensemble_forecast(
        self, lat: float, lon: float, timezone: str = "UTC"
    ) -> dict[str, dict[str, list[float]]]:
        """
        Fetch ensemble forecast members for the given location.

        Returns dict mapping date_str → {model_key: [values], "max_members": [...],
        "min_members": [...]} for each forecast day.

        The timezone parameter ensures daily aggregation aligns with the local
        calendar day (critical for correct daily max/min).
        """
        sem = self._get_semaphore()
        async with sem:
            session = await self._ensure_session()
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min",
                "temperature_unit": "fahrenheit",
                "models": "icon_seamless,gfs_seamless,ecmwf_ifs025,gem_global,meteofrance_seamless",
                "forecast_days": config.FORECAST_DAYS,
                "timezone": timezone,
            }

            backoff = 1.0
            for attempt in range(4):
                try:
                    async with session.get(
                        config.OPENMETEO_ENSEMBLE_URL, params=params
                    ) as resp:
                        if resp.status == 429:
                            retry_after = float(resp.headers.get("Retry-After", backoff))
                            logger.warning(
                                "Open-Meteo 429 for (%.2f, %.2f) — backing off %.1fs (attempt %d)",
                                lat, lon, retry_after, attempt + 1,
                            )
                            await asyncio.sleep(retry_after)
                            backoff *= 2
                            continue
                        resp.raise_for_status()
                        data = await resp.json()
                        result = self._parse_ensemble_response(data)
                        for date_str, day_data in result.items():
                            n_max = len(day_data.get("max_members", []))
                            n_min = len(day_data.get("min_members", []))
                            logger.info(
                                "Open-Meteo OK (%.2f, %.2f) [%s]: %d max members, %d min members",
                                lat, lon, date_str, n_max, n_min,
                            )
                        return result
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    if attempt == 3:
                        logger.error("Open-Meteo request failed after retries: %s", exc)
                        return {}
                    logger.warning("Open-Meteo request error (attempt %d): %s", attempt + 1, exc)
                    await asyncio.sleep(backoff)
                    backoff *= 2

            # Small delay between requests to avoid bursts
            await asyncio.sleep(config.OPENMETEO_REQUEST_DELAY)

        return {}

    @staticmethod
    def _parse_ensemble_response(
        data: dict[str, Any],
    ) -> dict[str, dict[str, list[float]]]:
        """
        Parse the Open-Meteo ensemble JSON into per-day member data.

        Returns {date_str: {"max_members": [...], "min_members": [...], ...}}.
        Each date has aggregated member values from all ensemble models.
        """
        daily = data.get("daily", {})
        if not daily:
            return {}

        dates = daily.get("time", [])
        if not dates:
            return {}

        result: dict[str, dict[str, list[float]]] = {}

        for day_idx, date_str in enumerate(dates):
            day_data: dict[str, list[float]] = {}

            for key, values in daily.items():
                if key == "time" or not isinstance(values, list):
                    continue
                if day_idx >= len(values) or values[day_idx] is None:
                    continue

                val = float(values[day_idx])

                if "temperature_2m_max" in key:
                    model_key = key.replace("temperature_2m_max", "max").strip("_")
                    if not model_key:
                        model_key = "max_default"
                    day_data.setdefault("max_members", []).append(val)
                    day_data[f"max_{model_key}"] = [val]
                elif "temperature_2m_min" in key:
                    model_key = key.replace("temperature_2m_min", "min").strip("_")
                    if not model_key:
                        model_key = "min_default"
                    day_data.setdefault("min_members", []).append(val)
                    day_data[f"min_{model_key}"] = [val]

            if day_data:
                result[date_str] = day_data

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
        self, lat: float, lon: float, timezone: str = "UTC"
    ) -> dict[str, dict[str, float | None]]:
        """
        Fetch the NWS grid forecast and extract max/min temps keyed by local date.

        Returns {date_str: {"max_temp_f": float|None, "min_temp_f": float|None}}.
        The timezone is used to convert NWS validTime periods to local dates.
        """
        grid_url = await self._get_grid_url(lat, lon)
        if not grid_url:
            return {}

        session = await self._ensure_session()

        try:
            async with session.get(grid_url) as resp:
                if resp.status != 200:
                    logger.warning("NWS grid request failed: %d", resp.status)
                    return {}
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error("NWS grid request error: %s", exc)
            return {}

        props = data.get("properties", {})
        local_tz = ZoneInfo(timezone)

        max_by_date = self._extract_values_by_date(
            props.get("maxTemperature", {}), local_tz
        )
        min_by_date = self._extract_values_by_date(
            props.get("minTemperature", {}), local_tz
        )

        # Merge into {date: {max_temp_f, min_temp_f}}
        all_dates = set(max_by_date.keys()) | set(min_by_date.keys())
        result: dict[str, dict[str, float | None]] = {}
        for date_str in sorted(all_dates):
            max_c = max_by_date.get(date_str)
            min_c = min_by_date.get(date_str)
            max_f = (max_c * 9 / 5 + 32) if max_c is not None else None
            min_f = (min_c * 9 / 5 + 32) if min_c is not None else None
            result[date_str] = {"max_temp_f": max_f, "min_temp_f": min_f}

        # Log first date for backward-compat visibility
        for date_str, temps in sorted(result.items()):
            max_f = temps.get("max_temp_f")
            min_f = temps.get("min_temp_f")
            logger.info(
                "NWS OK (%.2f, %.2f) [%s]: high=%.1f°F low=%.1f°F",
                lat, lon, date_str,
                max_f if max_f is not None else float("nan"),
                min_f if min_f is not None else float("nan"),
            )

        return result

    @staticmethod
    def _extract_values_by_date(
        field: dict[str, Any], local_tz: ZoneInfo
    ) -> dict[str, float]:
        """
        Extract {local_date_str: value_celsius} from an NWS grid data field.

        Uses the midpoint of each validTime period converted to the local timezone
        to determine which calendar date the value belongs to.
        """
        result: dict[str, float] = {}
        for entry in field.get("values", []):
            val = entry.get("value")
            valid_time = entry.get("validTime", "")
            if val is None or not valid_time:
                continue

            try:
                # Parse "2026-02-17T08:00:00+00:00/PT14H"
                parts = valid_time.split("/")
                time_str = parts[0]
                dt_utc = datetime.fromisoformat(time_str.replace("Z", "+00:00"))

                # Compute midpoint of the period for accurate date assignment
                duration_hours = 12.0  # default
                if len(parts) > 1:
                    dur = parts[1]
                    h_match = _DURATION_H_RE.search(dur)
                    d_match = _DURATION_D_RE.search(dur)
                    hours = 0.0
                    if d_match:
                        hours += int(d_match.group(1)) * 24
                    if h_match:
                        hours += int(h_match.group(1))
                    if hours > 0:
                        duration_hours = hours

                midpoint = dt_utc + timedelta(hours=duration_hours / 2)
                local_date = midpoint.astimezone(local_tz).date().isoformat()

                # Keep only the first (most recent) value per date
                if local_date not in result:
                    result[local_date] = float(val)
            except (ValueError, IndexError, OverflowError):
                continue

        return result
