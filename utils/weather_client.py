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


# ── METAR surface observations ────────────────────────────────────────────────


class METARClient:
    """Fetch real-time surface observations from aviationweather.gov (no API key)."""

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._session = session
        self._owns_session = session is None
        # Cache: station → (obs_dict, fetch_time)
        self._cache: dict[str, tuple[dict, float]] = {}

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

    async def get_observation(self, station: str) -> dict | None:
        """
        Get the latest METAR observation for an ICAO station (e.g. 'KORD').

        Returns dict with:
            temp_f:      float — current temperature in °F
            dewpoint_f:  float — dewpoint in °F
            wind_kt:     float — wind speed in knots
            cloud_cover: str   — BKN, OVC, SCT, FEW, SKC, CLR
            obs_time:    datetime — observation timestamp
            age_minutes: float — minutes since observation

        Returns None on failure or if observation is too old.
        """
        import time as _time

        # Check cache (don't re-fetch within 5 minutes)
        cached = self._cache.get(station)
        if cached and (_time.time() - cached[1]) < 300:
            return cached[0]

        session = await self._ensure_session()

        try:
            async with session.get(
                config.METAR_API_URL,
                params={"ids": station, "format": "json", "taf": "false"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.debug("METAR fetch failed for %s: HTTP %d", station, resp.status)
                    return None
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.debug("METAR fetch error for %s: %s", station, exc)
            return None

        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        metar = data[0]  # most recent observation

        # Parse temperature (Celsius → Fahrenheit)
        temp_c = metar.get("temp")
        if temp_c is None:
            return None
        temp_f = temp_c * 9.0 / 5.0 + 32.0

        dewpt_c = metar.get("dwpt")
        dewpt_f = (dewpt_c * 9.0 / 5.0 + 32.0) if dewpt_c is not None else None

        wind_kt = float(metar.get("wspd", 0) or 0)

        # Cloud cover — pick the most significant layer
        clouds = metar.get("clouds", [])
        cloud_cover = "SKC"
        cover_rank = {"SKC": 0, "CLR": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4}
        for layer in (clouds if isinstance(clouds, list) else []):
            cover = layer.get("cover", "SKC") if isinstance(layer, dict) else "SKC"
            if cover_rank.get(cover, 0) > cover_rank.get(cloud_cover, 0):
                cloud_cover = cover

        # Observation time
        obs_epoch = metar.get("obsTime")
        if obs_epoch:
            obs_time = datetime.fromtimestamp(obs_epoch, tz=ZoneInfo("UTC"))
            age_minutes = (_time.time() - obs_epoch) / 60.0
        else:
            obs_time = datetime.now(tz=ZoneInfo("UTC"))
            age_minutes = 0.0

        # Reject stale observations
        if age_minutes > config.METAR_MAX_AGE_MINUTES:
            logger.debug(
                "METAR for %s too old (%.0f min > %d min threshold)",
                station, age_minutes, config.METAR_MAX_AGE_MINUTES,
            )
            return None

        result = {
            "temp_f": temp_f,
            "dewpoint_f": dewpt_f,
            "wind_kt": wind_kt,
            "cloud_cover": cloud_cover,
            "obs_time": obs_time,
            "age_minutes": age_minutes,
            "station": station,
        }
        self._cache[station] = (result, _time.time())
        logger.info(
            "METAR %s: %.1f°F (dew=%.1f°F wind=%.0fkt clouds=%s age=%.0fmin)",
            station, temp_f,
            dewpt_f if dewpt_f is not None else float("nan"),
            wind_kt, cloud_cover, age_minutes,
        )
        return result


# ── Open-Meteo ensemble forecast ─────────────────────────────────────────────


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

    # Known model name fragments for weight lookup
    _MODEL_NAMES = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_global", "meteofrance_seamless"]

    @classmethod
    def _identify_model(cls, key: str) -> str:
        """Extract the model name from an Open-Meteo field key."""
        key_lower = key.lower()
        for name in cls._MODEL_NAMES:
            if name in key_lower:
                return name
        return "unknown"

    @staticmethod
    def _parse_ensemble_response(
        data: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """
        Parse the Open-Meteo ensemble JSON into per-day member data.

        Returns {date_str: {
            "max_members": [float, ...],
            "min_members": [float, ...],
            "max_model_labels": [str, ...],   # model name per member
            "min_model_labels": [str, ...],
            ...per-model keys...
        }}.
        """
        daily = data.get("daily", {})
        if not daily:
            return {}

        dates = daily.get("time", [])
        if not dates:
            return {}

        result: dict[str, dict[str, Any]] = {}

        for day_idx, date_str in enumerate(dates):
            day_data: dict[str, Any] = {}

            for key, values in daily.items():
                if key == "time" or not isinstance(values, list):
                    continue
                if day_idx >= len(values) or values[day_idx] is None:
                    continue

                val = float(values[day_idx])

                if "temperature_2m_max" in key:
                    model_name = OpenMeteoClient._identify_model(key)
                    model_key = key.replace("temperature_2m_max", "max").strip("_")
                    if not model_key:
                        model_key = "max_default"
                    day_data.setdefault("max_members", []).append(val)
                    day_data.setdefault("max_model_labels", []).append(model_name)
                    day_data[f"max_{model_key}"] = [val]
                elif "temperature_2m_min" in key:
                    model_name = OpenMeteoClient._identify_model(key)
                    model_key = key.replace("temperature_2m_min", "min").strip("_")
                    if not model_key:
                        model_key = "min_default"
                    day_data.setdefault("min_members", []).append(val)
                    day_data.setdefault("min_model_labels", []).append(model_name)
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
