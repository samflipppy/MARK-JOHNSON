#!/usr/bin/env python3
"""
Smoke test for the real weather APIs (Open-Meteo + NWS).

Run:  python test_weather_apis.py

Hits the live endpoints for one city (NYC by default) and prints
the raw results so you can verify the data looks correct.
Pass a city key as an argument to test a different city:
    python test_weather_apis.py CHI
"""
from __future__ import annotations

import asyncio
import sys

import aiohttp

import config
from utils.weather_client import NWSClient, OpenMeteoClient


async def smoke_test(city_key: str) -> None:
    city = config.CITIES[city_key]
    lat, lon = city["lat"], city["lon"]
    tz_name = city.get("tz", "UTC")
    print(f"\n{'=' * 60}")
    print(f"  Weather API Smoke Test — {city['name']} ({city_key})")
    print(f"  Coordinates: {lat}, {lon}")
    print(f"  Timezone: {tz_name}")
    print(f"{'=' * 60}\n")

    async with aiohttp.ClientSession(
        headers={"User-Agent": "MarkJohnson/1.0 (temperature-scanner)"}
    ) as session:
        openmeteo = OpenMeteoClient(session=session)
        nws = NWSClient(session=session)

        # ── Open-Meteo Ensemble ──────────────────────────────────────
        print("[1/2] Open-Meteo Ensemble API...")
        try:
            data_by_date = await openmeteo.get_ensemble_forecast(lat, lon, timezone=tz_name)
            if data_by_date:
                for date_str, day_data in sorted(data_by_date.items()):
                    max_members = day_data.get("max_members", [])
                    min_members = day_data.get("min_members", [])
                    print(f"  OK  [{date_str}] — {len(max_members)} high-temp members, "
                          f"{len(min_members)} low-temp members")
                    if max_members:
                        lo, hi = min(max_members), max(max_members)
                        avg = sum(max_members) / len(max_members)
                        print(f"       High temps: {lo:.1f}°F – {hi:.1f}°F  (avg {avg:.1f}°F)")
                    if min_members:
                        lo, hi = min(min_members), max(min_members)
                        avg = sum(min_members) / len(min_members)
                        print(f"       Low temps:  {lo:.1f}°F – {hi:.1f}°F  (avg {avg:.1f}°F)")
            else:
                print("  WARN — API returned no data")
        except Exception as exc:
            print(f"  FAIL — {exc}")

        print()

        # ── NWS Grid Forecast ────────────────────────────────────────
        print("[2/2] NWS Grid Forecast API...")
        try:
            data_by_date = await nws.get_forecast(lat, lon, timezone=tz_name)
            if data_by_date:
                for date_str, temps in sorted(data_by_date.items()):
                    max_f = temps.get("max_temp_f")
                    min_f = temps.get("min_temp_f")
                    parts = []
                    if max_f is not None:
                        parts.append(f"high {max_f:.1f}°F")
                    if min_f is not None:
                        parts.append(f"low {min_f:.1f}°F")
                    print(f"  OK  [{date_str}] — {', '.join(parts)}")
            else:
                print("  WARN — API returned no data")
        except Exception as exc:
            print(f"  FAIL — {exc}")

        print(f"\n{'=' * 60}")
        print("  Done. If both show OK, the weather APIs are working.")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    city = sys.argv[1].upper() if len(sys.argv) > 1 else "NYC"
    if city not in config.CITIES:
        print(f"Unknown city key: {city}")
        print(f"Available: {', '.join(sorted(config.CITIES))}")
        sys.exit(1)
    asyncio.run(smoke_test(city))
