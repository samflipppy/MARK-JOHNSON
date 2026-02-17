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
    print(f"\n{'=' * 60}")
    print(f"  Weather API Smoke Test — {city['name']} ({city_key})")
    print(f"  Coordinates: {lat}, {lon}")
    print(f"{'=' * 60}\n")

    async with aiohttp.ClientSession(
        headers={"User-Agent": "MarkJohnson/1.0 (temperature-scanner)"}
    ) as session:
        openmeteo = OpenMeteoClient(session=session)
        nws = NWSClient(session=session)

        # ── Open-Meteo Ensemble ──────────────────────────────────────
        print("[1/2] Open-Meteo Ensemble API...")
        try:
            data = await openmeteo.get_ensemble_forecast(lat, lon)
            max_members = data.get("max_members", [])
            min_members = data.get("min_members", [])
            if max_members or min_members:
                print(f"  OK  — {len(max_members)} high-temp members, "
                      f"{len(min_members)} low-temp members")
                if max_members:
                    lo, hi = min(max_members), max(max_members)
                    avg = sum(max_members) / len(max_members)
                    print(f"       High temps: {lo:.1f}°F – {hi:.1f}°F  (avg {avg:.1f}°F)")
                if min_members:
                    lo, hi = min(min_members), max(min_members)
                    avg = sum(min_members) / len(min_members)
                    print(f"       Low temps:  {lo:.1f}°F – {hi:.1f}°F  (avg {avg:.1f}°F)")
                # Show which model keys came back
                model_keys = [k for k in data if k not in ("max_members", "min_members")]
                if model_keys:
                    print(f"       Models: {', '.join(sorted(model_keys))}")
            else:
                print("  WARN — API returned data but no temperature members found")
                print(f"       Keys returned: {list(data.keys())}")
        except Exception as exc:
            print(f"  FAIL — {exc}")

        print()

        # ── NWS Grid Forecast ────────────────────────────────────────
        print("[2/2] NWS Grid Forecast API...")
        try:
            data = await nws.get_forecast(lat, lon)
            max_f = data.get("max_temp_f")
            min_f = data.get("min_temp_f")
            if max_f is not None or min_f is not None:
                parts = []
                if max_f is not None:
                    parts.append(f"high {max_f:.1f}°F")
                if min_f is not None:
                    parts.append(f"low {min_f:.1f}°F")
                print(f"  OK  — {', '.join(parts)}")
            else:
                print("  WARN — API returned OK but no temperature values")
                print(f"       Raw response: {data}")
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
