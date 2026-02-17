#!/usr/bin/env python3
"""
MARK JOHNSON v1.0 — Autonomous temperature market scanner and alert engine.

Continuously polls Kalshi temperature markets, builds weather forecast
probability distributions, detects pricing edges, and dispatches Discord alerts.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone

import aiohttp

import config
from services.alert_dispatcher import AlertDispatcher
from services.logger import log_forecast, log_market_snapshot, log_signal
from services.market_scanner import MarketScanner
from services.signal_engine import SignalEngine
from services.weather_engine import WeatherEngine
from utils.discord_client import DiscordWebhookClient
from utils.kalshi_client import KalshiClient
from utils.weather_client import NWSClient, OpenMeteoClient

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mark_johnson")

# ── Shared state ──────────────────────────────────────────────────────────────
_shutdown_event = asyncio.Event()
_data_updated = asyncio.Event()
_signals_today: list = []


def _print_banner() -> None:
    n_cities = len(config.CITIES)
    discord_status = "ENABLED" if config.DISCORD_WEBHOOK_URL else "DISABLED (no webhook URL)"
    banner = f"""
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551         MARK JOHNSON v1.0            \u2551
\u2551  Temperature Market Scanner          \u2551
\u2551  Monitoring {n_cities} cities{' ' * (16 - len(str(n_cities)))}\u2551
\u2551  Discord alerts: {discord_status:<20s}\u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d"""
    print(banner)


# ── Loop tasks ────────────────────────────────────────────────────────────────


async def market_scan_loop(scanner: MarketScanner) -> None:
    """Poll Kalshi for temperature markets every MARKET_POLL_INTERVAL_SECONDS."""
    while not _shutdown_event.is_set():
        try:
            markets = await scanner.scan()
            log_market_snapshot(markets)
            _data_updated.set()
            logger.info("Market scan complete — %d markets", len(markets))
        except Exception as exc:
            logger.error("Market scan loop error: %s", exc)

        try:
            await asyncio.wait_for(
                _shutdown_event.wait(),
                timeout=config.MARKET_POLL_INTERVAL_SECONDS,
            )
            break  # shutdown requested
        except asyncio.TimeoutError:
            pass


async def weather_refresh_loop(engine: WeatherEngine) -> None:
    """Refresh weather forecasts every WEATHER_POLL_INTERVAL_SECONDS."""
    while not _shutdown_event.is_set():
        try:
            distributions = await engine.refresh_all()
            for (city, _), dist in distributions.items():
                log_forecast(city, dist)
            _data_updated.set()
            logger.info("Weather refresh complete — %d distributions", len(distributions))
        except Exception as exc:
            logger.error("Weather refresh loop error: %s", exc)

        try:
            await asyncio.wait_for(
                _shutdown_event.wait(),
                timeout=config.WEATHER_POLL_INTERVAL_SECONDS,
            )
            break
        except asyncio.TimeoutError:
            pass


async def signal_scan_loop(
    scanner: MarketScanner,
    weather: WeatherEngine,
    signal_engine: SignalEngine,
    dispatcher: AlertDispatcher,
) -> None:
    """Run signal detection whenever market or weather data updates."""
    while not _shutdown_event.is_set():
        try:
            await asyncio.wait_for(_data_updated.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            continue

        _data_updated.clear()

        if not scanner.markets or not weather.distributions:
            continue

        try:
            signals = await signal_engine.scan_for_signals(
                list(scanner.markets.values()),
                weather.distributions,
            )

            for sig in signals:
                log_signal(sig)
                _signals_today.append(sig)

            if signals:
                sent = await dispatcher.send_batch_alerts(signals)
                logger.info("Dispatched %d/%d signal alerts", sent, len(signals))
        except Exception as exc:
            logger.error("Signal scan loop error: %s", exc)


async def heartbeat_loop(dispatcher: AlertDispatcher) -> None:
    """Send a heartbeat message every 60 minutes."""
    while not _shutdown_event.is_set():
        try:
            await asyncio.wait_for(
                _shutdown_event.wait(),
                timeout=3600.0,  # 60 minutes
            )
            break
        except asyncio.TimeoutError:
            pass

        try:
            stats = {
                "Signals today": len(_signals_today),
                "Uptime": "OK",
                "Time": datetime.now(timezone.utc).strftime("%H:%M UTC"),
            }
            await dispatcher.send_heartbeat(stats)
            logger.info("Heartbeat sent")
        except Exception as exc:
            logger.error("Heartbeat loop error: %s", exc)


async def daily_summary_loop(dispatcher: AlertDispatcher) -> None:
    """Send a daily summary at 23:59 UTC (approximate end-of-day)."""
    while not _shutdown_event.is_set():
        now = datetime.now(timezone.utc)
        # Schedule for 23:59 UTC today (or tomorrow if past that)
        target = now.replace(hour=23, minute=59, second=0, microsecond=0)
        if now >= target:
            # Already past today's summary time — wait until tomorrow
            target = target.replace(day=target.day + 1)

        wait_seconds = (target - now).total_seconds()
        try:
            await asyncio.wait_for(
                _shutdown_event.wait(),
                timeout=max(wait_seconds, 1.0),
            )
            break
        except asyncio.TimeoutError:
            pass

        try:
            await dispatcher.send_daily_summary(_signals_today)
            _signals_today.clear()
            logger.info("Daily summary sent")
        except Exception as exc:
            logger.error("Daily summary loop error: %s", exc)


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    _print_banner()

    # Set up graceful shutdown
    loop = asyncio.get_running_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, lambda: _shutdown_event.set())

    # Create shared HTTP session
    async with aiohttp.ClientSession(
        headers={"User-Agent": "MarkJohnson/1.0 (temperature-scanner)"}
    ) as session:
        # Initialize clients
        kalshi = KalshiClient(session=session)
        openmeteo = OpenMeteoClient(session=session)
        nws = NWSClient(session=session)
        discord = DiscordWebhookClient(session=session)

        # Initialize services
        scanner = MarketScanner(kalshi)
        weather = WeatherEngine(openmeteo, nws)
        signal_engine = SignalEngine()
        dispatcher = AlertDispatcher(discord)

        logger.info("All services initialized — starting loops")

        # Run all loops concurrently
        tasks = [
            asyncio.create_task(market_scan_loop(scanner), name="market_scan"),
            asyncio.create_task(weather_refresh_loop(weather), name="weather_refresh"),
            asyncio.create_task(
                signal_scan_loop(scanner, weather, signal_engine, dispatcher),
                name="signal_scan",
            ),
            asyncio.create_task(heartbeat_loop(dispatcher), name="heartbeat"),
            asyncio.create_task(daily_summary_loop(dispatcher), name="daily_summary"),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Shutting down gracefully...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("MARK JOHNSON stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested via Ctrl+C")
        sys.exit(0)
