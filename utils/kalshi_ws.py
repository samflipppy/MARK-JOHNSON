"""
Kalshi WebSocket client for real-time price updates.

Connects to the public ticker channel and streams price/volume changes
for temperature markets. No API key required for public channels.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

import aiohttp

import config

logger = logging.getLogger("mark_johnson.kalshi_ws")

KALSHI_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"


class KalshiWebSocket:
    """Streams real-time ticker updates for temperature markets."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        on_ticker_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._session = session
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._on_ticker_update = on_ticker_update
        self._subscribed_tickers: set[str] = set()
        self._msg_id = 0

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            self._ws = await self._session.ws_connect(
                KALSHI_WS_URL,
                heartbeat=30.0,
                timeout=15.0,
            )
            logger.info("WebSocket connected to Kalshi")
            return True
        except Exception as exc:
            logger.error("WebSocket connection failed: %s", exc)
            return False

    async def subscribe(self, tickers: list[str]) -> None:
        """Subscribe to ticker updates for the given market tickers."""
        if not self._ws or self._ws.closed:
            logger.warning("Cannot subscribe — WebSocket not connected")
            return

        new_tickers = [t for t in tickers if t not in self._subscribed_tickers]
        if not new_tickers:
            return

        msg = {
            "id": self._next_id(),
            "cmd": "subscribe",
            "params": {
                "channels": ["ticker"],
                "market_tickers": new_tickers,
            },
        }
        await self._ws.send_json(msg)
        self._subscribed_tickers.update(new_tickers)
        logger.info("Subscribed to %d tickers (%d total)", len(new_tickers), len(self._subscribed_tickers))

    async def unsubscribe(self, tickers: list[str]) -> None:
        """Unsubscribe from specific tickers."""
        if not self._ws or self._ws.closed:
            return

        msg = {
            "id": self._next_id(),
            "cmd": "unsubscribe",
            "params": {
                "channels": ["ticker"],
                "market_tickers": tickers,
            },
        }
        await self._ws.send_json(msg)
        self._subscribed_tickers -= set(tickers)

    async def listen(self, shutdown_event: asyncio.Event) -> None:
        """
        Read messages from the WebSocket until shutdown.
        Calls on_ticker_update for each ticker message received.
        """
        if not self._ws or self._ws.closed:
            logger.warning("Cannot listen — WebSocket not connected")
            return

        try:
            async for msg in self._ws:
                if shutdown_event.is_set():
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        self._handle_message(data)
                    except json.JSONDecodeError:
                        logger.debug("Non-JSON WS message: %s", msg.data[:200])

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket error: %s", self._ws.exception())
                    break

                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                    logger.info("WebSocket closed by server")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("WebSocket listen error: %s", exc)

    def _handle_message(self, data: dict[str, Any]) -> None:
        """Route incoming WebSocket messages."""
        msg_type = data.get("type")

        if msg_type == "ticker":
            # Price/volume update for a market
            payload = data.get("msg", {})
            if self._on_ticker_update and payload:
                self._on_ticker_update(payload)

        elif msg_type == "error":
            logger.warning("WebSocket error message: %s", data.get("msg"))

        elif msg_type == "subscribed":
            logger.debug("Subscription confirmed: %s", data)

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
            logger.info("WebSocket closed")
