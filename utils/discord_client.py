from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp

import config

logger = logging.getLogger("mark_johnson.discord")


class DiscordWebhookClient:
    """Rate-limited async Discord webhook sender."""

    MAX_MESSAGES_PER_MINUTE = 30

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._session = session
        self._owns_session = session is None
        self._send_times: list[float] = []  # timestamps of recent sends

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"}
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def _rate_limit(self) -> None:
        """Enforce 30 messages per 60 seconds."""
        now = time.monotonic()
        # Prune old timestamps
        self._send_times = [t for t in self._send_times if now - t < 60.0]
        if len(self._send_times) >= self.MAX_MESSAGES_PER_MINUTE:
            wait = 60.0 - (now - self._send_times[0])
            if wait > 0:
                logger.info("Discord rate limit — waiting %.1fs", wait)
                await asyncio.sleep(wait)

    async def send(self, payload: dict[str, Any]) -> bool:
        """Send a webhook payload. Returns True on success."""
        webhook_url = config.DISCORD_WEBHOOK_URL
        if not webhook_url:
            logger.warning("DISCORD_WEBHOOK_URL not set — skipping alert")
            return False

        await self._rate_limit()
        session = await self._ensure_session()

        try:
            async with session.post(webhook_url, json=payload) as resp:
                self._send_times.append(time.monotonic())
                if resp.status == 429:
                    retry_after = (await resp.json()).get("retry_after", 5.0)
                    logger.warning(
                        "Discord 429 — retrying after %.1fs", retry_after
                    )
                    await asyncio.sleep(retry_after)
                    async with session.post(webhook_url, json=payload) as retry:
                        return retry.status in (200, 204)
                return resp.status in (200, 204)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error("Discord webhook send failed: %s", exc)
            return False

    async def send_embed(self, embed: dict[str, Any]) -> bool:
        """Convenience method to send a single embed."""
        return await self.send({"embeds": [embed]})

    async def send_text(self, content: str) -> bool:
        """Send a plain text message."""
        return await self.send({"content": content})
