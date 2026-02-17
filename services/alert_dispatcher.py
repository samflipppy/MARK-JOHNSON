from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from models.signal import Signal
from utils.discord_client import DiscordWebhookClient

logger = logging.getLogger("mark_johnson.alert_dispatcher")


def _format_time_remaining(close_time: datetime) -> str:
    """Format a human-readable time remaining string."""
    now = datetime.now(timezone.utc)
    delta = close_time - now
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return "CLOSED"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


class AlertDispatcher:
    """Formats signals into Discord embeds and dispatches them."""

    def __init__(self, discord: DiscordWebhookClient) -> None:
        self._discord = discord

    async def send_signal_alert(self, signal: Signal) -> bool:
        """Send a single signal alert to Discord."""
        market = signal.market

        # Color coding by edge class
        color_map = {
            "MODERATE": 0x3B82F6,  # blue
            "STRONG": 0xF59E0B,  # amber
            "EXTREME": 0xEF4444,  # red
        }
        color = color_map.get(signal.edge_class, 0x3B82F6)

        # Edge direction
        edge_sign = "+" if signal.edge > 0 else ""
        edge_label = f"YES" if signal.edge > 0 else "NO"

        embed: dict[str, Any] = {
            "title": f"\U0001f321\ufe0f MARK JOHNSON \u2014 {market.city}",
            "description": f"Edge detected on **{market.raw_title}**",
            "color": color,
            "fields": [
                {
                    "name": "Band",
                    "value": market.band_label,
                    "inline": True,
                },
                {
                    "name": "Market Implied",
                    "value": f"{market.implied_prob:.0%}",
                    "inline": True,
                },
                {
                    "name": "Model Probability",
                    "value": f"{signal.model_prob:.0%}",
                    "inline": True,
                },
                {
                    "name": "Edge",
                    "value": f"{edge_sign}{signal.edge:.0%} ({signal.edge_class}) — lean {edge_label}",
                    "inline": True,
                },
                {
                    "name": "Forecast Mean",
                    "value": f"{signal.forecast_mean:.1f}\u00b0F (\u00b1{signal.forecast_std:.1f}\u00b0F)",
                    "inline": True,
                },
                {
                    "name": "Confidence",
                    "value": signal.confidence,
                    "inline": True,
                },
                {
                    "name": "Band Position",
                    "value": signal.band_position.upper(),
                    "inline": True,
                },
                {
                    "name": "Kelly Size",
                    "value": f"{signal.kelly_contracts:.1f} contracts" if signal.kelly_contracts > 0 else "SKIP",
                    "inline": True,
                },
                {
                    "name": "Expected Value",
                    "value": f"${signal.expected_value:+.3f}/contract",
                    "inline": True,
                },
                {
                    "name": "Volume",
                    "value": f"${market.volume:,.0f}",
                    "inline": True,
                },
                {
                    "name": "Time to Close",
                    "value": _format_time_remaining(market.close_time),
                    "inline": True,
                },
            ],
            "footer": {
                "text": "MARK JOHNSON v2.0 | Not financial advice | Paper signals only"
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        success = await self._discord.send_embed(embed)
        if success:
            logger.info("Alert sent for %s %s", market.city, market.ticker)
        else:
            logger.error("Failed to send alert for %s %s", market.city, market.ticker)
        return success

    async def send_batch_alerts(self, signals: list[Signal]) -> int:
        """Send alerts for a list of signals. Returns count of successful sends."""
        sent = 0
        for signal in signals:
            if await self.send_signal_alert(signal):
                sent += 1
        return sent

    async def send_daily_summary(
        self,
        signals_today: list[Signal],
        outcomes: dict[str, str] | None = None,
    ) -> bool:
        """Send an end-of-day summary embed."""
        if not signals_today and not outcomes:
            return await self._discord.send_embed(
                {
                    "title": "\U0001f4ca MARK JOHNSON \u2014 Daily Summary",
                    "description": "No signals generated today.",
                    "color": 0x6B7280,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        fields: list[dict[str, Any]] = [
            {
                "name": "Signals Generated",
                "value": str(len(signals_today)),
                "inline": True,
            },
        ]

        # Breakdown by edge class
        for cls in ("MODERATE", "STRONG", "EXTREME"):
            count = sum(1 for s in signals_today if s.edge_class == cls)
            if count:
                fields.append(
                    {"name": cls, "value": str(count), "inline": True}
                )

        # Cities covered
        cities = sorted({s.market.city for s in signals_today})
        if cities:
            fields.append(
                {"name": "Cities", "value": ", ".join(cities), "inline": False}
            )

        if outcomes:
            outcome_lines = [
                f"{ticker}: {result}" for ticker, result in outcomes.items()
            ]
            fields.append(
                {
                    "name": "Outcomes",
                    "value": "\n".join(outcome_lines[:10]),  # cap at 10
                    "inline": False,
                }
            )

        embed = {
            "title": "\U0001f4ca MARK JOHNSON \u2014 Daily Summary",
            "color": 0x10B981,
            "fields": fields,
            "footer": {"text": "MARK JOHNSON v1.0"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return await self._discord.send_embed(embed)

    async def send_heartbeat(self, stats: dict[str, Any] | None = None) -> bool:
        """Send an hourly heartbeat to confirm the system is alive."""
        description = "\U00002705 System is running."
        if stats:
            parts = [f"**{k}:** {v}" for k, v in stats.items()]
            description += "\n" + " | ".join(parts)

        embed = {
            "title": "\U0001f493 MARK JOHNSON \u2014 Heartbeat",
            "description": description,
            "color": 0x6B7280,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return await self._discord.send_embed(embed)
