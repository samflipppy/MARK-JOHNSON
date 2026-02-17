from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Market:
    """Parsed representation of a single Kalshi temperature-band contract."""

    ticker: str
    city: str  # key into config.CITIES (e.g. "NYC")
    market_type: str  # "high_temp" or "low_temp"
    band_min: float | None  # None for "X° or below" contracts
    band_max: float | None  # None for "X° or above" contracts
    implied_prob: float  # midpoint of best bid/ask, or last-trade fallback
    best_bid: float
    best_ask: float
    volume: float  # dollar volume
    close_time: datetime
    raw_title: str
    event_ticker: str = ""

    @property
    def band_label(self) -> str:
        if self.band_min is None and self.band_max is not None:
            return f"{self.band_max:.0f}\u00b0F or below"
        if self.band_max is None and self.band_min is not None:
            return f"{self.band_min:.0f}\u00b0F or above"
        if self.band_min is not None and self.band_max is not None:
            return f"{self.band_min:.0f}\u00b0\u2013{self.band_max:.0f}\u00b0F"
        return "unknown band"
