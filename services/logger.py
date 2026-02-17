from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.forecast import TemperatureDistribution
from models.market import Market
from models.signal import Signal

logger = logging.getLogger("mark_johnson.logger")

_BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"


def _ensure_dir(subdir: str) -> Path:
    path = _BASE_DIR / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _append_jsonl(subdir: str, record: dict[str, Any]) -> None:
    """Append a single JSON record to today's JSONL file in the given subdirectory."""
    dir_path = _ensure_dir(subdir)
    filepath = dir_path / f"{_today_str()}.jsonl"
    try:
        with open(filepath, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:
        logger.error("Failed to write log to %s: %s", filepath, exc)


def log_forecast(
    city: str,
    distribution: TemperatureDistribution,
    timestamp: datetime | None = None,
) -> None:
    """Log a forecast distribution to data/logs/forecasts/."""
    ts = timestamp or datetime.now(timezone.utc)
    record = {
        "timestamp": ts.isoformat(),
        "city": city,
        "forecast_date": distribution.forecast_date.isoformat() if distribution.forecast_date else None,
        "mean": distribution.mean,
        "std": distribution.std,
        "confidence": distribution.confidence,
        "member_count": len(distribution.member_values),
        "sources": {
            k: v if len(v) <= 10 else [v[0], f"...({len(v)} values)", v[-1]]
            for k, v in distribution.sources.items()
        },
    }
    _append_jsonl("forecasts", record)


def log_market_snapshot(
    markets: list[Market],
    timestamp: datetime | None = None,
) -> None:
    """Log a batch of market snapshots to data/logs/markets/."""
    ts = timestamp or datetime.now(timezone.utc)
    record = {
        "timestamp": ts.isoformat(),
        "market_count": len(markets),
        "markets": [
            {
                "ticker": m.ticker,
                "city": m.city,
                "type": m.market_type,
                "market_date": m.market_date.isoformat() if m.market_date else None,
                "band_min": m.band_min,
                "band_max": m.band_max,
                "implied_prob": round(m.implied_prob, 4),
                "best_bid": round(m.best_bid, 4),
                "best_ask": round(m.best_ask, 4),
                "volume": m.volume,
                "close_time": m.close_time.isoformat(),
                "title": m.raw_title,
            }
            for m in markets
        ],
    }
    _append_jsonl("markets", record)


def log_signal(
    signal: Signal,
    timestamp: datetime | None = None,
) -> None:
    """Log a generated signal to data/logs/signals/."""
    ts = timestamp or datetime.now(timezone.utc)
    record = {
        "timestamp": ts.isoformat(),
        "ticker": signal.market.ticker,
        "city": signal.market.city,
        "market_type": signal.market.market_type,
        "market_date": signal.market.market_date.isoformat() if signal.market.market_date else None,
        "band": signal.market.band_label,
        "implied_prob": round(signal.market.implied_prob, 4),
        "model_prob": round(signal.model_prob, 4),
        "edge": round(signal.edge, 4),
        "edge_class": signal.edge_class,
        "forecast_mean": round(signal.forecast_mean, 1),
        "forecast_std": round(signal.forecast_std, 2),
        "confidence": signal.confidence,
        "volume": signal.market.volume,
    }
    _append_jsonl("signals", record)
