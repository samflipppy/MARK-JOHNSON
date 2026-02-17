from __future__ import annotations

from dataclasses import dataclass, field
from models.market import Market


@dataclass
class Signal:
    """An actionable edge detected between model forecast and market price."""

    market: Market
    model_prob: float
    edge: float  # model_prob - implied_prob (positive = market underprices YES)
    edge_class: str  # "MODERATE", "STRONG", "EXTREME"
    forecast_mean: float
    forecast_std: float
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    sources: dict[str, list[float]] = field(default_factory=dict)

    @staticmethod
    def classify_edge(edge: float) -> str:
        abs_edge = abs(edge)
        if abs_edge >= 0.20:
            return "EXTREME"
        if abs_edge >= 0.12:
            return "STRONG"
        return "MODERATE"
