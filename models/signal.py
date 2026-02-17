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

    # v2 fields — simulation-driven additions
    band_position: str = "center"  # "tail", "shoulder", "center"
    effective_threshold: float = 0.08  # what edge threshold was applied
    kelly_contracts: float = 0.0  # recommended position size (contracts)

    @staticmethod
    def classify_edge(edge: float) -> str:
        abs_edge = abs(edge)
        if abs_edge >= 0.20:
            return "EXTREME"
        if abs_edge >= 0.12:
            return "STRONG"
        return "MODERATE"

    @property
    def direction(self) -> str:
        """BUY_YES if model says higher, BUY_NO if model says lower."""
        return "BUY_YES" if self.edge > 0 else "BUY_NO"

    @property
    def expected_value(self) -> float:
        """Expected profit per $1 contract (before fees).

        EV = model_prob * win_amount - (1 - model_prob) * loss_amount
        """
        if self.edge > 0:
            # BUY YES at implied_prob price
            win_amount = 1.0 - self.market.implied_prob
            loss_amount = self.market.implied_prob
            return self.model_prob * win_amount - (1.0 - self.model_prob) * loss_amount
        else:
            # BUY NO at (1 - implied_prob) price
            win_amount = self.market.implied_prob
            loss_amount = 1.0 - self.market.implied_prob
            p_no = 1.0 - self.model_prob
            return p_no * win_amount - self.model_prob * loss_amount
