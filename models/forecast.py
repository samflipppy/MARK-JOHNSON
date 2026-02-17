from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from scipy import stats


@dataclass
class TemperatureDistribution:
    """Probability distribution built from ensemble forecast members."""

    city: str  # key into config.CITIES
    mean: float  # degrees F
    std: float  # degrees F
    member_values: list[float] = field(default_factory=list)
    sources: dict[str, list[float]] = field(default_factory=dict)
    low_confidence: bool = False
    forecast_date: date | None = None  # local date this forecast targets

    def __post_init__(self) -> None:
        # Floor unreasonably small std
        if self.std < 0.5:
            self.std = 1.0
        # Flag unreasonably large std
        if self.std > 8.0:
            self.low_confidence = True

    @property
    def distribution(self) -> stats.norm:
        return stats.norm(loc=self.mean, scale=self.std)

    def probability_for_band(
        self, band_min: float | None, band_max: float | None
    ) -> float:
        """
        Compute P(band_min <= T < band_max) using the fitted normal CDF.

        - band_min=None  → "X° or below"   → P(T < band_max)
        - band_max=None  → "X° or above"   → P(T >= band_min)
        - both set       → P(band_min <= T < band_max)
        """
        d = self.distribution
        if band_min is None and band_max is not None:
            return float(d.cdf(band_max))
        if band_max is None and band_min is not None:
            return float(1.0 - d.cdf(band_min))
        if band_min is not None and band_max is not None:
            return float(d.cdf(band_max) - d.cdf(band_min))
        return 0.0

    @property
    def confidence(self) -> str:
        if self.low_confidence:
            return "LOW"
        if self.std <= 2.0:
            return "HIGH"
        return "MEDIUM"
