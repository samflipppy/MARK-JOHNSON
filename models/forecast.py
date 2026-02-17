"""
Temperature probability distribution — supports both Gaussian and KDE.

When sufficient ensemble members are available (>=10), uses Kernel Density
Estimation for more accurate probability estimates that capture skewness,
heavy tails, and multimodality.  Falls back to Gaussian for sparse data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
from scipy import stats

import config


@dataclass
class TemperatureDistribution:
    """Probability distribution built from weighted ensemble forecast members."""

    city: str  # key into config.CITIES
    mean: float  # degrees F (weighted mean after bias correction)
    std: float  # degrees F (weighted std)
    member_values: list[float] = field(default_factory=list)
    member_weights: list[float] = field(default_factory=list)  # per-member model weights
    sources: dict[str, list[float]] = field(default_factory=dict)
    low_confidence: bool = False
    forecast_date: date | None = None
    bias_correction_f: float = 0.0  # nowcasting correction applied (°F)
    cloud_cover: str = ""  # from METAR (for radiation physics)
    observation_temp_f: float | None = None  # current METAR observation

    # Internal KDE (fitted lazily)
    _kde: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Floor unreasonably small std
        if self.std < 0.5:
            self.std = 1.0
        # Flag unreasonably large std
        if self.std > 8.0:
            self.low_confidence = True

        # Build KDE if enough members and KDE is enabled
        if (
            config.USE_KDE
            and len(self.member_values) >= config.KDE_MIN_MEMBERS
        ):
            self._fit_kde()

    def _fit_kde(self) -> None:
        """Fit a weighted Gaussian KDE on the ensemble members."""
        values = np.array(self.member_values, dtype=np.float64)

        if len(self.member_weights) == len(values):
            weights = np.array(self.member_weights, dtype=np.float64)
            weights = weights / weights.sum()  # normalize
        else:
            weights = None

        try:
            kde = stats.gaussian_kde(
                values,
                bw_method="scott",
                weights=weights,
            )
            # Apply bandwidth scaling factor
            kde.set_bandwidth(kde.factor * config.KDE_BANDWIDTH_FACTOR)
            self._kde = kde
        except (np.linalg.LinAlgError, ValueError):
            # Degenerate data — fallback to Gaussian
            self._kde = None

    @property
    def _gaussian(self) -> stats.norm:
        """Fallback Gaussian distribution."""
        return stats.norm(loc=self.mean, scale=self.std)

    @property
    def uses_kde(self) -> bool:
        return self._kde is not None

    def probability_for_band(
        self, band_min: float | None, band_max: float | None
    ) -> float:
        """
        Compute P(band_min <= T < band_max).

        Uses KDE when available (better captures skewness and tails),
        otherwise falls back to Gaussian CDF.
        """
        if self._kde is not None:
            return self._kde_probability(band_min, band_max)
        return self._gaussian_probability(band_min, band_max)

    def _gaussian_probability(
        self, band_min: float | None, band_max: float | None
    ) -> float:
        """Gaussian CDF integration."""
        d = self._gaussian
        if band_min is None and band_max is not None:
            return float(d.cdf(band_max))
        if band_max is None and band_min is not None:
            return float(1.0 - d.cdf(band_min))
        if band_min is not None and band_max is not None:
            return float(d.cdf(band_max) - d.cdf(band_min))
        return 0.0

    def _kde_probability(
        self, band_min: float | None, band_max: float | None
    ) -> float:
        """Numerical integration of the KDE over the band."""
        kde = self._kde
        # Integration bounds — use ±6 sigma as practical infinity
        lo = self.mean - 6 * self.std
        hi = self.mean + 6 * self.std

        if band_min is None and band_max is not None:
            # P(T < band_max)
            result = kde.integrate_box_1d(lo, band_max)
        elif band_max is None and band_min is not None:
            # P(T >= band_min)
            result = 1.0 - kde.integrate_box_1d(lo, band_min)
        elif band_min is not None and band_max is not None:
            # P(band_min <= T < band_max)
            result = kde.integrate_box_1d(band_min, band_max)
        else:
            return 0.0

        # Clip to valid probability range
        return float(max(0.0, min(1.0, result)))

    @property
    def confidence(self) -> str:
        if self.low_confidence:
            return "LOW"
        if self.std <= 2.0:
            return "HIGH"
        return "MEDIUM"

    @property
    def distribution_type(self) -> str:
        """Human-readable description of the distribution method."""
        if self._kde is not None:
            n = len(self.member_values)
            return f"KDE ({n} members)"
        return "Gaussian"
