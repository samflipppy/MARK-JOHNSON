"""
Climatological priors — NOAA 1991-2020 Climate Normals for 20 US cities.

Provides Bayesian anchoring: when the ensemble predicts something extreme
(>3σ from climatology), the prior pulls the forecast back toward historical
norms.  This prevents wild outliers from dominating the probability estimate.

Also provides ensemble calibration (EMOS): raw ensemble spread is typically
too narrow.  We inflate variance using a configurable factor.

Sources:
  - NOAA/NCEI U.S. Climate Normals (1991-2020)
  - https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals
"""
from __future__ import annotations

import logging
import math

import config

logger = logging.getLogger("mark_johnson.climatology")


# ── NOAA 1991-2020 Climate Normals (February) ────────────────────────────────
# city_key → { month: { "high": avg high °F, "low": avg low °F, "high_std": σ, "low_std": σ }}
# Standard deviations estimated from typical interannual variability (~5-8°F).

CLIMATE_NORMALS: dict[str, dict[int, dict[str, float]]] = {
    "NYC": {
        1:  {"high": 39.5, "low": 26.1, "high_std": 7.0, "low_std": 7.5},
        2:  {"high": 42.2, "low": 28.3, "high_std": 7.5, "low_std": 7.8},
        3:  {"high": 50.0, "low": 34.5, "high_std": 8.0, "low_std": 7.5},
        4:  {"high": 62.2, "low": 44.2, "high_std": 7.0, "low_std": 6.0},
        5:  {"high": 72.0, "low": 54.0, "high_std": 6.5, "low_std": 5.5},
        6:  {"high": 80.5, "low": 63.5, "high_std": 5.5, "low_std": 4.5},
        7:  {"high": 85.5, "low": 69.0, "high_std": 5.0, "low_std": 4.0},
        8:  {"high": 84.0, "low": 68.0, "high_std": 5.0, "low_std": 4.0},
        9:  {"high": 76.5, "low": 60.5, "high_std": 5.5, "low_std": 5.0},
        10: {"high": 65.0, "low": 49.5, "high_std": 6.5, "low_std": 6.0},
        11: {"high": 54.0, "low": 40.0, "high_std": 7.0, "low_std": 6.5},
        12: {"high": 43.5, "low": 31.0, "high_std": 7.0, "low_std": 7.0},
    },
    "MIA": {
        1:  {"high": 76.5, "low": 61.5, "high_std": 5.0, "low_std": 5.5},
        2:  {"high": 78.0, "low": 63.0, "high_std": 5.0, "low_std": 5.5},
        3:  {"high": 80.0, "low": 66.0, "high_std": 4.5, "low_std": 4.5},
        4:  {"high": 83.5, "low": 70.0, "high_std": 3.5, "low_std": 3.5},
        5:  {"high": 87.0, "low": 74.0, "high_std": 3.0, "low_std": 3.0},
        6:  {"high": 90.0, "low": 77.0, "high_std": 2.5, "low_std": 2.5},
        7:  {"high": 91.5, "low": 78.5, "high_std": 2.5, "low_std": 2.0},
        8:  {"high": 91.0, "low": 78.5, "high_std": 2.5, "low_std": 2.0},
        9:  {"high": 89.5, "low": 77.0, "high_std": 2.5, "low_std": 2.5},
        10: {"high": 86.0, "low": 73.5, "high_std": 3.5, "low_std": 3.5},
        11: {"high": 81.5, "low": 68.0, "high_std": 4.0, "low_std": 4.5},
        12: {"high": 77.5, "low": 63.5, "high_std": 4.5, "low_std": 5.0},
    },
    "CHI": {
        1:  {"high": 32.0, "low": 18.5, "high_std": 9.0, "low_std": 9.5},
        2:  {"high": 34.5, "low": 19.5, "high_std": 9.5, "low_std": 10.0},
        3:  {"high": 46.0, "low": 29.5, "high_std": 9.0, "low_std": 8.5},
        4:  {"high": 59.0, "low": 39.5, "high_std": 8.0, "low_std": 7.0},
        5:  {"high": 70.5, "low": 49.5, "high_std": 7.0, "low_std": 6.0},
        6:  {"high": 80.5, "low": 60.0, "high_std": 5.5, "low_std": 5.0},
        7:  {"high": 84.5, "low": 65.5, "high_std": 5.0, "low_std": 4.5},
        8:  {"high": 83.0, "low": 64.5, "high_std": 5.0, "low_std": 4.5},
        9:  {"high": 76.0, "low": 56.0, "high_std": 6.5, "low_std": 6.0},
        10: {"high": 62.5, "low": 44.0, "high_std": 7.5, "low_std": 7.0},
        11: {"high": 48.5, "low": 33.5, "high_std": 8.5, "low_std": 8.0},
        12: {"high": 36.0, "low": 23.0, "high_std": 8.5, "low_std": 9.0},
    },
    "DEN": {
        1:  {"high": 45.0, "low": 19.0, "high_std": 9.0, "low_std": 8.5},
        2:  {"high": 46.0, "low": 22.0, "high_std": 9.5, "low_std": 9.0},
        3:  {"high": 53.5, "low": 27.5, "high_std": 9.0, "low_std": 8.0},
        4:  {"high": 60.5, "low": 34.0, "high_std": 8.5, "low_std": 7.5},
        5:  {"high": 69.5, "low": 43.5, "high_std": 7.0, "low_std": 6.0},
        6:  {"high": 81.0, "low": 52.5, "high_std": 6.0, "low_std": 5.0},
        7:  {"high": 88.5, "low": 59.0, "high_std": 5.0, "low_std": 4.0},
        8:  {"high": 86.5, "low": 57.5, "high_std": 5.0, "low_std": 4.5},
        9:  {"high": 78.5, "low": 48.5, "high_std": 7.0, "low_std": 6.0},
        10: {"high": 64.0, "low": 37.0, "high_std": 8.5, "low_std": 8.0},
        11: {"high": 52.0, "low": 26.0, "high_std": 9.0, "low_std": 8.5},
        12: {"high": 44.0, "low": 19.5, "high_std": 8.5, "low_std": 8.5},
    },
    "AUS": {
        1:  {"high": 62.0, "low": 40.0, "high_std": 7.5, "low_std": 7.0},
        2:  {"high": 64.5, "low": 41.5, "high_std": 8.0, "low_std": 7.5},
        3:  {"high": 72.0, "low": 49.0, "high_std": 7.0, "low_std": 6.5},
        4:  {"high": 79.5, "low": 57.0, "high_std": 6.0, "low_std": 5.5},
        5:  {"high": 85.5, "low": 65.0, "high_std": 5.0, "low_std": 4.5},
        6:  {"high": 93.0, "low": 72.0, "high_std": 4.0, "low_std": 3.5},
        7:  {"high": 97.0, "low": 75.0, "high_std": 3.5, "low_std": 3.0},
        8:  {"high": 98.0, "low": 75.0, "high_std": 3.5, "low_std": 3.0},
        9:  {"high": 92.0, "low": 69.0, "high_std": 5.0, "low_std": 4.5},
        10: {"high": 81.5, "low": 58.5, "high_std": 6.5, "low_std": 6.0},
        11: {"high": 70.5, "low": 47.5, "high_std": 7.5, "low_std": 7.0},
        12: {"high": 62.5, "low": 40.5, "high_std": 7.5, "low_std": 7.5},
    },
    "LAX": {
        1:  {"high": 67.0, "low": 49.0, "high_std": 5.0, "low_std": 4.0},
        2:  {"high": 67.0, "low": 49.5, "high_std": 5.0, "low_std": 4.0},
        3:  {"high": 67.5, "low": 51.0, "high_std": 4.5, "low_std": 3.5},
        4:  {"high": 69.5, "low": 53.5, "high_std": 4.0, "low_std": 3.0},
        5:  {"high": 71.0, "low": 57.5, "high_std": 3.5, "low_std": 3.0},
        6:  {"high": 75.0, "low": 61.5, "high_std": 3.5, "low_std": 2.5},
        7:  {"high": 81.0, "low": 65.5, "high_std": 3.5, "low_std": 2.5},
        8:  {"high": 82.5, "low": 66.0, "high_std": 3.5, "low_std": 2.5},
        9:  {"high": 82.0, "low": 64.5, "high_std": 4.0, "low_std": 3.0},
        10: {"high": 77.0, "low": 59.5, "high_std": 4.5, "low_std": 3.5},
        11: {"high": 72.0, "low": 53.0, "high_std": 5.0, "low_std": 4.0},
        12: {"high": 66.5, "low": 48.5, "high_std": 5.0, "low_std": 4.5},
    },
    "PHL": {
        1:  {"high": 40.5, "low": 26.5, "high_std": 7.0, "low_std": 7.5},
        2:  {"high": 43.5, "low": 27.5, "high_std": 7.5, "low_std": 8.0},
        3:  {"high": 52.5, "low": 34.5, "high_std": 8.0, "low_std": 7.5},
        4:  {"high": 64.0, "low": 44.0, "high_std": 7.0, "low_std": 6.0},
        5:  {"high": 74.0, "low": 54.0, "high_std": 6.0, "low_std": 5.0},
        6:  {"high": 83.0, "low": 64.0, "high_std": 5.0, "low_std": 4.0},
        7:  {"high": 87.5, "low": 69.5, "high_std": 4.5, "low_std": 3.5},
        8:  {"high": 85.5, "low": 68.0, "high_std": 4.5, "low_std": 3.5},
        9:  {"high": 78.5, "low": 60.5, "high_std": 5.5, "low_std": 5.0},
        10: {"high": 66.5, "low": 49.0, "high_std": 6.5, "low_std": 6.0},
        11: {"high": 55.0, "low": 39.5, "high_std": 7.0, "low_std": 6.5},
        12: {"high": 44.0, "low": 31.0, "high_std": 7.0, "low_std": 7.0},
    },
    "PHX": {
        1:  {"high": 67.5, "low": 45.5, "high_std": 5.5, "low_std": 4.5},
        2:  {"high": 71.0, "low": 47.5, "high_std": 5.5, "low_std": 5.0},
        3:  {"high": 77.0, "low": 52.5, "high_std": 5.5, "low_std": 4.5},
        4:  {"high": 85.5, "low": 59.0, "high_std": 5.0, "low_std": 4.0},
        5:  {"high": 95.0, "low": 68.0, "high_std": 4.0, "low_std": 3.5},
        6:  {"high": 104.5, "low": 78.0, "high_std": 3.5, "low_std": 3.0},
        7:  {"high": 106.5, "low": 84.5, "high_std": 3.0, "low_std": 2.5},
        8:  {"high": 104.5, "low": 83.0, "high_std": 3.0, "low_std": 2.5},
        9:  {"high": 100.5, "low": 76.5, "high_std": 3.5, "low_std": 3.0},
        10: {"high": 89.5, "low": 64.5, "high_std": 4.5, "low_std": 4.5},
        11: {"high": 76.5, "low": 52.5, "high_std": 5.0, "low_std": 5.0},
        12: {"high": 66.0, "low": 44.5, "high_std": 5.0, "low_std": 5.0},
    },
    "SEA": {
        1:  {"high": 47.5, "low": 36.5, "high_std": 5.5, "low_std": 5.0},
        2:  {"high": 49.5, "low": 37.0, "high_std": 5.5, "low_std": 5.0},
        3:  {"high": 53.5, "low": 39.0, "high_std": 5.0, "low_std": 4.5},
        4:  {"high": 58.5, "low": 42.5, "high_std": 5.0, "low_std": 4.0},
        5:  {"high": 65.0, "low": 48.5, "high_std": 5.0, "low_std": 4.0},
        6:  {"high": 70.5, "low": 53.5, "high_std": 4.5, "low_std": 3.5},
        7:  {"high": 76.5, "low": 57.5, "high_std": 5.0, "low_std": 3.5},
        8:  {"high": 76.5, "low": 57.5, "high_std": 5.0, "low_std": 3.5},
        9:  {"high": 71.0, "low": 53.0, "high_std": 5.5, "low_std": 4.0},
        10: {"high": 60.0, "low": 46.0, "high_std": 5.0, "low_std": 4.5},
        11: {"high": 51.5, "low": 39.5, "high_std": 5.5, "low_std": 5.0},
        12: {"high": 46.0, "low": 35.0, "high_std": 5.5, "low_std": 5.5},
    },
    "ATL": {
        1:  {"high": 52.0, "low": 34.5, "high_std": 7.0, "low_std": 6.5},
        2:  {"high": 55.5, "low": 36.5, "high_std": 7.0, "low_std": 7.0},
        3:  {"high": 63.5, "low": 43.0, "high_std": 7.0, "low_std": 6.5},
        4:  {"high": 72.5, "low": 51.0, "high_std": 6.0, "low_std": 5.5},
        5:  {"high": 80.0, "low": 60.0, "high_std": 5.0, "low_std": 4.5},
        6:  {"high": 87.0, "low": 68.5, "high_std": 4.0, "low_std": 3.5},
        7:  {"high": 89.5, "low": 72.0, "high_std": 3.5, "low_std": 3.0},
        8:  {"high": 89.0, "low": 71.5, "high_std": 3.5, "low_std": 3.0},
        9:  {"high": 83.5, "low": 65.5, "high_std": 5.0, "low_std": 4.5},
        10: {"high": 73.0, "low": 54.0, "high_std": 6.0, "low_std": 5.5},
        11: {"high": 62.5, "low": 43.5, "high_std": 7.0, "low_std": 6.5},
        12: {"high": 53.5, "low": 36.5, "high_std": 7.0, "low_std": 7.0},
    },
    "LAS": {
        1:  {"high": 58.5, "low": 37.5, "high_std": 6.0, "low_std": 5.0},
        2:  {"high": 62.5, "low": 39.5, "high_std": 6.0, "low_std": 5.5},
        3:  {"high": 70.0, "low": 45.5, "high_std": 6.0, "low_std": 5.0},
        4:  {"high": 79.0, "low": 53.0, "high_std": 5.5, "low_std": 4.5},
        5:  {"high": 90.0, "low": 62.5, "high_std": 4.5, "low_std": 4.0},
        6:  {"high": 100.5, "low": 72.5, "high_std": 4.0, "low_std": 3.0},
        7:  {"high": 106.5, "low": 80.5, "high_std": 3.5, "low_std": 2.5},
        8:  {"high": 104.5, "low": 79.0, "high_std": 3.5, "low_std": 2.5},
        9:  {"high": 96.5, "low": 70.0, "high_std": 4.5, "low_std": 3.5},
        10: {"high": 82.0, "low": 57.5, "high_std": 5.5, "low_std": 5.0},
        11: {"high": 68.0, "low": 45.0, "high_std": 6.0, "low_std": 5.5},
        12: {"high": 57.0, "low": 37.0, "high_std": 6.0, "low_std": 5.5},
    },
    "DCA": {
        1:  {"high": 43.5, "low": 29.0, "high_std": 7.0, "low_std": 7.0},
        2:  {"high": 46.5, "low": 30.5, "high_std": 7.5, "low_std": 7.5},
        3:  {"high": 55.5, "low": 37.5, "high_std": 7.5, "low_std": 7.0},
        4:  {"high": 66.5, "low": 46.5, "high_std": 6.5, "low_std": 6.0},
        5:  {"high": 76.0, "low": 56.5, "high_std": 5.5, "low_std": 5.0},
        6:  {"high": 85.0, "low": 66.5, "high_std": 4.5, "low_std": 4.0},
        7:  {"high": 89.5, "low": 71.5, "high_std": 4.0, "low_std": 3.5},
        8:  {"high": 87.5, "low": 70.5, "high_std": 4.0, "low_std": 3.5},
        9:  {"high": 81.0, "low": 63.0, "high_std": 5.0, "low_std": 5.0},
        10: {"high": 69.0, "low": 51.5, "high_std": 6.0, "low_std": 5.5},
        11: {"high": 57.5, "low": 41.0, "high_std": 6.5, "low_std": 6.0},
        12: {"high": 46.0, "low": 32.0, "high_std": 7.0, "low_std": 7.0},
    },
    "BOS": {
        1:  {"high": 36.5, "low": 22.5, "high_std": 7.5, "low_std": 8.0},
        2:  {"high": 39.0, "low": 24.0, "high_std": 7.5, "low_std": 8.0},
        3:  {"high": 46.0, "low": 30.5, "high_std": 7.5, "low_std": 7.0},
        4:  {"high": 57.0, "low": 40.5, "high_std": 7.0, "low_std": 6.0},
        5:  {"high": 67.0, "low": 50.0, "high_std": 6.5, "low_std": 5.0},
        6:  {"high": 76.5, "low": 59.5, "high_std": 5.5, "low_std": 4.5},
        7:  {"high": 82.0, "low": 65.5, "high_std": 5.0, "low_std": 4.0},
        8:  {"high": 80.5, "low": 64.5, "high_std": 5.0, "low_std": 4.0},
        9:  {"high": 73.0, "low": 57.0, "high_std": 6.0, "low_std": 5.0},
        10: {"high": 62.0, "low": 46.5, "high_std": 6.5, "low_std": 5.5},
        11: {"high": 51.5, "low": 37.5, "high_std": 7.0, "low_std": 6.5},
        12: {"high": 40.5, "low": 27.5, "high_std": 7.0, "low_std": 7.5},
    },
    "SFO": {
        1:  {"high": 57.5, "low": 44.5, "high_std": 5.0, "low_std": 4.0},
        2:  {"high": 60.0, "low": 46.0, "high_std": 5.0, "low_std": 4.0},
        3:  {"high": 62.0, "low": 47.5, "high_std": 4.5, "low_std": 3.5},
        4:  {"high": 63.5, "low": 48.5, "high_std": 4.5, "low_std": 3.5},
        5:  {"high": 65.5, "low": 50.5, "high_std": 4.0, "low_std": 3.0},
        6:  {"high": 68.0, "low": 53.0, "high_std": 4.0, "low_std": 3.0},
        7:  {"high": 68.5, "low": 54.0, "high_std": 3.5, "low_std": 2.5},
        8:  {"high": 69.5, "low": 55.0, "high_std": 4.0, "low_std": 3.0},
        9:  {"high": 72.5, "low": 55.5, "high_std": 5.0, "low_std": 3.5},
        10: {"high": 69.5, "low": 53.0, "high_std": 5.0, "low_std": 3.5},
        11: {"high": 62.5, "low": 48.5, "high_std": 5.0, "low_std": 4.0},
        12: {"high": 57.0, "low": 44.0, "high_std": 5.0, "low_std": 4.5},
    },
    "DFW": {
        1:  {"high": 55.0, "low": 34.5, "high_std": 8.0, "low_std": 7.5},
        2:  {"high": 57.5, "low": 36.5, "high_std": 9.0, "low_std": 8.5},
        3:  {"high": 65.5, "low": 44.5, "high_std": 8.0, "low_std": 7.0},
        4:  {"high": 74.5, "low": 53.5, "high_std": 6.5, "low_std": 5.5},
        5:  {"high": 83.0, "low": 63.0, "high_std": 5.5, "low_std": 4.5},
        6:  {"high": 92.0, "low": 72.0, "high_std": 4.5, "low_std": 3.5},
        7:  {"high": 97.0, "low": 76.0, "high_std": 4.0, "low_std": 3.0},
        8:  {"high": 97.0, "low": 76.0, "high_std": 4.0, "low_std": 3.0},
        9:  {"high": 89.5, "low": 67.5, "high_std": 5.5, "low_std": 5.0},
        10: {"high": 77.5, "low": 56.0, "high_std": 7.0, "low_std": 6.5},
        11: {"high": 65.0, "low": 44.0, "high_std": 7.5, "low_std": 7.0},
        12: {"high": 55.5, "low": 35.5, "high_std": 8.0, "low_std": 7.5},
    },
    "MSP": {
        1:  {"high": 24.0, "low": 7.0, "high_std": 11.0, "low_std": 12.0},
        2:  {"high": 28.5, "low": 11.5, "high_std": 11.0, "low_std": 11.5},
        3:  {"high": 41.5, "low": 24.0, "high_std": 10.0, "low_std": 9.5},
        4:  {"high": 57.5, "low": 36.5, "high_std": 8.5, "low_std": 7.5},
        5:  {"high": 69.5, "low": 48.0, "high_std": 7.5, "low_std": 6.5},
        6:  {"high": 79.5, "low": 58.5, "high_std": 6.0, "low_std": 5.0},
        7:  {"high": 84.0, "low": 64.0, "high_std": 5.0, "low_std": 4.5},
        8:  {"high": 82.0, "low": 61.5, "high_std": 5.5, "low_std": 5.0},
        9:  {"high": 73.5, "low": 52.5, "high_std": 7.0, "low_std": 6.5},
        10: {"high": 58.0, "low": 40.0, "high_std": 8.5, "low_std": 7.5},
        11: {"high": 42.0, "low": 26.5, "high_std": 9.5, "low_std": 9.0},
        12: {"high": 27.5, "low": 12.0, "high_std": 10.0, "low_std": 11.0},
    },
    "IAH": {
        1:  {"high": 62.5, "low": 42.5, "high_std": 7.0, "low_std": 7.0},
        2:  {"high": 65.0, "low": 44.5, "high_std": 7.5, "low_std": 7.5},
        3:  {"high": 71.5, "low": 51.5, "high_std": 7.0, "low_std": 6.5},
        4:  {"high": 78.5, "low": 58.5, "high_std": 5.5, "low_std": 5.0},
        5:  {"high": 86.0, "low": 66.5, "high_std": 4.5, "low_std": 4.0},
        6:  {"high": 92.0, "low": 73.0, "high_std": 3.5, "low_std": 3.0},
        7:  {"high": 95.0, "low": 75.0, "high_std": 3.0, "low_std": 2.5},
        8:  {"high": 95.5, "low": 75.0, "high_std": 3.0, "low_std": 2.5},
        9:  {"high": 90.5, "low": 69.5, "high_std": 4.5, "low_std": 4.0},
        10: {"high": 81.5, "low": 59.0, "high_std": 6.0, "low_std": 6.0},
        11: {"high": 71.0, "low": 49.0, "high_std": 7.0, "low_std": 7.0},
        12: {"high": 63.0, "low": 43.5, "high_std": 7.0, "low_std": 7.0},
    },
    "SAT": {
        1:  {"high": 61.5, "low": 39.0, "high_std": 7.5, "low_std": 7.0},
        2:  {"high": 63.5, "low": 41.5, "high_std": 8.5, "low_std": 8.0},
        3:  {"high": 72.0, "low": 49.5, "high_std": 7.5, "low_std": 6.5},
        4:  {"high": 80.0, "low": 57.5, "high_std": 6.0, "low_std": 5.5},
        5:  {"high": 87.5, "low": 66.0, "high_std": 5.0, "low_std": 4.5},
        6:  {"high": 94.0, "low": 72.5, "high_std": 4.0, "low_std": 3.0},
        7:  {"high": 97.0, "low": 74.5, "high_std": 3.5, "low_std": 2.5},
        8:  {"high": 97.5, "low": 74.5, "high_std": 3.5, "low_std": 2.5},
        9:  {"high": 91.5, "low": 69.0, "high_std": 5.0, "low_std": 4.5},
        10: {"high": 81.0, "low": 58.5, "high_std": 6.5, "low_std": 6.0},
        11: {"high": 70.5, "low": 47.0, "high_std": 7.5, "low_std": 7.0},
        12: {"high": 62.0, "low": 39.5, "high_std": 7.5, "low_std": 7.5},
    },
    "OKC": {
        1:  {"high": 48.5, "low": 27.5, "high_std": 9.0, "low_std": 8.5},
        2:  {"high": 52.0, "low": 30.5, "high_std": 9.5, "low_std": 9.0},
        3:  {"high": 61.5, "low": 39.0, "high_std": 9.0, "low_std": 7.5},
        4:  {"high": 71.0, "low": 48.5, "high_std": 7.5, "low_std": 6.5},
        5:  {"high": 79.5, "low": 58.5, "high_std": 6.0, "low_std": 5.0},
        6:  {"high": 89.0, "low": 68.0, "high_std": 4.5, "low_std": 4.0},
        7:  {"high": 94.0, "low": 72.5, "high_std": 4.0, "low_std": 3.5},
        8:  {"high": 93.5, "low": 72.0, "high_std": 4.0, "low_std": 3.5},
        9:  {"high": 85.0, "low": 62.0, "high_std": 6.0, "low_std": 5.5},
        10: {"high": 72.5, "low": 50.0, "high_std": 7.5, "low_std": 7.0},
        11: {"high": 59.5, "low": 38.0, "high_std": 8.0, "low_std": 7.5},
        12: {"high": 49.5, "low": 28.5, "high_std": 8.5, "low_std": 8.5},
    },
    "MSY": {
        1:  {"high": 62.5, "low": 44.0, "high_std": 6.5, "low_std": 6.5},
        2:  {"high": 65.0, "low": 46.0, "high_std": 7.0, "low_std": 7.0},
        3:  {"high": 72.0, "low": 53.0, "high_std": 6.5, "low_std": 5.5},
        4:  {"high": 78.5, "low": 59.5, "high_std": 5.5, "low_std": 5.0},
        5:  {"high": 85.5, "low": 67.5, "high_std": 4.5, "low_std": 4.0},
        6:  {"high": 90.5, "low": 74.0, "high_std": 3.5, "low_std": 2.5},
        7:  {"high": 91.5, "low": 75.5, "high_std": 3.0, "low_std": 2.5},
        8:  {"high": 91.5, "low": 75.5, "high_std": 3.0, "low_std": 2.5},
        9:  {"high": 88.5, "low": 72.0, "high_std": 4.0, "low_std": 3.5},
        10: {"high": 80.5, "low": 62.0, "high_std": 5.5, "low_std": 5.5},
        11: {"high": 71.0, "low": 52.0, "high_std": 6.5, "low_std": 6.5},
        12: {"high": 63.5, "low": 45.0, "high_std": 6.5, "low_std": 6.5},
    },
}


# ── API ───────────────────────────────────────────────────────────────────────

def get_climate_normal(
    city_key: str, month: int, field: str
) -> tuple[float, float] | None:
    """
    Return (climo_mean, climo_std) for a city/month/field.

    field: "high" or "low"
    Returns None if no data available for this city.
    """
    normals = CLIMATE_NORMALS.get(city_key)
    if normals is None:
        return None
    month_data = normals.get(month)
    if month_data is None:
        return None

    mean = month_data.get(field)
    std = month_data.get(f"{field}_std")
    if mean is None or std is None:
        return None
    return (mean, std)


def bayesian_climo_blend(
    forecast_mean: float,
    forecast_std: float,
    climo_mean: float,
    climo_std: float,
    climo_weight: float,
) -> tuple[float, float]:
    """
    Bayesian blend of the forecast with the climatological prior.

    Uses precision-weighted averaging (optimal under Gaussian assumption):
      posterior_mean = (forecast_precision * forecast_mean + prior_precision * climo_mean)
                     / (forecast_precision + prior_precision)

    The climo_weight parameter scales the prior precision to control
    how aggressively we pull toward climatology.  Higher = more pull.

    Returns (blended_mean, blended_std).
    """
    # Precision = 1/variance
    forecast_prec = 1.0 / (forecast_std ** 2)
    climo_prec = climo_weight / (climo_std ** 2)

    total_prec = forecast_prec + climo_prec
    blended_mean = (forecast_prec * forecast_mean + climo_prec * climo_mean) / total_prec
    blended_std = math.sqrt(1.0 / total_prec)

    return blended_mean, blended_std


def compute_climo_anomaly(
    forecast_mean: float,
    city_key: str,
    month: int,
    field: str,
) -> float | None:
    """
    Return the number of climatological standard deviations the forecast
    departs from normal.  Positive = warmer than normal.

    Returns None if no climo data available.
    """
    normal = get_climate_normal(city_key, month, field)
    if normal is None:
        return None
    climo_mean, climo_std = normal
    if climo_std < 0.1:
        return None
    return (forecast_mean - climo_mean) / climo_std


def emos_spread_calibration(
    raw_std: float,
    inflation_factor: float = 1.0,
    min_spread: float = 1.0,
) -> float:
    """
    Ensemble Model Output Statistics (EMOS) variance calibration.

    Raw ensemble spread is typically too narrow (under-dispersive).
    This applies a multiplicative inflation factor and enforces a minimum.

    The inflation_factor should be tuned from historical verification
    (typical values: 1.1 - 1.5 for temperature).
    """
    calibrated = raw_std * inflation_factor
    return max(calibrated, min_spread)
