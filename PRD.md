# MARK JOHNSON

**Autonomous Temperature Market Scanner & Alert Engine**

Product Requirements Document v1.0 | February 2026

---

| Field | Value |
|-------|-------|
| Product Name | MARK JOHNSON |
| Subtitle | Autonomous Temperature Market Scanner & Alert Engine |
| Version | 1.0 |
| Author | Sam |
| Status | Draft |

---

## 1. Executive Summary

MARK JOHNSON is an autonomous temperature market intelligence system that continuously monitors prediction markets on Kalshi, generates independent probabilistic temperature forecasts using a multi-source ensemble weather engine, identifies statistically favorable positions where model-derived probabilities meaningfully diverge from market-implied odds, and delivers structured trade alerts to Discord for human review and execution.

The system operates in a read-only capacity with respect to trading: it scans, models, compares, and alerts. It does not execute trades autonomously. The human operator retains full decision-making authority over capital deployment.

MARK JOHNSON is purpose-built for daily high temperature markets across all US cities listed on Kalshi. Temperature is the most predictable atmospheric variable, making it the optimal starting point for building systematic edge in weather prediction markets.

---

## 2. Problem Statement

Kalshi lists daily temperature prediction markets for 20+ US cities. These markets resolve based on official weather station readings, creating a structured, repeatable opportunity set. However, exploiting these markets requires solving three problems simultaneously:

1. **Weather Accuracy:** Consumer weather apps provide point forecasts without calibrated uncertainty. To price market bands correctly, you need a full probability distribution of possible temperatures, not a single number.
2. **Speed:** Markets move as new weather data arrives. The trader who assimilates new model runs, observations, and radar data fastest gains a structural advantage. Manual monitoring is too slow.
3. **Signal Identification:** Even with accurate forecasts, the trader must compare model probabilities to market-implied probabilities across dozens of simultaneous markets, identify meaningful deviations, and act before the edge disappears.

MARK JOHNSON automates all three.

---

## 3. Product Vision & Goals

### 3.1 Vision

Build the fastest, most accurate temperature forecasting and market scanning system available to an individual trader, producing calibrated probabilistic forecasts that systematically outperform Kalshi market consensus.

### 3.2 Primary Goals

- Generate calibrated probability distributions for daily max temperature in all Kalshi-listed cities
- Continuously scan open temperature markets and compute edge (model probability vs. market-implied probability)
- Deliver actionable alerts to Discord when edge exceeds configurable thresholds
- Log all forecasts and outcomes for systematic performance evaluation

### 3.3 Non-Goals (v1)

- No automated trade execution (alerts only)
- No capital or portfolio management
- No precipitation or snow markets (future expansion)
- No high-frequency arbitrage or market-making

---

## 4. System Architecture

MARK JOHNSON is composed of five core services that run autonomously and communicate through a shared data layer.

### 4.1 Service Overview

| Service | Function | Update Cadence |
|---------|----------|----------------|
| **Weather Engine** | Ingests model data, observations, and ensemble forecasts. Produces bias-corrected probability distributions per city. | 1–60 min (varies by source) |
| **Market Scanner** | Polls Kalshi public API. Parses open temperature markets, extracts bands, and computes implied probabilities. | Every 2–5 minutes |
| **Signal Engine** | Compares model distributions to market odds. Flags statistically favorable positions. | On each market or forecast update |
| **Alert Dispatcher** | Formats and sends structured Discord messages when signal thresholds are met. | Event-driven |
| **Logger / Evaluator** | Stores all forecasts, market snapshots, and settlement outcomes. Computes performance metrics. | Continuous + daily batch |

### 4.2 Data Flow

```
Weather Sources ──┐
                   ├──▶ Signal Engine ──▶ Alert Dispatcher ──▶ Discord
Kalshi Markets ───┘          │
                             ▼
                     Logger / Evaluator ──▶ Postgres
```

The system follows a linear pipeline: Ingest → Process → Compare → Alert → Log. Weather data and market data are ingested independently on their own schedules. The Signal Engine triggers whenever either input updates. Alerts are dispatched only when edge thresholds are met. Every prediction and market snapshot is logged for post-hoc evaluation.

---

## 5. Data Sources

### 5.1 Weather Data

| Source | Type | Resolution | Update Freq | Purpose |
|--------|------|------------|-------------|---------|
| HRRR (NOAA) | Mesoscale NWP model | 3 km, hourly | Every hour | Primary short-range forecast (0–18h) |
| GFS (NOAA) | Global NWP model | 0.25°, 6-hourly | 4x daily | Medium-range backbone (1–10 days) |
| ECMWF (if accessible) | Global NWP model | 0.1°, 6-hourly | 2x daily | Best global skill, ensemble spread |
| Open-Meteo Ensemble API | Multi-model ensemble | Varies | 2–4x daily | Fast multi-model access (MVP) |
| NWS Forecast Grid API | Official forecast | 2.5 km grid | Every 10–30 min | Baseline + local office context |
| METAR / ASOS Stations | Surface observations | Station-level | Every 1–5 min | Real-time bias correction & nowcasting |
| GOES-16/18 Satellite | Cloud cover imagery | Regional | Every 5–15 min | Cloud onset detection for max temp adjustment |
| MRMS Radar | Precipitation detection | 1 km | Every 2 min | Precip timing (future use) |

### 5.2 Market Data (Kalshi Public API)

**Base URL:** `https://api.elections.kalshi.com/trade-api/v2`

| Endpoint | Purpose | Auth Required |
|----------|---------|---------------|
| `/markets?status=open` | List all open markets, filter by series_ticker | No |
| `/markets/{ticker}` | Market details: prices, bands, settlement rules | No |
| `/markets/{ticker}/orderbook` | Current bid/ask, depth, spread | No |
| `/series/{ticker}` | Series metadata for recurring temperature markets | No |
| `/events` | Grouped events including weather forecasts | No |

All market data endpoints are unauthenticated. No API key is required for read-only access. Polling interval: every 2–5 minutes. Rate limit compliance is mandatory.

---

## 6. Core Engine: Forecast Model

The forecast engine transforms raw weather data into calibrated probability distributions for daily maximum temperature at each target city. It operates in four layers.

### 6.1 Layer 1: Bias Correction

Every model has systematic errors that vary by location, time of day, season, and weather regime. MARK JOHNSON learns and corrects these biases.

- For each (model, city, lead_time) tuple, compute rolling bias: `corrected_temp = raw_temp − historical_bias`
- Start with a 30-day rolling mean bias
- Upgrade to gradient-boosted bias model using features: station elevation, urban heat island index, distance to water, wind direction, cloud cover regime, season

### 6.2 Layer 2: Ensemble Blending (Stacking)

Combine bias-corrected outputs from all models into a single optimal prediction using a stacking ensemble.

- **Base learners:** each model's corrected forecast
- **Meta-model** (LightGBM or XGBoost) learns optimal weights conditioned on: lead time bucket (0–1h, 1–3h, 3–6h, 6–12h, 12–24h), time of day, season, weather regime (clear/cloudy/windy/frontal), city-specific factors
- Retrain weights weekly using most recent 90 days of verification data

### 6.3 Layer 3: Probability Distribution

Point forecasts are insufficient. MARK JOHNSON produces a full probability distribution for max temperature.

- **Method:** quantile regression (p05, p10, p25, p50, p75, p90, p95)
- **Alternative:** fit a normal or skew-normal distribution using ensemble spread + historical model error variance
- **Output:** for any temperature band [X, Y], compute P(X ≤ Tmax < Y) via CDF integration

### 6.4 Layer 4: Nowcasting Corrections

This is the competitive edge. Real-time observations adjust the forecast curve as the day progresses.

- If actual temperature is tracking +2°F above forecast curve at 10am, shift the entire distribution upward
- If satellite shows cloud deck arriving 2 hours early, reduce max temp distribution mean and increase variance
- If wind shifts indicate frontal passage timing change, adjust accordingly
- Correction magnitude scales inversely with time remaining to market close

---

## 7. Core Engine: Market Scanner

### 7.1 Market Parsing

For each open temperature market on Kalshi, extract and normalize:

| Field | Source | Example |
|-------|--------|---------|
| City | Event title parsing | NYC |
| Market type | Series ticker | Highest temperature |
| Temperature band | Market contracts | 46° to 47° |
| Implied probability | Best bid/ask midpoint | 0.46 |
| Best bid / Best ask | Orderbook endpoint | $0.45 / $0.47 |
| Volume | Market endpoint | $148,076 |
| Settlement station | Settlement rules | Central Park (KNYC) |
| Market close time | Market endpoint | 11:59 PM ET |

### 7.2 Normalized Data Structure

```json
{
  "city": "NYC",
  "type": "high_temp",
  "band_min": 46,
  "band_max": 47,
  "implied_prob": 0.46,
  "best_bid": 0.45,
  "best_ask": 0.47,
  "volume": 148076,
  "settlement_station": "KNYC",
  "close_time": "2026-02-17T23:59:00-05:00"
}
```

---

## 8. Signal Detection & Edge Calculation

### 8.1 Edge Formula

```
edge = model_probability − market_implied_probability
EV (YES position) = (model_probability × payout) − cost_to_enter
```

### 8.2 Alert Trigger Conditions

A signal is generated when **ALL** of the following conditions are met:

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| Absolute edge | ≥ 8% | Minimum meaningful deviation after accounting for model uncertainty |
| Market volume | ≥ $5,000 | Ensures sufficient liquidity for entry/exit |
| Time to close | ≥ 60 minutes | Avoids last-minute volatility and thin books |
| Forecast stability | Edge persists across 2 consecutive polling intervals | Filters transient noise from single data point updates |
| Model confidence | Ensemble spread < 4°F for the band in question | Ensures model is not wildly uncertain |

### 8.3 Edge Classification

| Edge Range | Classification | Recommended Action |
|------------|---------------|-------------------|
| 8–12% | **Moderate** | Alert sent. Consider if liquidity and confidence support entry. |
| 12–20% | **Strong** | Alert sent with emphasis. High-priority review. |
| >20% | **Extreme** | Alert sent with caution flag. Verify model inputs — may indicate data error. |

---

## 9. Alert System (Discord)

### 9.1 Delivery Method

Alerts are delivered via Discord webhook POST to a dedicated channel. No bot framework required for v1.

### 9.2 Alert Message Format

```
🌡️  MARK JOHNSON ALERT — NYC

Band: 46°–47°F
Market Implied: 46%
Model Probability: 58%
Edge: +12% (STRONG)

Model Inputs:
  HRRR:  47.8°F
  NWS:   46.0°F
  Ensemble Mean: 47.2°F
  Current Obs: 44°F (+0.9°F/hr)

Confidence: HIGH
Time to Close: 4h 12m
Volume: $148,076
```

### 9.3 Alert Types

| Type | Trigger | Emoji |
|------|---------|-------|
| New Signal | Edge threshold first exceeded | 🟢 |
| Signal Update | Edge changed significantly since last alert | 🟡 |
| Signal Expired | Market closed or edge dropped below threshold | 🔴 |
| Daily Summary | End-of-day recap of all signals and outcomes | 📊 |
| System Health | Hourly heartbeat confirming all services running | ❤️ |

---

## 10. Target Markets & Cities

### 10.1 Supported Cities (All Kalshi Temperature Markets)

| City | Station | Notes |
|------|---------|-------|
| New York City | Central Park (KNYC) | Highest volume market |
| Miami | Miami Intl (KMIA) | Tropical regime, low variance |
| Chicago | O'Hare (KORD) | Lake effect considerations |
| Denver | Denver Intl (KDEN) | Altitude + Chinook wind variability |
| Austin | Austin-Bergstrom (KAUS) | |
| Los Angeles | Downtown (KCQT) | Marine layer critical |
| Philadelphia | Phila Intl (KPHL) | |
| Phoenix | Sky Harbor (KPHX) | Urban heat island extreme |
| Seattle | Sea-Tac (KSEA) | Marine influence |
| Atlanta | Hartsfield (KATL) | |
| Las Vegas | McCarran (KLAS) | Arid, high predictability |
| Washington DC | Reagan (KDCA) | |
| Boston | Logan (KBOS) | Coastal influence |
| San Francisco | Downtown (KSFO) | Microclimates, fog |
| Dallas | DFW (KDFW) | |
| Minneapolis | MSP (KMSP) | Arctic air intrusion risk |
| Houston | Hobby/Bush (KIAH) | Gulf moisture |
| San Antonio | SAT (KSAT) | |
| Oklahoma City | Will Rogers (KOKC) | Frontal boundary zone |
| New Orleans | MSY (KMSY) | Gulf regime |

### 10.2 Market Types

- **Daily highest temperature** (primary focus, v1)
- **Daily lowest temperature** (secondary, monitor only in v1)

---

## 11. Evaluation & Performance Framework

### 11.1 Forecast Accuracy Metrics

| Metric | Target | Measured Over |
|--------|--------|---------------|
| Mean Absolute Error (MAE) | < 1.5°F for 0–12h lead | Rolling 30 days per city |
| Brier Score (band probs) | < 0.20 | All bands, all cities |
| Calibration (reliability) | 70% events should verify 65–75% of time | Weekly calibration curves |
| Bias | < 0.3°F systematic | Per model, per city, per season |

### 11.2 Signal Quality Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Signal hit rate | > 55% | % of signaled bands that resolve YES |
| Expected value realization | Positive over 30-day rolling window | Paper P&L tracking |
| False signal rate | < 30% | % of alerts where model was wrong by > 5% |
| Alert-to-close time | Report average lead time | How early signals fire before market close |

### 11.3 Evaluation Cadence

- **Daily:** log all forecasts, market snapshots, and settlement outcomes
- **Weekly:** update bias correction tables, re-weight ensemble models, publish calibration curves
- **Monthly:** full performance review, model retraining, threshold tuning

---

## 12. Risk Controls & Guardrails

Even in alert-only mode, MARK JOHNSON implements safeguards to prevent bad signals.

| Control | Rule | Rationale |
|---------|------|-----------|
| Minimum time to close | Ignore markets closing within 60 minutes | Thin order books, high slippage risk |
| Minimum volume | Ignore markets with < $5,000 volume | Insufficient liquidity |
| Data freshness | Suppress alerts if weather data is > 90 minutes stale | Stale data produces unreliable distributions |
| Extreme edge cap | Flag edges > 25% as potential data errors | Model bugs or API issues can produce false extremes |
| City exclusion list | Configurable list to exclude high-noise cities | SF fog, Denver Chinooks may produce unreliable signals initially |
| Rate limiting | Max 1 alert per city per 15 minutes | Prevent alert fatigue from oscillating signals |
| Heartbeat monitoring | Alert if any service is down > 10 minutes | Ensures system is running during market hours |

---

## 13. Technology Stack

### 13.1 Core Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.11+ | Best ecosystem for weather data (xarray, cfgrib, scipy) |
| API Framework | FastAPI | Async, fast, type-safe |
| Weather Data | xarray, cfgrib, zarr, Herbie | Standard NWP data tooling |
| ML / Stats | LightGBM, scikit-learn, scipy.stats | Ensemble blending + distribution fitting |
| Database | PostgreSQL (Supabase) | Structured storage for features, predictions, evaluations |
| Cache | Redis | Hot forecast cache, rate limiting |
| Scheduler | APScheduler or cron | Orchestrate polling and model runs |
| Discord | Webhook POST (aiohttp) | Zero-dependency alert delivery |
| Deployment | Railway / Fly.io / VPS | Persistent compute for continuous polling |

### 13.2 Optional Enhancements

- **Dashboard:** Next.js + Vercel for real-time monitoring UI
- **WebSocket:** Kalshi WS feed for real-time market updates (upgrade from polling)
- **Grafana:** Forecast error monitoring dashboards

---

## 14. Development Roadmap

### Phase 1: MVP (Weeks 1–2)

> **Goal:** End-to-end pipeline running. Alerts flowing to Discord.

1. Kalshi market scanner: poll open temperature markets, parse bands, compute implied probabilities
2. Open-Meteo ensemble integration: pull multi-model forecasts for all target cities
3. Simple probability model: fit normal distribution from ensemble mean + spread
4. Edge calculator: compare model probabilities to market-implied probabilities
5. Discord webhook: send structured alerts when edge > threshold
6. Logger: store all forecasts and market snapshots in Postgres

### Phase 2: Accuracy Engine (Weeks 3–4)

> **Goal:** Meaningfully better forecasts than Phase 1.

1. HRRR ingest via Herbie: pull hourly high-resolution model data
2. Station-level bias correction: learn systematic errors per model per city
3. Ensemble stacking: LightGBM meta-model with lead time and regime features
4. Real-time observation feed: METAR/ASOS for nowcasting corrections
5. Observation-based drift correction: if current temp is tracking hot/cold, adjust distribution

### Phase 3: Edge Optimization (Weeks 5–6)

> **Goal:** Sharpen signals and reduce false alerts.

1. Quantile regression for calibrated uncertainty bands
2. Weather regime classification (clear, cloudy, frontal, lake-effect)
3. Dynamic model weighting conditioned on regime
4. Reliability calibration curves and Brier score tracking
5. Performance dashboard with per-city, per-lead-time error breakdowns

### Phase 4: Scale & Harden (Weeks 7–8)

> **Goal:** Production-grade reliability.

1. Kalshi WebSocket integration for real-time market updates
2. Satellite cloud cover integration for max temp nowcasting
3. Automated model retraining pipeline (weekly)
4. System health monitoring and alerting
5. Paper trading P&L tracker with daily reports

---

## 15. Success Metrics

### 15.1 Launch Criteria (Phase 1 Complete)

- System runs autonomously for 48+ hours without manual intervention
- Alerts are delivered to Discord for all cities with open temperature markets
- Forecasts are logged and can be compared to settlement outcomes

### 15.2 Performance Criteria (Phase 2+ Complete)

- Temperature MAE < 1.5°F for same-day forecasts across all cities
- Signal hit rate > 55% over a 30-day rolling window
- Positive expected value on paper trades over 30-day rolling window
- Zero missed markets (all open temperature markets scanned every polling interval)

---

## 16. Future Expansion

| Capability | Description | Prerequisite |
|-----------|-------------|--------------|
| Automated trading | Direct order placement via Kalshi authenticated API | Proven positive EV over 60+ days of paper trading |
| Snow / precipitation markets | Extend forecast engine to precip variables | Radar ingest + precip-specific ensemble calibration |
| Low temperature markets | Full support for overnight low predictions | Nighttime bias correction + radiative cooling model |
| Teleconnection features | ENSO, NAO, MJO, PNA as long-range regime signals | Historical verification of teleconnection skill |
| Multi-platform alerts | Slack, SMS, email in addition to Discord | Demand from user workflow |
| Public-facing dashboard | Web UI showing model forecasts vs. market odds | Frontend engineering effort |
| Global markets | Extend to non-US cities if Kalshi adds them | International weather data pipeline |

---

## 17. Appendix: API Reference

### 17.1 Kalshi Public API

- **Base URL:** `https://api.elections.kalshi.com/trade-api/v2`
- **Authentication:** None required for read-only market data
- **WebSocket:** `wss://api.elections.kalshi.com/trade-api/ws/v2`
- **Docs:** https://docs.kalshi.com/
- Rate limits apply. Respect HTTP headers. Use exponential backoff on 429 responses.

### 17.2 Weather APIs

- **NWS API:** https://api.weather.gov — No auth required
- **Open-Meteo:** https://open-meteo.com/en/docs — No auth required for free tier
- **HRRR Data:** AWS Open Data Registry (`s3://noaa-hrrr-bdp-pds`) — Public access
- **Herbie (Python):** https://herbie.readthedocs.io — Simplifies NWP data access

### 17.3 Discord Webhook

Create a webhook in Discord channel settings. POST JSON to the webhook URL. No bot token required. Rate limit: 30 messages per 60 seconds per webhook.

---

*End of Document*
