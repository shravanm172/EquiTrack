# EquiTrack

**EquiTrack** is a full-stack portfolio analytics and market simulation platform designed to analyze, stress test, and forecast equity portfolio performance across varying market regimes.

It combines historical market data, deterministic portfolio analytics, stress-scenario transforms, and forward projections in a modular Flask backend + React frontend architecture.
---

## Core Features

### Portfolio Analytics

- Portfolio defined by holdings (tickers + shares)
- Historical price data over a user-selected analysis window
- Analysis pipeline:
  1. Prices → daily returns
  2. Asset returns → weighted portfolio returns
  3. Portfolio returns → equity curve
- Risk metrics:
  - Annualized return
  - Annualized volatility
  - Maximum drawdown
  - Sharpe ratio

### Stress Testing Scenarios

Supported scenario transforms:

1. **Permanent Shock**
   - Instant price shock on a chosen date
   - Price level remains shifted after shock date

2. **Linear Rebound Shock**
   - Initial shock
   - Linear recovery back toward baseline over configurable days

3. **Regime Shift (Volatility + Drift)**
   - Volatility multiplier on returns
   - Drift shift on returns
   - Useful for prolonged bullish/bearish regime simulation

### Deterministic Forecasting

Forecasting extends historical equity behavior using model-driven deterministic returns.
Forecasting operates deterministically (non-stochastic) and is designed as a foundation for future Monte Carlo simulation extensions.

Supported forecast modes:

1. **Mean** (full-sample average return)
2. **Rolling mean** (windowed average)
3. **EWMA** (lambda/alpha-weighted estimate)

Forecasts can run on baseline and stress outputs, and are visualized alongside historical equity curves.

---

## Architecture Overview

The system is intentionally layered to separate pure analytical logic from orchestration and API exposure.
Backend modules align with current code layout:

- `services/analysis_service.py`
  - Baseline portfolio analysis orchestration
- `services/stress_service.py`
  - Baseline/scenario stress analysis and metric deltas
- `engines/scenario_engine.py`
  - Scenario transformation logic
- `engines/forecast_engine.py`
  - Deterministic forecasting logic
- `services/analysis_store.py`
  - Stores completed analysis artifacts for downstream forecast calls

---

## API Endpoints

- `GET /api/health`
  - Health check

- `POST /api/holdings/validate`
  - Validates ticker/date and resolves next valid trading day/price when needed

- `POST /api/analyze`
  - Baseline portfolio analytics

- `POST /api/analyze_shock`
  - Stress scenario analytics (baseline + scenario + deltas)

- `POST /api/forecast`
  - Forecast projection for baseline/scenario analysis outputs

---

## Technology Stack

### Backend
- Python
- Flask + Flask-CORS
- pandas / NumPy
- pytest

### Frontend
- React + Vite
- Recharts

---

## Development Setup (Local)

### 1) Backend (Flask API)

From project root:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Backend runs on `http://127.0.0.1:5000` by default.

### 2) Frontend (Vite)

In a new terminal from project root:

```powershell
cd frontend
npm install
npm run dev
```

### 3) Frontend API config (Dev mode)

The frontend reads API base URL from `VITE_API_URL` (see `frontend/src/config/api.js`).

Create `frontend/.env.development` with:

```dotenv
VITE_API_URL=http://localhost:5000
```

Important: restart `npm run dev` after changing `.env.*` files.

### 4) Production API config

Set `VITE_API_URL` in your deployment environment (or in `frontend/.env.production`) to your deployed backend URL.

---

## Test Suite

Run backend tests:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
pytest
```

Current tests cover baseline analytics pipeline, stress scenarios, and forecast integration.

**AI usage disclosure:** GPT-5.2 was used to assist in generating parts of the pytest test suite.

---

## Notes / Disclaimers

### Responsiveness

EquiTrack interface is currently optimized for desktop usage. Mobile responsiveness is planned for a future iteration.