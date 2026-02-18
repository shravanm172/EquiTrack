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

The system is intentionaly layered to separate pure analytical logic from orchestration and API exposure.
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
  - Validates ticker/date and resolves next valid trading day/price when needded

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

Set environment variables in your deployment platforms (recommended) instead of committing `.env.production`.

- Frontend (`VITE_API_URL`):
  - `VITE_API_URL=https://equitrack-p4yp.onrender.com`
- Backend (`CORS_ORIGINS`):
  - `CORS_ORIGINS=https://equi-track.vercel.app`
  - For multiple origins, use comma-separated values.

After updating frontend env vars, trigger a redeploy so Vite rebuilds with the new values.


## Live Deployment

Frontend (Vercel):
https://equi-track.vercel.app

Backend (Render):
https://equitrack-p4yp.onrender.com

Backend Health Check:
https://equitrack-p4yp.onrender.com/api/health
---

## Test Suite

Run backend tests:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
pytest
```

Current tests cover baseline analytics pipeline, stress scenarios, and forecast integration.

----

## CI / CD Pipeline

EquiTrack uses GitHub Actions for continuous integration.

- Automated backend test suite runs on every pull request.
- Merges to `main` require passing status checks.
- Backend auto-deploys to Render on successful merge.
- Frontend auto-deploys to Vercel on push to `main`.

This ensures code correctness before deployment and enforces disciplined development workflow.

----

**AI usage disclosure:** GPT-5.2 was used to assist in generating parts of the pytest test suite.

---

## Notes / Disclaimers

### Responsiveness

EquiTrack interface is currently optimized for desktop usage. Mobile responsiveness is planned for a future iteration.