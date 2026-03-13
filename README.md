# EquiTrack

**EquiTrack** is a full-stack portfolio analytics and market simulation platform designed to analyze, stress test, and forecast equity portfolio performance across varying market regimes.

It combines historical market data, deterministic  portfolio analytics, stress-scenario transforms, and stochastic Monte Carlo forecasting in a modular Flask backend + React frontend architecture.
---

## Core Features

### Portfolio Analytics

- Portfolio defined by holdings (tickers + shares)
- Historical price data over a user-defined analysis window
- Analysis pipeline:
  1. Prices → daily returns
  2. Asset returns → weighted portfolio returns
  3. Portfolio returns → equity curve
- Risk metrics:
  - Annualized return
  - Annualized volatility
  - Maximum drawdown
  - Sharpe ratio
These basline metrics serve as the foundation for stress testing and forward simulations

### Stress Testing Scenarios
Equitrack supports deterministic stress transforms applied directly to historical price series
Supported scenario transforms:

1. **Permanent Shock**
   - Instant price shock on a chosen date
   - Price level remains permanently shifted after shock date

2. **Linear Rebound Shock**
   - Initial shock
   - Linear recovery back to baseline over configurable days

3. **Regime Shift (Volatility + Drift)**
   - Volatility multiplier applied to returns
   - Drift shift applied to returns
   - Useful for prolonged bullish/bearish regime simulation

### Stochastic Forecasting

EquiTrack includes a Monte Carlo simulation engine for forecasting potential future portfolio paths based on historical return characteristics.

Rather than projecting a single deterministic path, the system generates many possible future trajectories, allowing the distribution of potential outcomes to be analyzed.

## Simulation Model
The simulation engine models future daily returns using parameters derived from historical data:
  - Expected return (mean)
  - Volatility estimate (currently EWMA based)

Each simulation produces a potential future equity curve.

Running N simulations produces a distribution of portfolio outcomes.

## Forecasting Outputs
From the simulated distribution of equity curves, EquiTrack computes forecast statistics including:
  - Median projected equity curve
  - Percentile bands (e.g., 5th / 95th percentiles)
  - Forecasted return metrics
  - Forecasted volatility estimates
  - Projected drawdown behavior

These outputs allow users to visualize expected portfolio growth as well as downside risk under stochastic market dynamics.

# Visualization
Forecast results are visualized in the frontend using interactive equity curves, including:
  - Historical baseline equity curve
  - Stress scenario equity curve (if applied)
  - Forecasted equity trajectory
  - Optional percentile confidence bands

Charts are rendered using Recharts and automatically update when forecasts are generated.


## Architecture Overview

The system is intentionaly layered to separate pure analytical computation from API orchestration and exposure.
Backend modules align with current code layout:

- `services/analysis_service.py`
  - Baseline portfolio analysis orchestration
- `services/stress_service.py`
  - Baseline/scenario stress analysis and metric deltas
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

The EquiTrack interface is currently optimized for desktop usage. Mobile responsiveness/compatibility is planned for a future iteration.