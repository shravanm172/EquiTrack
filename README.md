# EquiTrack

**EquiTrack** is a full-stack portfolio analytics and market simulation platform designed to analyze, stress test, and forecast equity portfolio performance across varying market regimes.

It combines historical market data, deterministic portfolio analytics, stress-scenario transforms, and stochastic Monte Carlo forecasting in a modular Flask backend + React frontend architecture.
---

## Core Features

### Portfolio Analytics

- Portfolio defined by holdings (tickers + shares)
- Historical price data over a user-defined analysis window
- Core Analysis pipeline:
  1. Prices → daily returns
  2. Asset returns → weighted portfolio returns
  3. Portfolio returns → equity curve
- Standard risk metrics used:
  - Annualized return
  - Annualized volatility
  - Maximum drawdown
  - Sharpe ratio
These basline metrics serve as the foundation for stress testing and forward simulations

### Stress Testing Scenarios
EquiTrack's stress tests are forward Monte Carlo simulations from a chosen as-of date (any trading day within the analysis window, up to and including the most recent one) — not a reshaping of already-realized returns. When the as-of date is in the past, the app also shows what actually happened afterward as a real, non-simulated baseline for comparison.

Two shock mechanisms, both producing the same response shape (historical curve, actual-realized curve, simulated percentile bands):

1. **Deterministic Regime Shift**
   - Estimates mu and sigma from historical returns up to the as-of date
   - Applies a user-configurable drift shift and volatility multiplier
   - Runs a GBM Monte-Carlo simulation forward from there

2. **Calibrated Regime-Switching**
   - Trains a Gaussian Mixture Model on historical returns up to the as-of date
   - User previews the fitted regime states and selects one to force every simulated path to start in
   - Runs a regime-switching Monte-Carlo simulation forward from that state

### Stochastic Forecasting
EquiTrack includes a Monte Carlo simulation engine for forecasting potential future portfolio paths based on historical return characteristics.

Rather than projecting a single deterministic path, the system generates many possible future trajectories, allowing the distribution of potential outcomes to be analyzed for a more accurate understanding of downside risk.

Each simulation produces a potential future equity curve. Running N simulations produces a distribution of portfolio outcomes from which percentile bands, terminal value statistics, and drawdown probabilities are derived.

Four simulation models are currently supported:
  - **GBM** — Geometric Brownian Motion with constant drift and volatility
  - **Heston** — Stochastic volatility model with historically calibrated parameters
  - **GJR-GARCH(1,1)** — Conditional volatility model with MLE-optimized parameters
  - **GMM** -- Gaussian Mixture model with Expectation-Minimization optimized parameters

## Geometric Brownian Motion (GBM)
The baseline simulation models future daily portfolio returns as log-normal with constant parameters:
  - Drift (expected return)
  - Volatility

These can be estimated using the full historical window, a rolling window, or exponentially weighted moving averages (EWMA).

## Heston Stochastic Volatility
The Heston model extends GBM by allowing volatility itself to follow a mean-reverting stochastic process, capturing volatility clustering and leverage effects observed in real markets.

Parameters (v0, θ, κ, ξ, ρ) are calibrated from historical rolling realized variance using least-squares regression. Due to the absence of implied volatility surfaces, all calibration is backward-looking from realized data rather than option-implied.

## GJR-GARCH(1,1)
The GJR-GARCH model treats variance as conditionally deterministic — each day's variance is a function of the previous day's variance and squared shock, with an asymmetric leverage term for negative returns.

Parameters are optimized via maximum likelihood estimation (MLE) under Student-t innovations. The optimizer uses a two-phase approach (Nelder-Mead → L-BFGS-B) with unconstrained reparameterization and variance targeting. Its performance has been validated in rolling backtests against the `arch` package reference implementation.

## GMM
The Gaussian Mixture model classifies all the realized trading days into n regime states. It is trained using the Expectation-Minimization algorithm.
The GMM will compute the probability that each trading day belongs to a given regime, and then from the results, compute the daily mean and standard 
deviation.

To simulate paths from this regime-switching model, the engine will first calculate the nowcasted probabilities of what regime the t0 trading day belongs to, and then use the transition probability matrix to predict the next day's regime. It will then simulate the next day's return from that regime's distribution (mean, std) and repeat.


## Forecasting Outputs
From the simulated distribution of equity curves, EquiTrack computes forecast statistics including:
  - Median projected equity curve
  - Percentile bands (e.g., 5th / 95th percentiles)
  - Forecasted return metrics
  - Forecasted volatility estimates
  - Projected drawdown behavior

These outputs allow users to visualize expected portfolio growth as well as downside risk under stochastic market dynamics.

# Visualization
Forecast and stress-test results are visualized in the frontend using interactive equity curves, including:
  - Historical equity curve, up to the forecast/as-of date
  - Actual-realized equity curve (stress tests only, when the as-of date is in the past) — what really happened afterward, shown as a real, non-simulated comparison baseline
  - Simulated median trajectory
  - Percentile confidence bands (p10–p90)

Charts are rendered using Recharts and automatically render when forecasts or stress tests are generated.


## Architecture Overview

The system is intentionaly layered to separate pure analytical computation from API orchestration and exposure.
Backend modules align with current code layout:

- `services/analysis_service.py`
  - Baseline portfolio analysis orchestration
- `services/forecast_service.py`
  - Deterministic and stochastic (GBM/Heston/GARCH/GMM regime-switching) forecast orchestration
- `services/stress_service.py`
  - Stress test orchestration 
- `services/store_singleton.py`
  - Stores completed analysis artifacts for downstream forecast/stress_service calls
- `engines/regime_engine.py`
  - Shared GMM regime-fitting and regime-switching Monte Carlo engine, used by both forecasting and stress testing services

---

## API Endpoints

- `GET /api/health`
  - Health check

- `POST /api/holdings/validate`
  - Validates ticker/date and resolves next valid trading day/price when needded

- `POST /api/analyze`
  - Baseline portfolio analytics

- `POST /api/forecast`
  - Deterministic or stochastic forecast projection for a previously-analyzed portfolio

- `POST /api/stress/deterministic_regime`
  - One-shot deterministic-regime stress test: user-specified drift shift/volatility multiplier, forward GBM Monte Carlo from a chosen as-of date

- `POST /api/stress/calibrated_regime/preview`
  - Step 1 of the calibrated-regime stress test: fits a GMM on returns up to a chosen as-of date, returns each regime state's stats for the user to inspect

- `POST /api/stress/calibrated_regime/stress`
  - Step 2 of the calibrated-regime stress test: forces every simulated path to start in a user-selected state, runs the regime-switching Monte Carlo forward

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

Production-facing code from the backend has automated test coverage which is automatically run on every commit, ensuring that every deployment is regression-tested. The test suite contains unit-level testing for each module as well as integration tests for the core workflows.

----

## CI / CD Pipeline

EquiTrack uses GitHub Actions for continuous integration.

- Automated backend test suite runs on every pull request.
- Merges to `main` require passing status checks.
- Backend auto-deploys to Render on successful merge.
- Frontend auto-deploys to Vercel on push to `main`.

This ensures code correctness before deployment and enforces disciplined development workflow.

----

**AI usage disclosure:** Claude Sonnet 5 was used to assist in generating parts of the pytest test suite.

---

## Notes / Disclaimers

### Responsiveness

The EquiTrack interface is currently optimized for desktop usage. Mobile responsiveness/compatibility is planned for a future iteration. 