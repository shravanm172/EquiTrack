# EquiTrack

**EquiTrack** is a full-stack portfolio analytics and market simulation platform designed to explore how equity portfolios behave under different market conditions.  
It combines real market data, quantitative risk metrics, and scenario-based stress testing in a modular, extensible architecture.

The project is intentionally built as an evolving system: today it focuses on deterministic portfolio analysis and stress scenarios; future iterations will introduce stochastic Monte Carlo simulation and advanced risk modeling.


## Core Features

### Portfolio Analytics
- Portfolio defined as a set of tickers and weights
  Weights may be provided directly by the user or derived from user-specified share counts using market prices on the portfolio start date.
- Real historical price data fetched over a user-defined date range
- Analysis pipeline:
  1. Prices → daily returns  
  2. Asset returns → weighted portfolio returns  
  3. Portfolio returns → equity curve
- Risk metrics computed from portfolio behavior:
  - Annualized return
  - Annualized volatility
  - Maximum drawdown
  - Sharpe ratio


### Scenario-Based Market Simulation (Stress Testing)

EquiTrack includes a simulation sandbox environment that allows users to simulate how a portfolio behaves under different market conditions.

Scenarios are implemented as price or return transformations

Currently supported scenarios:

1. **Permanent Shock**
   - Instant price level shock applied on a chosen date
   - Prices remain permanently re-scaled afterward

2. **Shock with Linear Rebound**
   - Initial price shock
   - Gradual, linear recovery back to baseline over a user-defined number of trading days

3. **Regime Shift (Volatility + Drift)**
   - Increase in return volatility after a shock date
   - Negative drift applied to returns, simulating prolonged market deterioration
   - Produces realistic bear market behavior

> Monte Carlo simulation is planned as a future scenario type built on this same regime shift foundation.


### Baseline vs Scenario Comparison

For stress testing, EquiTrack:
- Computes a **baseline analysis** using historical data
- Computes a **scenario analysis** using transformed prices/returns
- Produces **delta metrics** to quantify the impact of the scenario:
  - Change in return
  - Change in volatility
  - Change in drawdown
  - Change in Sharpe ratio

This allows direct comparison between normal market conditions and stressed environments.

---

## Architecture Overview

The system is deliberately modular and scalable.

### `analysis_service`
- Core deterministic analytics engine
- Given prices → computes:
  - portfolio returns
  - equity curve
  - risk metrics

### `scenario_engine`
- Defines market scenarios as transformations
- Operates only on prices or returns
- Current implementations:
  - permanent shock
  - linear rebound shock
  - volatility + drift regime shift
- Designed to be extended (Monte Carlo planned)

### `stress_service`
- Performs stress testing
- Runs:
  - baseline analysis
  - scenario analysis
- Computes metric deltas for comparison

---

## API Endpoints

### Baseline Analysis
`POST /api/analyze`

Returns:
- equity curve
- portfolio risk metrics

### Stress Analysis
`POST /api/analyze_shock`

Returns:
- baseline results
- scenario results
- delta metrics (scenario − baseline)

Scenarios are selected via request payload configuration.


## Technology Stack

### Backend
- Python
- Flask
- pandas / NumPy
- REST API architecture

### Frontend (planned)
- React
- Interactive dashboard
- Scenario configuration UI
- Equity curve visualization

### CI/CD (planned)
- GitHub-integrated deployment (Vercel + Render)
- Automated testing and validation before deployment
- Full CI pipeline to be added as the project matures

## Development Setup

### Create and activate virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1


## Test Suite:
- test_build_and_analyze_portfolio_integration
    - test_analyze_from_prices_integration
        - test_prices_to_returns
        - test_portfolio_returns
        - test_equity_curve
        - test_risk_metrics
- test_analyze_with_shock_integration
    - test_scenarios
