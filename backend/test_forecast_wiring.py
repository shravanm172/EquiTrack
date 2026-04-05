"""Smoke test: all three stochastic forecast models through the service layer."""

from services.forecast_service import forecast_portfolio
from services.analysis_service import analyze_portfolio

# Step 1: Create an analysis to get an analysis_id
payload = {
    "portfolio": {
        "holdings": [
            {"ticker": "AAPL", "weight": 0.25},
            {"ticker": "MSFT", "weight": 0.25},
            {"ticker": "GOOGL", "weight": 0.25},
            {"ticker": "NVDA", "weight": 0.25},
        ],
        "starting_cash": 100000,
    },
    "date_range": {"start": "2020-01-01", "end": "2025-12-31"},
}

result = analyze_portfolio(payload)
aid = result["analysis_id"]
print(f"analysis_id: {aid}")
tickers = result["inputs"].get("tickers", "NOT PRESENT")
print(f"cached tickers: {tickers}")
print()

# Step 2: Test GBM (existing, should still work)
print("=== GBM ===")
gbm = forecast_portfolio({
    "analysis_id": aid,
    "source": "baseline",
    "forecast": {"type": "stochastic", "model": "gbm", "days": 30, "simulations": 500},
})
print(f"  model: {gbm['inputs']['forecast']['model']}")
print(f"  median terminal: {gbm['terminal']['median_terminal_value']}")
print(f"  prob loss: {gbm['terminal']['probability_of_loss']}")
print()

# Step 3: Test Heston
print("=== HESTON ===")
hes = forecast_portfolio({
    "analysis_id": aid,
    "source": "baseline",
    "forecast": {"type": "stochastic", "model": "heston", "days": 30, "simulations": 500},
})
print(f"  model: {hes['inputs']['forecast']['model']}")
print(f"  median terminal: {hes['terminal']['median_terminal_value']}")
print(f"  prob loss: {hes['terminal']['probability_of_loss']}")
if "variance" in hes:
    print(f"  median terminal variance: {hes['variance']['median_terminal_variance']:.6f}")
if "calibrated_params" in hes:
    print(f"  calibrated params keys: {list(hes['calibrated_params'].keys())}")
print()

# Step 4: Test GARCH
print("=== GARCH ===")
gar = forecast_portfolio({
    "analysis_id": aid,
    "source": "baseline",
    "forecast": {"type": "stochastic", "model": "garch", "days": 30, "simulations": 500},
})
print(f"  model: {gar['inputs']['forecast']['model']}")
print(f"  median terminal: {gar['terminal']['median_terminal_value']}")
print(f"  prob loss: {gar['terminal']['probability_of_loss']}")
if "variance" in gar:
    print(f"  median terminal variance: {gar['variance']['median_terminal_variance']:.6f}")
if "calibrated_params" in gar:
    print(f"  calibrated params keys: {list(gar['calibrated_params'].keys())}")
print()

# Step 5: Test backward compat (no model field = defaults to gbm)
print("=== DEFAULT (no model field) ===")
default = forecast_portfolio({
    "analysis_id": aid,
    "source": "baseline",
    "forecast": {"type": "stochastic", "days": 30, "simulations": 500},
})
print(f"  model: {default['inputs']['forecast']['model']}")
print(f"  median terminal: {default['terminal']['median_terminal_value']}")
print()
print("ALL TESTS PASSED")
