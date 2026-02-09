from providers.market_data import fetch_price_history
from engines.scenario_engine import apply_shock_with_linear_rebound

ph = fetch_price_history(["AAPL", "MSFT"], start="2025-01-01", end="2025-02-01")
prices = ph.prices

shocked = apply_shock_with_linear_rebound(
    prices,
    shock_date="2025-01-15",
    shock_pct=-0.10,
    rebound_days=5,
)

print("BASELINE:")
print(prices.loc["2025-01-13":"2025-01-24"])

print("\nSHOCKED (linear rebound):")
print(shocked.loc["2025-01-13":"2025-01-24"])
