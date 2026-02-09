from providers.market_data import fetch_price_history
from engines.scenario_engine import apply_price_shock

ph = fetch_price_history(["AAPL", "MSFT"], start="2025-01-01", end="2025-02-01")
prices = ph.prices

shock_date = "2025-01-15"
shock_pct = -0.10

shocked = apply_price_shock(prices, shock_date=shock_date, shock_pct=shock_pct)

print("BASELINE prices around shock date:")
print(prices.loc["2025-01-13":"2025-01-17"])

print("\nSHOCKED prices around shock date:")
print(shocked.loc["2025-01-13":"2025-01-17"])