from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns, portfolio_returns

ph = fetch_price_history(["AAPL", "MSFT"], start="2025-01-01", end="2025-02-01")
asset_r = prices_to_returns(ph.prices)

weights = {"AAPL": 0.6, "MSFT": 0.4}
port_r = portfolio_returns(asset_r, weights)

print("ASSET RETURNS:")
print(asset_r.head())

print("\nPORTFOLIO RETURNS (60% AAPL, 40% MSFT):")
print(port_r.head())
