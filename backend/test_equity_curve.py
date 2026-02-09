from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns, portfolio_returns
from engines.analytics_engine import equity_curve

# 1) Fetch prices
ph = fetch_price_history(["AAPL", "MSFT"], start="2025-01-01", end="2025-02-01")

# 2) Asset returns
asset_r = prices_to_returns(ph.prices)

# 3) Portfolio returns (weights)
weights = {"AAPL": 0.6, "MSFT": 0.4}
port_r = portfolio_returns(asset_r, weights)

# 4) Equity curve
starting_cash = 100_000
curve = equity_curve(port_r, starting_cash)

print("PORTFOLIO RETURNS:")
print(port_r.head())

print("\nEQUITY CURVE:")
print(curve.head())
