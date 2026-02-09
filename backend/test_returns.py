from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns

ph = fetch_price_history(["AAPL", "MSFT"], start="2025-01-01", end="2025-02-01")

returns = prices_to_returns(ph.prices)

print("PRICES:")
print(ph.prices.head())

print("\nRETURNS:")
print(returns.head())
