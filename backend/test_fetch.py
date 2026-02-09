from providers.market_data import fetch_price_history

ph = fetch_price_history(["AAPL", "MSFT"], start="2025-01-01", end="2025-02-01")
print(ph.prices.head())
print(ph.prices.tail())
