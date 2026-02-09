from providers.market_data import fetch_price_history
from engines.scenario_engine import apply_regime_shift

ph = fetch_price_history(["AAPL", "MSFT"], start="2025-01-01", end="2025-02-15")
prices = ph.prices

shocked = apply_regime_shift(
    prices,
    shock_date="2025-01-15",
    vol_mult=2.0,
    drift_shift=-0.001,  # ~ -0.1% per day
)

print("BASELINE (tail):")
print(prices.tail(8))

print("\nSHOCKED regime_shift (tail):")
print(shocked.tail(8))