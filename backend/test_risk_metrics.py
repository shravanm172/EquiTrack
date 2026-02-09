from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns, portfolio_returns
from engines.analytics_engine import equity_curve, annualized_volatility, max_drawdown, annualized_return

ph = fetch_price_history(["AAPL", "MSFT"], start="2025-01-01", end="2025-02-01")
asset_r = prices_to_returns(ph.prices)

weights = {"AAPL": 0.6, "MSFT": 0.4}
port_r = portfolio_returns(asset_r, weights)

curve = equity_curve(port_r, 100_000)

vol = annualized_volatility(port_r)
mdd = max_drawdown(curve)
ann_ret = annualized_return(curve)

print("Annualized Return:", round(ann_ret * 100, 2), "%")
print("Annualized Volatility:", round(vol * 100, 2), "%")
print("Max Drawdown:", round(mdd * 100, 2), "%")
