from providers.market_data import fetch_price_history
from engines.portfolio_engine import prices_to_returns, portfolio_returns
from engines.analytics_engine import (
    equity_curve,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)

ph = fetch_price_history(["AAPL", "MSFT"], start="2025-01-01", end="2025-02-01")
asset_r = prices_to_returns(ph.prices)

weights = {"AAPL": 0.6, "MSFT": 0.4}
port_r = portfolio_returns(asset_r, weights)

curve = equity_curve(port_r, 100_000)

print("Annualized Return:", round(annualized_return(curve) * 100, 2), "%")
print("Annualized Volatility:", round(annualized_volatility(port_r) * 100, 2), "%")
print("Max Drawdown:", round(max_drawdown(curve) * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio(port_r), 2))
