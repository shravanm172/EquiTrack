from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class PriceHistory:
    prices: pd.DataFrame  # index=date, columns=tickers


def fetch_price_history(tickers: Iterable[str], start: str, end: str) -> PriceHistory:
    tickers_list = sorted({t.upper().strip() for t in tickers if t and t.strip()}) # Dedupe & clean tickers
    if not tickers_list:
        raise ValueError("No tickers provided.")

    df = yf.download(
        tickers=tickers_list,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    ) # Download daily historical data for the given tickers and date ranges

    # Normalize to a simple DataFrame: index=date, columns=tickers, values=close prices
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.levels[0]:
            prices = df["Close"].copy()
        elif "Adj Close" in df.columns.levels[0]:
            prices = df["Adj Close"].copy()
        else:
            prices = df.xs(df.columns.levels[0][0], axis=1, level=0).copy()
    else:
        # Single ticker case -> columns like Open/High/Low/Close/Volume
        if "Close" not in df.columns:
            raise ValueError("Unexpected yfinance response: missing Close column.")
        prices = df[["Close"]].copy()
        prices.columns = tickers_list

    prices = prices.dropna(how="all") # Drop rows where all tickers are NaN (non-trading days)
    if prices.empty:
        raise ValueError("No price data returned (bad tickers or empty date range).")

    return PriceHistory(prices=prices)
