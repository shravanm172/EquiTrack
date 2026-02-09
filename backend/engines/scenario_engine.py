from __future__ import annotations

import pandas as pd


def apply_price_shock(prices: pd.DataFrame, shock_date: str, shock_pct: float) -> pd.DataFrame:
    """
    Apply a multiplicative price shock to ALL assets on and after shock_date.

    shock_pct:
      -0.10 means -10%
       0.05 means +5%

    This is a simple stress-test model:
      P_t(shocked) = P_t * (1 + shock_pct)  for all t >= shock_date
    """
    if prices.empty:
        return prices.copy()

    shocked = prices.copy()

    shock_ts = pd.to_datetime(shock_date, errors="raise")

    # Apply to dates on/after shock date
    mask = shocked.index >= shock_ts
    shocked.loc[mask, :] = shocked.loc[mask, :] * (1.0 + float(shock_pct))

    return shocked


def apply_shock_with_linear_rebound(
    prices: pd.DataFrame,
    shock_date: str,
    shock_pct: float,
    rebound_days: int,
) -> pd.DataFrame:
    """
    Apply a shock that begins on (or the next trading day after) shock_date,
    then linearly rebounds back to baseline over rebound_days trading days.

    - shock_pct: -0.10 means prices drop 10% on shock start.
    - rebound_days: number of trading days to recover back to multiplier=1.0.
      Example: rebound_days=10 means by the 10th trading day after start
      the multiplier is back to 1.0.

    The multiplier is applied to ALL tickers.
    """
    if prices.empty:
        return prices.copy()

    if rebound_days < 1:
        raise ValueError("rebound_days must be >= 1")

    shocked = prices.copy()
    idx = shocked.index

    shock_ts = pd.to_datetime(shock_date, errors="raise")

    # Find first trading day on/after shock_date
    start_pos = idx.searchsorted(shock_ts)
    if start_pos >= len(idx):
        # shock_date is after all data; no change
        return shocked

    end_pos = min(start_pos + rebound_days, len(idx) - 1)

    # Build per-day multipliers
    # Day 0 -> 1 + shock_pct
    # Day rebound_days -> 1.0
    m0 = 1.0 + float(shock_pct)

    # Number of steps from start_pos to end_pos
    steps = end_pos - start_pos
    if steps == 0:
        multipliers = pd.Series([m0], index=idx[start_pos : start_pos + 1])
    else:
        # inclusive linspace start->end
        multipliers = pd.Series(
            [m0 + (1.0 - m0) * (i / steps) for i in range(steps + 1)],
            index=idx[start_pos : end_pos + 1],
        )

    # Apply multipliers only during the rebound window
    shocked.loc[multipliers.index, :] = shocked.loc[multipliers.index, :].mul(
        multipliers, axis=0
    )

    # After end_pos, multiplier is 1.0 (so no further changes)
    return shocked


def apply_regime_shift(
    prices: pd.DataFrame,
    shock_date: str,
    vol_mult: float,
    drift_shift: float,
) -> pd.DataFrame:
    """
    Regime shift stress test: after shock_date, make returns more volatile and
    add a constant drift shift (often negative).

    - vol_mult: multiplies the "noise" part of returns (e.g., 1.5 -> 50% more volatile)
    - drift_shift: constant added to daily returns after shock (e.g., -0.0005 ≈ -0.05% per day)

    Method:
    1) Convert prices -> daily returns r_t
    2) For t >= shock_date:
         r'_t = mean(r_post) + vol_mult * (r_t - mean(r_post)) + drift_shift
       (We scale deviations from the post-shock mean to increase volatility, then add drift bias.)
    3) Rebuild shocked prices by compounding r'_t from the original starting price.
    """
    if prices.empty:
        return prices.copy()

    shocked = prices.copy()
    idx = shocked.index
    shock_ts = pd.to_datetime(shock_date, errors="raise")

    # Find first trading day on/after shock_date
    start_pos = idx.searchsorted(shock_ts)
    if start_pos >= len(idx):
        return shocked  # shock date is beyond data range, no change

    # Compute simple daily returns
    rets = shocked.pct_change().dropna()

    # Slice returns from the first affected return date onward
    # Note: returns are for (t-1 -> t), so align start on rets.index >= idx[start_pos]
    start_date = idx[start_pos]
    post_mask = rets.index >= start_date
    post_rets = rets.loc[post_mask].copy()

    if post_rets.empty:
        return shocked

    # Mean return per asset in the post period (vector)
    mu = post_rets.mean()

    # Scale volatility around mean and apply drift shift
    post_rets = mu + float(vol_mult) * (post_rets - mu) + float(drift_shift)

    # Rebuild prices from start_date onward using compounded returns
    # Start from the baseline price on the day BEFORE start_date (anchor)
    anchor_pos = max(start_pos - 1, 0)
    anchor_date = idx[anchor_pos]
    anchor_prices = shocked.loc[anchor_date]

    # Build shocked price path for dates >= start_date
    shocked_path = (1.0 + post_rets).cumprod()
    shocked_prices = shocked_path.mul(anchor_prices, axis=1)

    # Insert into shocked dataframe
    shocked.loc[shocked_prices.index, shocked_prices.columns] = shocked_prices

    return shocked