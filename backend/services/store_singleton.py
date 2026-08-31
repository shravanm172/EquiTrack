from __future__ import annotations

import pandas as pd

from services.analysis_store import AnalysisStore

# 30 min TTL, adjust as needed for dev
analysis_store = AnalysisStore(ttl_seconds=1800, max_items=5000)


def get_cached_returns_and_starting_cash(
    analysis_id: str,
    source: str,
) -> tuple[pd.Series, float, dict]:
    """
    Load a cached return series, starting cash, and inputs from
    analysis_store. Generic lookup, not owned by any one service --
    forecast_service.py and stress_service.py both depend on this.
    """
    item = analysis_store.get(analysis_id)
    if item is None:
        raise ValueError("analysis_id not found or expired. Re-run analysis.")

    kind = item.get("kind")
    cached_inputs = item.get("inputs", {})

    if kind != "analyze":
        raise ValueError(f"Unsupported cached analysis kind: {kind}")

    if source != "baseline":
        raise ValueError("source must be 'baseline'.")

    port_r = item["portfolio_returns"]
    starting_cash = float(cached_inputs["starting_cash"])
    return port_r, starting_cash, cached_inputs