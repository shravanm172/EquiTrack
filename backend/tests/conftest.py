# AI Disclosure: This file includes content generated with GPT-5.2.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


@pytest.fixture
def synthetic_returns() -> pd.Series:
    """
    Deterministic synthetic portfolio returns series, shared across the
    GBM/GARCH/Regime verification tests. NOT meant to resemble real markets
    -- these are verification tests (does the code compute what it's
    supposed to), not validation tests (is the model accurate), which is
    what the real-data backtests are for. Long enough and structured enough
    (two visibly different vol/mean blocks) for GARCH's MLE and the regime
    GMM to fit stable, non-degenerate parameters. Fully reproducible: fixed
    seed, no network access, no randomness beyond what's seeded here.
    """
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2018-01-01", periods=750)

    calm = rng.normal(0.0006, 0.008, size=300)
    crisis = rng.normal(-0.0015, 0.025, size=150)
    calm2 = rng.normal(0.0006, 0.008, size=300)

    returns = np.concatenate([calm, crisis, calm2])
    return pd.Series(returns, index=idx, name="portfolio")
