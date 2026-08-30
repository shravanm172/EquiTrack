"""
General-purpose summary statistics for Monte Carlo simulation paths --
mean/median terminal value, bear/bull case, probability of loss, drawdown
risk, percentile path bands for charting, and terminal variance stats.

None of these functions know or care which model produced the paths (GBM,
Heston, GARCH, regime-switching, or anything else) -- they just take an
array of simulated paths and compute stats. Moved out of stochastic_engine.py
(which owns the forecast-specific model dispatch logic) so other callers,
like the stress-testing service, can reuse them without depending on the
forecast dispatcher or any single model's calibration logic.
"""
import numpy as np


def summarize_terminal_metrics(paths):
    """
    Compute terminal-value summary metrics from simulated paths.
    """
    paths = np.asarray(paths)
    terminal_values = paths[:, -1]
    s0 = paths[0, 0]

    median_terminal_value = np.median(terminal_values)
    mean_terminal_value = np.mean(terminal_values)
    bear_case = np.percentile(terminal_values, 10)
    bull_case = np.percentile(terminal_values, 90)
    probability_of_loss = np.mean(terminal_values < s0)

    return {
        "terminal_values": terminal_values,
        "mean_terminal_value": float(mean_terminal_value),
        "median_terminal_value": float(median_terminal_value),
        "bear_case": float(bear_case),
        "bull_case": float(bull_case),
        "probability_of_loss": float(probability_of_loss),
    }


def summarize_drawdown_metrics(paths):
    """
    Compute drawdown-based risk metrics from simulated paths.
    """
    paths = np.asarray(paths)

    running_peaks = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_peaks) / running_peaks
    max_drawdowns = np.min(drawdowns, axis=1)

    median_max_drawdown = np.median(max_drawdowns)
    prob_drawdown_gt_20 = np.mean(max_drawdowns <= -0.20)

    return {
        "max_drawdowns": max_drawdowns,
        "median_max_drawdown": float(median_max_drawdown),
        "prob_drawdown_gt_20": float(prob_drawdown_gt_20),
    }


def summarize_path_metrics(paths):
    """
    Compute percentile path summaries for chart visualization.
    """
    paths = np.asarray(paths)

    p10_path = np.percentile(paths, 10, axis=0)
    p25_path = np.percentile(paths, 25, axis=0)
    p50_path = np.median(paths, axis=0)
    p75_path = np.percentile(paths, 75, axis=0)
    p90_path = np.percentile(paths, 90, axis=0)

    return {
        "p10_path": p10_path,
        "p25_path": p25_path,
        "p50_path": p50_path,
        "p75_path": p75_path,
        "p90_path": p90_path,
    }


def summarize_variance_metrics(variance_paths):
    """
    Compute summary statistics for variance paths when available.
    """
    variance_paths = np.asarray(variance_paths)
    terminal_variances = variance_paths[:, -1]

    return {
        "terminal_variances": terminal_variances,
        "mean_terminal_variance": float(np.mean(terminal_variances)),
        "median_terminal_variance": float(np.median(terminal_variances)),
        "p10_terminal_variance": float(np.percentile(terminal_variances, 10)),
        "p90_terminal_variance": float(np.percentile(terminal_variances, 90)),
    }
