# For stochastic forecasting
import numpy as np

def simulate_gbm_path(s0, mu, sigma, T, N):
    '''
        Return path
    '''
    dt = T / N
    path = [s0]
    s = s0
    for _ in range(N):
        z = np.random.normal(0,1)
        s = s * np.exp(((mu - 0.5 * sigma**2)*dt) + (sigma * np.sqrt(dt) * z))
        path.append(s)

    return path

def simulate_many_paths(s0, mu, sigma, T, N, n):
    '''
        Return list of paths
    '''
    paths = []
    for _ in range(n):
        paths.append(simulate_gbm_path(s0, mu, sigma, T, N))
    return np.array(paths)

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


def run_stochastic_forecast(s0, mu, sigma, T, N, n):
    '''
        Single entrypoint for stochastic engine
    '''
    paths = simulate_many_paths(s0, mu, sigma, T, N, n)

    terminal = summarize_terminal_metrics(paths)
    drawdown = summarize_drawdown_metrics(paths)
    path_metrics = summarize_path_metrics(paths)

    return {
        "paths": paths,
        "terminal": terminal,
        "drawdown": drawdown,
        "path_metrics": path_metrics,
    }