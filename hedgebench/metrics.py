"""Scores. Main score: CVaR (expected shortfall) of the hedging P&L, the average of
the worst (1 - alpha) outcomes, reported as a positive loss. Lower is better."""
import numpy as np


def cvar(pnl, alpha=0.95):
    s = np.sort(pnl)
    k = max(1, int(np.floor(len(s) * (1 - alpha))))
    return float(-s[:k].mean())


def summarize(pnl, costs, alpha=0.95):
    return {
        "cvar": cvar(pnl, alpha),
        "var": float(-np.quantile(pnl, 1 - alpha)),
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
        "worst": float(pnl.min()),
        "cost": float(costs.mean()),
    }


def paired_bootstrap(pnl, pnl_base, alpha=0.95, block=1, n_boot=2000, seed=0):
    """Is the policy really better than the baseline on the same test episodes?
    Moving-block bootstrap (block > 1 for overlapping real-data windows).
    Returns the CVaR and mean differences (policy - baseline) with 95% intervals,
    and the share of resamples in which the policy's CVaR is lower."""
    rng = np.random.default_rng(seed)
    n = len(pnl)
    starts = np.arange(n - block + 1)
    n_blocks = int(np.ceil(n / block))
    d_cvar, d_mean = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = (rng.choice(starts, n_blocks)[:, None] + np.arange(block)).ravel()[:n]
        d_cvar[b] = cvar(pnl[idx], alpha) - cvar(pnl_base[idx], alpha)
        d_mean[b] = pnl[idx].mean() - pnl_base[idx].mean()
    return {
        "cvar_diff": cvar(pnl, alpha) - cvar(pnl_base, alpha),
        "cvar_diff_ci": [float(np.quantile(d_cvar, .025)), float(np.quantile(d_cvar, .975))],
        "mean_diff": float(pnl.mean() - pnl_base.mean()),
        "mean_diff_ci": [float(np.quantile(d_mean, .025)), float(np.quantile(d_mean, .975))],
        "p_better": float((d_cvar < 0).mean()),
    }
