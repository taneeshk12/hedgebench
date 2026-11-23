"""Market simulators. Each returns (S, v): arrays of shape (n_paths, n_steps + 1)
holding the asset price and the instantaneous variance at each hedging date."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Heston:
    v0: "float | np.ndarray"     # initial variance (scalar or one per path)
    kappa: float                 # mean-reversion speed
    theta: "float | np.ndarray"  # long-run variance (scalar or one per path)
    xi: float                    # vol of variance
    rho: float                   # price/variance correlation


@dataclass
class Jumps:
    lam: float      # jumps per year
    mu: float       # mean log jump
    sig: float      # std of log jump


def gbm(sigma, T, n_steps, n_paths, rng, S0=100.0):
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    logS = np.cumsum(-0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z, axis=1)
    S = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), logS]))
    return S, np.full_like(S, sigma**2)


def bates(p: Heston, T, n_steps, n_paths, rng, jumps: "Jumps | None" = None, S0=100.0, substeps=8):
    """Heston (plus optional Merton jumps = Bates), Euler with full truncation,
    simulated on a finer grid and sampled at hedging dates. Martingale (zero drift)."""
    dt = T / (n_steps * substeps)
    v = np.broadcast_to(np.asarray(p.v0, float), (n_paths,)).copy()
    theta = np.broadcast_to(np.asarray(p.theta, float), (n_paths,))
    s = np.full(n_paths, S0)
    S = np.empty((n_paths, n_steps + 1)); V = np.empty_like(S)
    S[:, 0], V[:, 0] = s, v
    comp = jumps.lam * (np.exp(jumps.mu + 0.5 * jumps.sig**2) - 1) if jumps else 0.0
    for i in range(n_steps):
        for _ in range(substeps):
            z1 = rng.standard_normal(n_paths)
            z2 = p.rho * z1 + np.sqrt(1 - p.rho**2) * rng.standard_normal(n_paths)
            vp = np.maximum(v, 0.0)
            s = s * np.exp((-comp - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1)
            if jumps:
                k = rng.poisson(jumps.lam * dt, n_paths)
                s = s * np.exp(k * jumps.mu + np.sqrt(k) * jumps.sig * rng.standard_normal(n_paths))
            v = v + p.kappa * (theta - vp) * dt + p.xi * np.sqrt(vp * dt) * z2
        S[:, i + 1], V[:, i + 1] = s, np.maximum(v, 0.0)
    return S, V


def calibrate(price, vol_index, per_year, jump_threshold=3.0):
    """Fit Heston + jumps to a daily price series and its implied-vol index (in %).
    Variance proxy v = (vol/100)^2; AR(1) on v gives kappa/theta; residuals give xi;
    a 'jump' is a daily move > jump_threshold x the move the vol index predicted."""
    price, vol_index = np.asarray(price, float), np.asarray(vol_index, float)
    dt = 1 / per_year
    v = (vol_index / 100) ** 2
    ret = np.diff(np.log(price))
    b, a = np.polyfit(v[:-1], v[1:], 1)
    resid = v[1:] - (a + b * v[:-1])
    heston = Heston(v0=a / (1 - b), kappa=(1 - b) / dt, theta=a / (1 - b),
                    xi=float(np.std(resid / np.sqrt(v[:-1] * dt))),
                    rho=float(np.corrcoef(ret, np.diff(v))[0, 1]))
    j = ret[np.abs(ret / (vol_index[:-1] / 100 * np.sqrt(dt))) > jump_threshold]
    jumps = Jumps(lam=len(j) / (len(ret) * dt), mu=float(j.mean()), sig=float(j.std())) if len(j) > 1 else None
    return heston, jumps
