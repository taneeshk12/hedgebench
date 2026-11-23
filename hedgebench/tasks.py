"""Benchmark tasks.

Every task is: hedge a SHORT European call over `n_steps` dates, trading the
underlying with proportional cost `cost`. A task provides
  * test_episodes(): the fixed test set every submission is scored on
  * sample_train(n, seed): unlimited training episodes from a simulator. For real-data
    tasks the simulator is calibrated on the task's training period only.
  * history(): for real-data tasks, the raw training-period data (never the test period)

Prices are normalised so every episode starts at S0 = 100.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import data, sim

S0 = 100.0
# Simulated tasks use fixed parameters (calibrated once to SPY/VIX 2005-2025) so
# they are exactly reproducible and need no internet.
SPY_HESTON = dict(kappa=9.83, theta=0.0443, xi=0.65, rho=-0.75)
SPY_JUMPS = sim.Jumps(lam=1.0, mu=-0.030, sig=0.056)


@dataclass
class Episodes:
    S: np.ndarray            # (n, n_steps+1) prices
    v: np.ndarray            # (n, n_steps+1) observed variance (model variance, or vol index^2)
    K: np.ndarray            # (n,) strikes
    premium: np.ndarray      # (n,) premium received (0 when not part of the task)
    dates: Optional[list] = None

    def __len__(self):
        return len(self.S)


@dataclass
class Task:
    name: str
    description: str
    n_steps: int
    T: float                 # years to expiry at the first date
    cost: float              # proportional cost per unit traded
    kind: str                # "sim" or "real"
    block: int               # bootstrap block length (>1 when test episodes overlap)
    _test: Callable = field(repr=False)
    _train: Callable = field(repr=False)
    _history: Optional[Callable] = field(default=None, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def test_episodes(self) -> Episodes:
        if "test" not in self._cache:
            self._cache["test"] = self._test()
        return self._cache["test"]

    def sample_train(self, n, seed=0) -> Episodes:
        return self._train(n, np.random.default_rng(seed))

    def history(self) -> pd.DataFrame:
        if self._history is None:
            raise ValueError(f"{self.name} is simulated: use sample_train()")
        return self._history()


def _atm(S, v, K=S0):
    n = len(S)
    return Episodes(S, v, np.full(n, float(K)), np.zeros(n))


def _sim_task(name, desc, cost, make):
    n_steps, T = 30, 30 / 252
    def train(n, rng):
        S, v = make(n, rng, n_steps, T)
        return Episodes(S, v, rng.uniform(94, 106, n), np.zeros(n))
    def test():
        S, v = make(20_000, np.random.default_rng(20240101), n_steps, T)
        return _atm(S, v)
    return Task(name, desc, n_steps, T, cost, "sim", 1, test, train)


def _windows(df, start, end, n_steps, stride):
    seg = df[(df.index >= start) & (df.index < end)]
    p, vol = seg["price"].values, (seg["vol"].values / 100) ** 2
    idx = range(0, len(seg) - n_steps, stride)
    S = np.array([p[i:i + n_steps + 1] / p[i] * S0 for i in idx])
    v = np.array([vol[i:i + n_steps + 1] for i in idx])
    return S, v, [seg.index[i].date().isoformat() for i in idx]


def _calibrated_train(history, per_year, n_steps, T):
    """Bates simulator fitted to the training period, with each path's vol regime
    drawn from the training period's vol-index levels (domain randomization)."""
    def train(n, rng):
        df = history()
        h, j = sim.calibrate(df["price"], df["vol"], per_year)
        hv = (df["vol"].values / 100) ** 2
        h.v0, h.theta = rng.choice(hv, n), rng.choice(hv, n)
        S, v = sim.bates(h, T, n_steps, n, rng, jumps=j)
        return Episodes(S, v, rng.uniform(94, 106, n), np.zeros(n))
    return train


def _spy_real():
    n_steps, T, split = 30, 30 / 252, "2016-01-01"
    history = lambda: data.spy_vix().loc[:split].iloc[:-1]
    def test():
        S, v, d = _windows(data.spy_vix(), split, "2026-01-01", n_steps, stride=5)
        e = _atm(S, v); e.dates = d
        return e
    return Task("spy-real", "Real SPY 30-trading-day windows, 2016-2025 (every 5th day), ATM call, "
                "0.2% cost. Vol input: VIX. Train period: 2005-2015.",
                n_steps, T, 0.002, "real", 6, test, _calibrated_train(history, 252, n_steps, T), history)


def _btc_options_real():
    n_steps, T, split = 30, 30 / 365, pd.Timestamp("2024-01-01")
    history = lambda: data.btc_dvol().loc[:split - pd.Timedelta(days=1)]
    def test():
        mkt, calls = data.btc_dvol(), data.btc_monthly_calls(n_steps)
        calls = calls[(calls.index >= split) & (calls["expiry"] < pd.Timestamp("2026-01-01"))]
        S, v = [], []
        for start in calls.index:
            seg = mkt.loc[start:start + pd.Timedelta(days=n_steps)]
            S.append(seg["price"].values / seg["price"].values[0] * S0)
            v.append((seg["vol"].values / 100) ** 2)
        return Episodes(np.array(S), np.array(v), (calls["strike"] / calls["S0"] * S0).values,
                        (calls["premium"] / calls["S0"] * S0).values, list(calls["instrument"]))
    return Task("btc-options-real", "Real monthly BTC calls on Deribit, 2024-2025: sold at their real "
                "market price 30 days before expiry, hedged daily with BTC, 0.1% cost. Vol input: DVOL. "
                "Train period: 2021-03 to 2023-12.",
                n_steps, T, 0.001, "real", 1, test, _calibrated_train(history, 365, n_steps, T), history)


def _bates_make(jumps):
    def make(n, rng, n_steps, T):
        return sim.bates(sim.Heston(v0=SPY_HESTON["theta"], **SPY_HESTON), T, n_steps, n, rng, jumps=jumps)
    return make


TASKS = {t.name: t for t in [
    _sim_task("gbm-frictionless", "Black-Scholes world, 20% vol, no costs. Sanity check: "
              "Black-Scholes delta is near-optimal here.", 0.0,
              lambda n, rng, k, T: sim.gbm(0.2, T, k, n, rng)),
    _sim_task("gbm-cost", "Black-Scholes world, 20% vol, 0.2% costs.", 0.002,
              lambda n, rng, k, T: sim.gbm(0.2, T, k, n, rng)),
    _sim_task("heston-cost", "Heston stochastic volatility calibrated to SPY/VIX, 0.2% costs.", 0.002,
              _bates_make(None)),
    _sim_task("bates-cost", "Heston plus jumps (Bates) calibrated to SPY/VIX, 0.2% costs.", 0.002,
              _bates_make(SPY_JUMPS)),
    _spy_real(),
    _btc_options_real(),
]}


def get(name) -> Task:
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}. Available: {', '.join(TASKS)}")
    return TASKS[name]
