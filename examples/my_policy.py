"""A complete HedgeBench submission in 20 lines.

    hedgebench run gbm-cost --policy examples/my_policy.py:LazyDelta

Idea: Black-Scholes delta, but only rebalance when the position is off by
more than a fixed threshold. A simpler cousin of Whalley-Wilmott.
"""
import numpy as np
from scipy.stats import norm

import hedgebench as hb


class LazyDelta(hb.Policy):
    name = "lazy-delta-0.05"

    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def act(self, obs: hb.Obs) -> np.ndarray:
        S, K = obs.S[:, -1], self.info.K                 # only the past is visible
        sig = np.sqrt(obs.v[:, -1])
        d1 = (np.log(S / K) + 0.5 * sig**2 * obs.tau) / (sig * np.sqrt(obs.tau))
        delta = norm.cdf(d1)
        stale = np.abs(delta - obs.position) > self.threshold
        return np.where((obs.t == 0) | stale, delta, obs.position)
