"""Reference policies every submission is compared with."""
import numpy as np
from scipy.stats import norm

from .evaluate import Policy, Obs


def _d1(S, K, sig, tau):
    return (np.log(S / K) + 0.5 * sig**2 * tau) / (sig * np.sqrt(tau))


class NoHedge(Policy):
    name = "no-hedge"

    def act(self, obs: Obs):
        return np.zeros(self.info.n)


class BlackScholes(Policy):
    """Delta hedge using the currently observed volatility sqrt(v_t)."""
    name = "black-scholes"

    def act(self, obs: Obs):
        sig = np.sqrt(np.maximum(obs.v[:, -1], 1e-8))
        return norm.cdf(_d1(obs.S[:, -1], self.info.K, sig, obs.tau))


class WhalleyWilmott(Policy):
    """Whalley & Wilmott (1997): keep the position inside a no-trade band around
    Black-Scholes delta; trade only to the band's edge. Band half-width
    H = (3/2 * cost * S * Gamma^2 / risk_aversion)^(1/3)."""
    name = "whalley-wilmott"

    def __init__(self, risk_aversion=1.0):
        self.ra = risk_aversion

    def act(self, obs: Obs):
        S, K = obs.S[:, -1], self.info.K
        sig = np.sqrt(np.maximum(obs.v[:, -1], 1e-8))
        d1 = _d1(S, K, sig, obs.tau)
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sig * np.sqrt(obs.tau))
        H = (1.5 * self.info.cost * S * gamma**2 / self.ra) ** (1 / 3)
        if obs.t == 0:
            return delta
        return np.clip(obs.position, delta - H, delta + H)


BASELINES = {p.name: p for p in (NoHedge, BlackScholes, WhalleyWilmott)}
