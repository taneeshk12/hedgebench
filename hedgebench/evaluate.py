"""Run a policy on a task and score it.

The evaluator feeds the policy one date at a time and only ever shows it the past:
look-ahead is impossible by construction, whatever the policy's code does.
"""
import json
import platform
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np

from . import __version__
from .metrics import summarize, paired_bootstrap
from .tasks import Task, Episodes


@dataclass
class Info:
    """Given to Policy.reset() at the start of a batch of episodes."""
    n: int
    K: np.ndarray
    T: float
    n_steps: int
    cost: float


@dataclass
class Obs:
    """Given to Policy.act() at date t. Histories hold dates 0..t only."""
    t: int
    tau: float               # years left to expiry
    S: np.ndarray            # (n, t+1) price history
    v: np.ndarray            # (n, t+1) variance history
    position: np.ndarray     # (n,) shares currently held


class Policy:
    """Subclass this. Positions are shares held per short call (usually 0..1)."""
    name = "unnamed"

    def fit(self, task: Task):
        """Optional training. May use task.sample_train() and task.history()."""

    def reset(self, info: Info):
        self.info = info

    def act(self, obs: Obs) -> np.ndarray:
        raise NotImplementedError


def rollout(policy: Policy, task: Task, ep: Episodes):
    """Returns positions (n, n_steps), P&L (n,) and trading costs (n,)."""
    n, N = len(ep), task.n_steps
    policy.reset(Info(n=n, K=ep.K.copy(), T=task.T, n_steps=N, cost=task.cost))
    pos = np.zeros(n)
    P = np.empty((n, N))
    for t in range(N):
        obs = Obs(t=t, tau=(N - t) * task.T / N, S=ep.S[:, :t + 1].copy(),
                  v=ep.v[:, :t + 1].copy(), position=pos.copy())
        a = np.asarray(policy.act(obs), dtype=float).reshape(n)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{policy.name} returned a non-finite position at t={t}")
        P[:, t] = pos = a
    full = np.hstack([np.zeros((n, 1)), P, np.zeros((n, 1))])     # flat before and after
    costs = task.cost * (np.abs(np.diff(full, axis=1)) * ep.S).sum(1)
    gains = (P * np.diff(ep.S, axis=1)).sum(1)
    payoff = np.maximum(ep.S[:, -1] - ep.K, 0.0)
    return P, ep.premium + gains - costs - payoff, costs


def evaluate(policy: Policy, task: Task, baseline: "Policy | None" = None, alpha=0.95, fit=True):
    """Fit (optional), roll out on the task's test set, and score.
    If `baseline` is given, adds a paired bootstrap comparison against it."""
    if fit:
        policy.fit(task)
    ep = task.test_episodes()
    _, pnl, costs = rollout(policy, task, ep)
    result = {
        "policy": policy.name, "task": task.name, "n_test": len(ep),
        "metrics": summarize(pnl, costs, alpha),
        "hedgebench": __version__, "python": platform.python_version(),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    if baseline is not None:
        baseline.fit(task)
        _, pnl_b, _ = rollout(baseline, task, ep)
        result["vs"] = {"baseline": baseline.name, **paired_bootstrap(pnl, pnl_b, alpha, block=task.block)}
    return result, pnl


def save(result, path):
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
