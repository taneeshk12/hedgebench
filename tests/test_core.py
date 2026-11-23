"""Fast tests on simulated tasks only (no internet needed)."""
import numpy as np
import pytest

import hedgebench as hb
from hedgebench.metrics import cvar, paired_bootstrap


class Peeker(hb.Policy):
    """Tries to read the future: must never see more than t+1 prices."""
    name = "peeker"

    def act(self, obs):
        assert obs.S.shape[1] == obs.t + 1 and obs.v.shape[1] == obs.t + 1
        return np.zeros(self.info.n)


class Mutator(hb.Policy):
    """Tries to change the data it is shown: must not affect the test set."""
    name = "mutator"

    def act(self, obs):
        obs.S[:] = 0.0
        return np.zeros(self.info.n)


def small(task_name, n=2000):
    task = hb.get(task_name)
    return task, task.sample_train(n, seed=1)


def test_no_lookahead():
    task, ep = small("gbm-cost")
    hb.rollout(Peeker(), task, ep)


def test_policy_cannot_change_data():
    task, ep = small("gbm-cost")
    before = ep.S.copy()
    hb.rollout(Mutator(), task, ep)
    assert np.array_equal(before, ep.S)


def test_black_scholes_hedges_frictionless_world():
    task, ep = small("gbm-frictionless", 5000)
    ep.K[:] = 100.0
    _, pnl_bs, _ = hb.rollout(hb.BlackScholes(), task, ep)
    _, pnl_none, _ = hb.rollout(hb.NoHedge(), task, ep)
    # daily delta hedging removes most of the risk; P&L ~ -premium
    assert pnl_bs.std() < 0.3 * pnl_none.std()
    assert abs(pnl_bs.mean() + 2.76) < 0.1          # Black-Scholes ATM price, 20% vol, 30/252y


def test_whalley_wilmott_trades_less_with_costs():
    task, ep = small("gbm-cost")
    _, _, c_bs = hb.rollout(hb.BlackScholes(), task, ep)
    _, _, c_ww = hb.rollout(hb.WhalleyWilmott(), task, ep)
    assert c_ww.mean() < c_bs.mean()


def test_costs_and_payoff_accounting():
    task = hb.get("gbm-cost")
    S = np.array([[100.0, 110.0, 120.0] + [120.0] * 28])
    ep = hb.Episodes(S, np.full_like(S, 0.04), np.array([100.0]), np.array([0.0]))

    class Hold(hb.Policy):
        def act(self, obs):
            return np.ones(1)
    _, pnl, cost = hb.rollout(Hold(), task, ep)
    assert np.isclose(cost[0], 0.002 * (100 + 120))    # buy 1 at 100, sell 1 at 120
    assert np.isclose(pnl[0], 20 - cost[0] - 20)        # gain 20, payoff 20


def test_cvar_and_bootstrap():
    x = np.arange(100, dtype=float) - 50
    assert cvar(x, 0.95) == pytest.approx(48.0)
    r = paired_bootstrap(x - 1.0, x, block=1, n_boot=200)
    assert r["cvar_diff"] == pytest.approx(1.0)
    assert r["cvar_diff_ci"][0] == pytest.approx(1.0) and r["p_better"] == 0.0


def test_test_set_is_fixed():
    a = hb.get("bates-cost")._test()
    b = hb.get("bates-cost")._test()
    assert np.array_equal(a.S, b.S)
