"""Deep hedging baseline (Buehler et al. 2019), in PyTorch: `pip install hedgebench[deep]`.

A feed-forward network maps (time left, log-moneyness, current vol, current position)
to the next position, trained to minimise CVaR of the hedging P&L on simulated
episodes from task.sample_train().
"""
import numpy as np
import torch
import torch.nn as nn

from .evaluate import Policy, Obs


def _features(tau_frac, S, K, v, pos):
    return torch.stack([tau_frac, torch.log(S / K), torch.sqrt(v.clamp_min(1e-8)), pos], dim=-1)


class DeepHedger(Policy):
    name = "deep-hedger"

    def __init__(self, hidden=64, n_train=200_000, epochs=40, batch=4096, lr=1e-3, alpha=0.95, seed=0):
        self.cfg = dict(n_train=n_train, epochs=epochs, batch=batch, lr=lr, alpha=alpha, seed=seed)
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1), nn.Sigmoid())

    def _pnl(self, S, v, K, prem, N, cost):
        pos = torch.zeros(S.shape[0]); P = []
        for t in range(N):
            tf = torch.full_like(pos, (N - t) / N)
            pos = self.net(_features(tf, S[:, t], K, v[:, t], pos)).squeeze(-1)
            P.append(pos)
        P = torch.stack(P, 1)
        z = torch.zeros_like(P[:, :1])
        costs = cost * ((torch.cat([z, P, z], 1).diff(dim=1)).abs() * S).sum(1)
        return prem + (P * S.diff(dim=1)).sum(1) - costs - torch.relu(S[:, -1] - K)

    def fit(self, task):
        c = self.cfg
        ep = task.sample_train(c["n_train"], seed=c["seed"])
        S, v, K, prem = (torch.tensor(x, dtype=torch.float32) for x in (ep.S, ep.v, ep.K, ep.premium))
        w = nn.Parameter(torch.tensor(0.0))      # CVaR threshold (Rockafellar-Uryasev)
        with torch.no_grad():
            w.fill_(torch.quantile(-self._pnl(S[:c["batch"]], v[:c["batch"]], K[:c["batch"]],
                                              prem[:c["batch"]], task.n_steps, task.cost), c["alpha"]).item())
        opt = torch.optim.Adam([{"params": self.net.parameters(), "lr": c["lr"]}, {"params": [w], "lr": 0.05}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c["epochs"])
        n = len(S)
        for _ in range(c["epochs"]):
            for idx in torch.randperm(n).split(c["batch"]):
                pnl = self._pnl(S[idx], v[idx], K[idx], prem[idx], task.n_steps, task.cost)
                loss = w + torch.relu(-pnl - w).mean() / (1 - c["alpha"])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

    @torch.no_grad()
    def act(self, obs: Obs):
        f = lambda x: torch.tensor(x, dtype=torch.float32)
        tf = torch.full((self.info.n,), obs.tau / self.info.T)
        x = _features(tf, f(obs.S[:, -1]), f(self.info.K), f(obs.v[:, -1]), f(obs.position))
        return self.net(x).squeeze(-1).numpy().astype(float)
