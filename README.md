# HedgeBench

**A reproducible benchmark for option hedging strategies, from Black-Scholes to deep hedging,
on simulated and real market data.**

[![tests](https://github.com/taneeshk12/hedgebench/actions/workflows/ci.yml/badge.svg)](https://github.com/taneeshk12/hedgebench/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Deep hedging papers each test on their own setup: different markets, costs, option
contracts and scores. Their results can't be compared. HedgeBench fixes the test, so every
method is scored on the same episodes, with the same costs, and the same statistics:

- **6 tasks**: four simulated markets (exactly reproducible, no internet needed) and two on
  real data: **SPY 2016–2025** and **real traded Bitcoin options** on Deribit.
- **No look-ahead, by construction.** The evaluator feeds a policy one date at a time and
  only ever shows it the past.
- **Wins must be significant.** Every result is compared with Black-Scholes using a paired
  (block) bootstrap. A win counts only if the whole 95% interval is below zero.
- **Strong baselines included:** no hedge, Black-Scholes delta, Whalley-Wilmott (the
  classic no-trade band) and a deep hedger (Buehler et al., 2019).

## Leaderboard

Score: **CVaR 95%**, the average loss in the worst 5% of test episodes, per option sold
(the stock starts at 100). Lower is better. ✅ = significantly better than Black-Scholes, ❌ = significantly worse.

| Task | No hedge | Black-Scholes | Whalley-Wilmott | Deep hedger |
|---|---:|---:|---:|---:|
| `gbm-frictionless` | 15.04 ❌ | **3.76** | **3.76** | 3.86 ❌ |
| `gbm-cost` | 15.04 ❌ | 4.49 | 4.53 ❌ | **4.36** ✅ |
| `heston-cost` | 12.40 ❌ | 4.97 | 5.20 ❌ | **4.77** ✅ |
| `bates-cost` | 13.09 ❌ | 6.09 | 6.15 ❌ | **5.52** ✅ |
| `spy-real` | 11.64 ❌ | 5.76 | **5.59** | 5.70 |
| `btc-options-real` | 25.56 ❌ | **2.12** | 2.25 | 3.78 |

Full tables, with intervals, mean P&L and trading costs: **[LEADERBOARD.md](LEADERBOARD.md)**

### What the baselines show
- **In simulation, learning works.** The deep hedger significantly beats Black-Scholes in all
  three markets with costs, by up to 9% (`bates-cost`). It does this by paying 20–30% less in trading costs.
- **On real data, nothing significantly beats Black-Scholes yet.** On SPY, both cost-aware
  methods score better, but their intervals include zero. On real Bitcoin options, the
  textbook hedge is best.
- **That gap between simulation and real markets is the open problem.** This is what
  HedgeBench is for.

## Quick start

```bash
git clone https://github.com/taneeshk12/hedgebench && cd hedgebench
pip install -e ".[deep,real]"           # torch for the deep baseline, yfinance for SPY

hedgebench list
hedgebench run bates-cost --policy black-scholes
hedgebench run bates-cost --policy examples/my_policy.py:LazyDelta
```

### Write a policy

```python
import numpy as np
from scipy.stats import norm
import hedgebench as hb

class LazyDelta(hb.Policy):
    name = "lazy-delta"

    def fit(self, task):                 # optional: train on task.sample_train(n) or task.history()
        pass

    def act(self, obs: hb.Obs):          # obs.S, obs.v: price and variance history up to today only
        S, K, sig = obs.S[:, -1], self.info.K, np.sqrt(obs.v[:, -1])
        delta = norm.cdf((np.log(S / K) + 0.5 * sig**2 * obs.tau) / (sig * np.sqrt(obs.tau)))
        stale = np.abs(delta - obs.position) > 0.05
        return np.where((obs.t == 0) | stale, delta, obs.position)   # shares to hold

task = hb.get("bates-cost")
result, pnl = hb.evaluate(LazyDelta(), task, baseline=hb.BlackScholes())
print(result["metrics"]["cvar"], result["vs"]["cvar_diff_ci"])
```

## Tasks

Every task: you have **sold one call option** and hedge it by trading the underlying once a
day for 30 days, paying a proportional cost on every trade.

| Task | Market | Cost | Test set | What it tests |
|---|---|---|---|---|
| `gbm-frictionless` | Black-Scholes, 20% vol | 0% | 20,000 paths | Sanity check: BS delta is near-optimal |
| `gbm-cost` | Black-Scholes, 20% vol | 0.2% | 20,000 paths | Pure cost/risk trade-off |
| `heston-cost` | Heston, calibrated to SPY/VIX | 0.2% | 20,000 paths | Random volatility |
| `bates-cost` | Heston + jumps (Bates) | 0.2% | 20,000 paths | Crashes |
| `spy-real` | Real SPY + VIX, 2016–2025 | 0.2% | 497 windows | Simulation-to-real transfer |
| `btc-options-real` | Real Deribit BTC calls, 2024–2025 | 0.1% | 23 options, sold at their market price | Real traded options |

- **Training data:** `task.sample_train(n, seed)` gives unlimited simulated episodes. For the
  real tasks, its simulator is calibrated on the **training period only** (SPY 2005–2015;
  BTC 2021–2023), and `task.history()` returns that period's raw data.
- **Data licensing:** HedgeBench ships **no market data**. Real data is downloaded by each
  user from the free original source (Yahoo Finance, Deribit's public API) and cached in
  `~/.cache/hedgebench`.

## Scoring

| Metric | Meaning |
|---|---|
| **CVaR 95** (main score) | Average loss in the worst 5% of episodes. The quantity a risk desk cares about. |
| Δ vs BS + 95% CI | Paired bootstrap against Black-Scholes on the same episodes; block bootstrap when real windows overlap |
| Mean P&L, Std, VaR | The rest of the distribution |
| Cost paid | Average trading cost: shows *how* a policy wins |

## Submit a result

Run your policy, then open a pull request with the result JSON and your code. See
**[CONTRIBUTING.md](CONTRIBUTING.md)**. Rules in short: no tuning on the test set, train real-data
tasks only on their training period, and say how many variants you tried.

## Roadmap
- More real tasks: ETH options, SPX options, longer maturities
- Hedging with options as well as the underlying (vega hedging)
- Market-impact costs, not just proportional ones
- Leaderboard website on GitHub Pages

## References
- Buehler, Gonon, Teichmann & Wood (2019). *Deep Hedging*. Quantitative Finance.
- Whalley & Wilmott (1997). *An asymptotic analysis of an optimal hedging model for option pricing with transaction costs.*
- Heston (1993); Bates (1996); Rockafellar & Uryasev (2000), CVaR optimisation.

## Citation
```bibtex
@software{hedgebench2026,
  title  = {HedgeBench: a reproducible benchmark for option hedging strategies},
  author = {taneeshk12},
  year   = {2026},
  url    = {https://github.com/taneeshk12/hedgebench}
}
```

MIT licensed.
