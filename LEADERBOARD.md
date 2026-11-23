# HedgeBench leaderboard

Score: **CVaR 95%** of hedging P&L (average loss in the worst 5% of test episodes; lower is better). **Δ vs BS** is the CVaR difference to Black-Scholes delta with a 95% bootstrap interval; a win counts only if the whole interval is below 0.

## `gbm-frictionless`

Black-Scholes world, 20% vol, no costs. Sanity check: Black-Scholes delta is near-optimal here.

| # | Policy | CVaR 95 | Δ vs BS (95% CI) | Mean P&L | Std | Cost paid | Submitted |
|---|---|---:|---|---:|---:|---:|---|
| 1 | black-scholes | 3.763 | — | -2.756 | 0.432 | 0.000 | 2026-11-22 |
| 2 | whalley-wilmott | 3.763 | +0.000 [+0.000, +0.000] | -2.756 | 0.432 | 0.000 | 2026-11-22 |
| 3 | deep-hedger | 3.864 | +0.101 [+0.078, +0.123] ❌ | -2.755 | 0.505 | 0.000 | 2026-11-22 |
| 4 | no-hedge | 15.044 | +11.281 [+11.004, +11.537] ❌ | -2.721 | 4.187 | 0.000 | 2026-11-22 |

## `gbm-cost`

Black-Scholes world, 20% vol, 0.2% costs.

| # | Policy | CVaR 95 | Δ vs BS (95% CI) | Mean P&L | Std | Cost paid | Submitted |
|---|---|---:|---|---:|---:|---:|---|
| 1 | deep-hedger | 4.363 | -0.128 [-0.158, -0.098] ✅ | -3.200 | 0.545 | 0.445 | 2026-11-22 |
| 2 | black-scholes | 4.492 | — | -3.303 | 0.479 | 0.547 | 2026-11-22 |
| 3 | whalley-wilmott | 4.528 | +0.037 [+0.002, +0.072] ❌ | -3.093 | 0.697 | 0.336 | 2026-11-22 |
| 4 | no-hedge | 15.044 | +10.553 [+10.278, +10.808] ❌ | -2.721 | 4.187 | 0.000 | 2026-11-22 |

## `heston-cost`

Heston stochastic volatility calibrated to SPY/VIX, 0.2% costs.

| # | Policy | CVaR 95 | Δ vs BS (95% CI) | Mean P&L | Std | Cost paid | Submitted |
|---|---|---:|---|---:|---:|---:|---|
| 1 | deep-hedger | 4.774 | -0.200 [-0.228, -0.171] ✅ | -3.236 | 0.757 | 0.414 | 2026-11-22 |
| 2 | black-scholes | 4.973 | — | -3.374 | 0.679 | 0.557 | 2026-11-22 |
| 3 | whalley-wilmott | 5.202 | +0.229 [+0.193, +0.265] ❌ | -3.173 | 0.879 | 0.352 | 2026-11-22 |
| 4 | no-hedge | 12.395 | +7.422 [+7.255, +7.585] ❌ | -2.835 | 3.664 | 0.000 | 2026-11-22 |

## `bates-cost`

Heston plus jumps (Bates) calibrated to SPY/VIX, 0.2% costs.

| # | Policy | CVaR 95 | Δ vs BS (95% CI) | Mean P&L | Std | Cost paid | Submitted |
|---|---|---:|---|---:|---:|---:|---|
| 1 | deep-hedger | 5.516 | -0.574 [-0.643, -0.508] ✅ | -3.326 | 0.977 | 0.385 | 2026-11-22 |
| 2 | black-scholes | 6.090 | — | -3.503 | 0.939 | 0.559 | 2026-11-22 |
| 3 | whalley-wilmott | 6.147 | +0.057 [+0.005, +0.106] ❌ | -3.301 | 1.088 | 0.354 | 2026-11-22 |
| 4 | no-hedge | 13.086 | +6.996 [+6.757, +7.232] ❌ | -2.937 | 3.844 | 0.000 | 2026-11-22 |

## `spy-real`

Real SPY 30-trading-day windows, 2016-2025 (every 5th day), ATM call, 0.2% cost. Vol input: VIX. Train period: 2005-2015.

| # | Policy | CVaR 95 | Δ vs BS (95% CI) | Mean P&L | Std | Cost paid | Submitted |
|---|---|---:|---|---:|---:|---:|---|
| 1 | whalley-wilmott | 5.585 | -0.173 [-0.416, +0.123] | -2.427 | 1.118 | 0.355 | 2026-11-22 |
| 2 | deep-hedger | 5.698 | -0.060 [-0.194, +0.111] | -2.533 | 1.183 | 0.335 | 2026-11-23 |
| 3 | black-scholes | 5.758 | — | -2.577 | 1.062 | 0.531 | 2026-11-22 |
| 4 | no-hedge | 11.643 | +5.885 [+3.783, +8.018] ❌ | -3.059 | 3.140 | 0.000 | 2026-11-22 |

## `btc-options-real`

Real monthly BTC calls on Deribit, 2024-2025: sold at their real market price 30 days before expiry, hedged daily with BTC, 0.1% cost. Vol input: DVOL. Train period: 2021-03 to 2023-12.

| # | Policy | CVaR 95 | Δ vs BS (95% CI) | Mean P&L | Std | Cost paid | Submitted |
|---|---|---:|---|---:|---:|---:|---|
| 1 | black-scholes | 2.115 | — | +0.659 | 1.650 | 0.267 | 2026-11-22 |
| 2 | whalley-wilmott | 2.254 | +0.139 [-0.565, +0.627] | +0.613 | 1.813 | 0.197 | 2026-11-22 |
| 3 | deep-hedger | 3.775 | +1.661 [-0.516, +2.643] | +0.574 | 2.111 | 0.212 | 2026-11-23 |
| 4 | no-hedge | 25.561 | +23.446 [+9.288, +23.934] ❌ | -1.135 | 9.184 | 0.000 | 2026-11-22 |

