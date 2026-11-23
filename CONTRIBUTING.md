# Contributing to HedgeBench

## Submit a result to the leaderboard

1. Write your policy as a subclass of `hedgebench.Policy` (see [`examples/my_policy.py`](examples/my_policy.py)).
   - `fit(task)` (optional): train with `task.sample_train(n, seed)` or `task.history()`.
     Never use the test set for training or tuning.
   - `act(obs)`: return the number of shares to hold, given only the past.
2. Run it on one or more tasks:
   ```bash
   hedgebench run bates-cost --policy my_policy.py:MyPolicy --name my-policy-v1
   ```
   This writes `results/bates-cost/my-policy-v1.json`.
3. Open a pull request that adds:
   - the result JSON files,
   - your policy's code (in `submissions/<name>/`), or a link to a public repo with it,
   - a short description: method, training data used, seed, and a paper link if there is one.

A maintainer re-runs your code before merging. Results that can't be reproduced, within
the seed noise, aren't merged.

## Rules
- **No test-set tuning.** Choose hyperparameters with `sample_train()` or `history()` only.
- **Real-data tasks:** train only on the task's training period. `task.history()` already
  stops at the split date.
- **No data in the repo.** Market data is downloaded by each user from the original source
  and cached locally. Don't commit downloaded data: the data providers' terms restrict
  redistribution.
- **Report what you tried.** If you ran many variants, say how many. Benchmarks reward
  honesty.

## Propose a task
Open an issue with: the market (simulator or real data source, which must be free), the
instrument, cost, train/test split, and why it tests something the current tasks don't.

## Development
```bash
pip install -e ".[dev,deep,real]"
pytest
```
