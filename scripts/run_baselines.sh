#!/usr/bin/env bash
# Re-run every baseline on every task and rebuild LEADERBOARD.md.
set -eo pipefail
cd "$(dirname "$0")/.."
for task in $(python3 -m hedgebench.cli list | awk '{print $1}'); do
  for policy in no-hedge black-scholes whalley-wilmott deep-hedger; do
    python3 -m hedgebench.cli run "$task" --policy "$policy"
  done
done
python3 -m hedgebench.cli leaderboard results > LEADERBOARD.md
