"""Command line:

  hedgebench list
  hedgebench run TASK --policy black-scholes           # a built-in baseline
  hedgebench run TASK --policy my_policy.py:MyPolicy   # your own Policy subclass
  hedgebench leaderboard results/ > LEADERBOARD.md
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

from . import tasks as T
from .baselines import BASELINES, BlackScholes
from .evaluate import evaluate, save


def load_policy(spec):
    if spec in BASELINES:
        return BASELINES[spec]()
    if spec == "deep-hedger":
        from .deep import DeepHedger
        return DeepHedger()
    module, _, attr = spec.rpartition(":")
    if module.endswith(".py"):                      # path/to/file.py:Class
        import importlib.util
        s = importlib.util.spec_from_file_location(Path(module).stem, module)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
    else:                                           # package.module:Class
        sys.path.insert(0, str(Path.cwd()))
        mod = importlib.import_module(module)
    obj = getattr(mod, attr)
    return obj() if isinstance(obj, type) else obj


def leaderboard(folder):
    rows = [json.loads(p.read_text()) for p in sorted(Path(folder).rglob("*.json"))]
    out = ["# HedgeBench leaderboard", "",
           "Score: **CVaR 95%** of hedging P&L (average loss in the worst 5% of test episodes; lower is better). "
           "**Δ vs BS** is the CVaR difference to Black-Scholes delta with a 95% bootstrap interval; "
           "a win counts only if the whole interval is below 0.", ""]
    for name, task in T.TASKS.items():
        rs = sorted((r for r in rows if r["task"] == name), key=lambda r: r["metrics"]["cvar"])
        if not rs:
            continue
        out += [f"## `{name}`", "", task.description, "",
                "| # | Policy | CVaR 95 | Δ vs BS (95% CI) | Mean P&L | Std | Cost paid | Submitted |",
                "|---|---|---:|---|---:|---:|---:|---|"]
        for i, r in enumerate(rs, 1):
            m, vs = r["metrics"], r.get("vs")
            if vs and r["policy"] != "black-scholes":
                lo, hi = vs["cvar_diff_ci"]
                tag = " ✅" if hi < 0 else (" ❌" if lo > 0 else "")
                delta = f"{vs['cvar_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]{tag}"
            else:
                delta = "—"
            out.append(f"| {i} | {r['policy']} | {m['cvar']:.3f} | {delta} | {m['mean']:+.3f} | "
                       f"{m['std']:.3f} | {m['cost']:.3f} | {r['date']} |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hedgebench", description="Benchmark for option hedging strategies")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list tasks")
    r = sub.add_parser("run", help="evaluate a policy on a task")
    r.add_argument("task")
    r.add_argument("--policy", required=True, help="baseline name, path/to/file.py:Class, or module:Class")
    r.add_argument("--name", help="name shown on the leaderboard")
    r.add_argument("--out", default="results", help="folder for the result JSON")
    lb = sub.add_parser("leaderboard", help="build the leaderboard markdown from result files")
    lb.add_argument("folder", nargs="?", default="results")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for name, t in T.TASKS.items():
            print(f"{name:<20} [{t.kind}] {t.description}")
    elif a.cmd == "run":
        task, policy = T.get(a.task), load_policy(a.policy)
        if a.name:
            policy.name = a.name
        res, _ = evaluate(policy, task, baseline=BlackScholes())
        m = res["metrics"]
        print(f"{policy.name} on {task.name}: CVaR95 {m['cvar']:.3f}  mean {m['mean']:+.3f}  "
              f"std {m['std']:.3f}  cost {m['cost']:.3f}")
        if "vs" in res:
            lo, hi = res["vs"]["cvar_diff_ci"]
            print(f"  vs black-scholes: ΔCVaR {res['vs']['cvar_diff']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
        path = Path(a.out) / task.name
        path.mkdir(parents=True, exist_ok=True)
        save(res, path / f"{policy.name}.json")
    elif a.cmd == "leaderboard":
        print(leaderboard(a.folder))


if __name__ == "__main__":
    main()
