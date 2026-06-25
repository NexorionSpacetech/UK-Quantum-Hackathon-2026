#!/usr/bin/env python
"""Run the classical baselines and print the table teams aim to beat.

    python scripts/run_baseline.py

Shows greedy, greedy+local-search, and (where tractable) the brute-force optimum
for the full base instance and for each per-satellite sub-problem.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eo_tasking.instance import Instance
from eo_tasking.baseline import report, subproblems_by_sat

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    inst = Instance.load_json(root / "dataset" / "instance.json")

    print("=" * 60)
    print(inst.summary())
    print("=" * 60)
    print(report(inst))

    print("\nPer-satellite sub-problems (each a natural QAOA unit):")
    print("=" * 60)
    for sat_id, sub in subproblems_by_sat(inst).items():
        if sub.n_vars == 0:
            continue
        print(report(sub))
        print("-" * 60)
