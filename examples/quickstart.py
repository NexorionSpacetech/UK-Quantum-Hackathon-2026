#!/usr/bin/env python
"""60-second quickstart: load the instance, build the QUBO, score a solution.

    python examples/quickstart.py

This is the "hour one" path from the data provision statement - no orbital
mechanics, no quantum hardware, just the pre-computed instance.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from eo_tasking.instance import Instance
from eo_tasking.qubo import build_qubo, to_ising, evaluate
from eo_tasking.baseline import greedy

root = Path(__file__).resolve().parent.parent
inst = Instance.load_json(root / "dataset" / "instance.json")
print(inst.summary(), "\n")

# 1. the QUBO matrix  H(x) = x^T Q x
Q = build_qubo(inst)
print(f"QUBO matrix: {Q.shape[0]} x {Q.shape[1]}")

# 2. the Ising form QAOA runs on
h, J, offset = to_ising(Q)
print(f"Ising: {np.count_nonzero(h)} fields, {np.count_nonzero(J)} couplings\n")

# 3. score the greedy baseline
x = greedy(inst)
print("greedy solution:", evaluate(inst, x))
print("selected opportunities:",
      [o.id for o in inst.opportunities if x[o.id]])
