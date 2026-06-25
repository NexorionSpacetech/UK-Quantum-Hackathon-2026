"""Build the QUBO / Ising cost function for a scheduling instance.

This is the exact cost function from the use-case submission (section 4):

    minimise  H(x) = - sum_w  p_w x_w
                     + A sum_(single-service pairs)  x_w x_w'
                     + B sum_(slew-conflict pairs)   x_w x_w'

with x_w in {0, 1} one binary variable per opportunity, p_w the reward of the
request that opportunity serves, A the single-service penalty and B the slew
penalty.  Pick A and B larger than any reward they could unlock and the global
minimum of H is a feasible, maximum-value schedule.

This module is deliberately free of any quantum dependency: it produces the QUBO
matrix and the equivalent Ising (h, J, offset) as plain numpy / dicts, which the
classical baselines and the optional Qiskit scaffold both consume.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .instance import Instance


def build_qubo(inst: Instance) -> np.ndarray:
    """Symmetric QUBO matrix Q (n x n) with  H(x) = x^T Q x  for x in {0,1}^n.

    Linear reward terms sit on the diagonal (Q_ii = -p_i, since x_i^2 = x_i);
    each conflict edge contributes its penalty split symmetrically off-diagonal.
    """
    n = inst.n_vars
    Q = np.zeros((n, n))
    for o in inst.opportunities:
        Q[o.id, o.id] -= o.reward
    for e in inst.conflicts:
        pen = inst.penalty_single_service if e.kind == "single_service" else inst.penalty_slew
        Q[e.u, e.v] += pen / 2.0
        Q[e.v, e.u] += pen / 2.0
    return Q


def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    """H(x) = x^T Q x."""
    return float(x @ Q @ x)


def to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert a QUBO matrix to Ising (h, J, offset) via x = (1 - z) / 2.

    Returns linear field ``h`` (length n), coupling matrix ``J`` (upper
    triangular, n x n) and the scalar energy ``offset`` so that, for spins
    z in {+1, -1}:  H = sum_i h_i z_i + sum_(i<j) J_ij z_i z_j + offset.
    """
    n = Q.shape[0]
    # split into linear (diagonal) and quadratic (symmetric off-diag) parts
    a = np.diag(Q).copy()
    Qoff = Q - np.diag(a)            # symmetric, zero diagonal
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0
    # linear:  a_i x_i = a_i (1 - z_i)/2
    for i in range(n):
        h[i] += -a[i] / 2.0
        offset += a[i] / 2.0
    # quadratic:  Q_ij x_i x_j summed over ordered pairs; fold to i<j
    for i in range(n):
        for j in range(i + 1, n):
            q = Qoff[i, j] + Qoff[j, i]      # total weight on the {i,j} pair
            if q == 0.0:
                continue
            # q * (1 - z_i)/2 * (1 - z_j)/2
            J[i, j] += q / 4.0
            h[i] += -q / 4.0
            h[j] += -q / 4.0
            offset += q / 4.0
    return h, J, offset


# --------------------------------------------------------------------------- #
#  Solution evaluation
# --------------------------------------------------------------------------- #
def served_value(inst: Instance, x: np.ndarray) -> int:
    """Total reward of the selected opportunities (ignores feasibility)."""
    return int(sum(o.reward for o in inst.opportunities if x[o.id]))


def violations(inst: Instance, x: np.ndarray) -> List[Tuple[str, int, int]]:
    """List the conflict edges that ``x`` violates (both endpoints selected)."""
    return [(e.kind, e.u, e.v) for e in inst.conflicts if x[e.u] and x[e.v]]


def is_feasible(inst: Instance, x: np.ndarray) -> bool:
    return len(violations(inst, x)) == 0


def evaluate(inst: Instance, x: np.ndarray) -> Dict:
    """Convenience: value, feasibility, energy and violation count for x."""
    Q = build_qubo(inst)
    return {
        "value": served_value(inst, x),
        "feasible": is_feasible(inst, x),
        "n_violations": len(violations(inst, x)),
        "energy": qubo_energy(Q, x),
        "n_selected": int(x.sum()),
    }
