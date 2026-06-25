"""Classical baselines: the heuristic teams must beat, and the exact optimum.

  * ``greedy``        - greedy-by-value insertion, the class of heuristic that
                        operational schedulers use today.  This is the ground
                        truth a quantum-assisted solver is challenged to beat.
  * ``local_search``  - cheap hill-climbing on top of greedy (an honest, slightly
                        stronger classical control).
  * ``brute_force``   - exact optimum by enumeration, tractable on the small
                        instances and per-satellite sub-problems (<= ~22 vars).

All of them return a 0/1 numpy vector indexed by opportunity id, so they can be
scored with ``eo_tasking.qubo.evaluate``.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional

import numpy as np

from .instance import Instance
from .qubo import served_value, is_feasible


# --------------------------------------------------------------------------- #
#  Greedy-by-value heuristic  (the baseline to beat)
# --------------------------------------------------------------------------- #
def greedy(inst: Instance) -> np.ndarray:
    """Insert opportunities in decreasing reward, skipping any that conflict.

    Ties are broken by earlier ``t_start`` then lower off-nadir angle - a
    deterministic, defensible rule.  This mirrors operational greedy insertion:
    take the most valuable feasible acquisition next, never look back.
    """
    adj = inst.adjacency()
    order = sorted(
        inst.opportunities,
        key=lambda o: (-o.reward, o.t_start, o.off_nadir_deg),
    )
    x = np.zeros(inst.n_vars, dtype=int)
    chosen: set = set()
    for o in order:
        if any(c in chosen for c in adj[o.id]):
            continue
        x[o.id] = 1
        chosen.add(o.id)
    return x


def local_search(inst: Instance, x0: Optional[np.ndarray] = None,
                 max_passes: int = 50) -> np.ndarray:
    """Hill-climb from ``x0`` (default: greedy) with feasible 1-add / 1-1 swaps.

    A standard local-search control: repeatedly try to add a free opportunity,
    or swap one selected opportunity for a more valuable conflicting one, as long
    as feasibility is preserved and value strictly improves.
    """
    adj = inst.adjacency()
    x = greedy(inst) if x0 is None else x0.copy()
    rewards = {o.id: o.reward for o in inst.opportunities}

    def selected() -> set:
        return {i for i in range(inst.n_vars) if x[i]}

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        sel = selected()
        # 1-add: any free variable with no selected neighbour
        for i in range(inst.n_vars):
            if not x[i] and not (adj[i] & sel):
                x[i] = 1
                sel.add(i)
                improved = True
        # 1-1 swap: drop a neighbour-blocked low-value var for a better one
        for i in range(inst.n_vars):
            if x[i]:
                continue
            blockers = adj[i] & sel
            if len(blockers) == 1:
                b = next(iter(blockers))
                if rewards[i] > rewards[b]:
                    x[b], x[i] = 0, 1
                    sel.discard(b)
                    sel.add(i)
                    improved = True
    return x


def repair(inst: Instance, x: np.ndarray) -> np.ndarray:
    """Drop selections until ``x`` is feasible (a classical feasibility repair).

    Repeatedly removes the selected opportunity in the most conflicts (ties
    broken by lowest reward) until no conflict edge has both endpoints set.
    Turns a raw QAOA bitstring into a feasible warm start before local-search
    refinement.
    """
    x = x.copy()
    adj = inst.adjacency()
    rewards = {o.id: o.reward for o in inst.opportunities}
    while True:
        sel = {i for i in range(inst.n_vars) if x[i]}
        conflicted = [(i, len(adj[i] & sel)) for i in sel if adj[i] & sel]
        if not conflicted:
            return x
        worst = min(conflicted, key=lambda t: (-t[1], rewards[t[0]]))[0]
        x[worst] = 0

# --------------------------------------------------------------------------- #
#  Exact optimum by enumeration  (small instances / sub-problems only)
# --------------------------------------------------------------------------- #
def brute_force(inst: Instance, max_vars: int = 22) -> np.ndarray:
    """Maximum-value feasible schedule by exhaustive search over 2^n.

    Refuses instances larger than ``max_vars`` (2^22 ~ 4M already takes a few
    seconds).  Use it on the per-satellite sub-problems, or on the small bundled
    instances, to obtain the ground-truth optimum.
    """
    n = inst.n_vars
    if n > max_vars:
        raise ValueError(
            f"brute_force refuses n={n} > max_vars={max_vars}; "
            f"decompose by satellite (see subproblems_by_sat) first."
        )
    adj = inst.adjacency()
    rewards = np.array([o.reward for o in inst.opportunities])

    best_val = -1
    best_x = np.zeros(n, dtype=int)
    for bits in product((0, 1), repeat=n):
        x = np.array(bits, dtype=int)
        # feasibility: no conflict edge with both endpoints set
        ok = True
        for i in range(n):
            if x[i] and (adj[i] & {j for j in range(n) if x[j] and j != i}):
                ok = False
                break
        if not ok:
            continue
        val = int(rewards @ x)
        if val > best_val:
            best_val, best_x = val, x
    return best_x


def subproblems_by_sat(inst: Instance) -> Dict[str, Instance]:
    """Split the instance into independent per-satellite sub-problems.

    Drops cross-satellite single-service edges (the classical layer resolves
    those when composing the global schedule, exactly as the submission's
    "logic map for teams" describes).  Each sub-instance is small enough to
    brute-force and is the natural unit for one QAOA run.
    """
    from dataclasses import replace
    out: Dict[str, Instance] = {}
    for s in inst.satellites:
        opps = inst.opportunities_for_sat(s.id)
        local_ids = {o.id: k for k, o in enumerate(opps)}
        new_opps = [replace(o, id=local_ids[o.id]) for o in opps]
        new_conf = [
            replace(e, u=local_ids[e.u], v=local_ids[e.v])
            for e in inst.conflicts
            if e.u in local_ids and e.v in local_ids
        ]
        out[s.id] = Instance(
            name=f"{inst.name}::{s.id}",
            epoch_utc=inst.epoch_utc,
            horizon_hours=inst.horizon_hours,
            satellites=[s],
            requests=inst.requests,
            opportunities=new_opps,
            conflicts=new_conf,
            penalty_single_service=inst.penalty_single_service,
            penalty_slew=inst.penalty_slew,
        )
    return out


def exact_ilp(inst: Instance) -> np.ndarray:
    """Exact maximum-value feasible schedule via integer programming (PuLP/CBC).

    Optional - requires PuLP (``pip install pulp``).  Solves the full base
    instance to proven optimality in well under a second, giving the
    ground-truth gap for the whole problem, not only the per-satellite pieces.
    Each conflict edge becomes a "no more than one endpoint" constraint, so the
    optimum is the maximum-weight independent set the QUBO also encodes.
    """
    import pulp

    prob = pulp.LpProblem("eo_tasking", pulp.LpMaximize)
    x = {o.id: pulp.LpVariable(f"x{o.id}", cat="Binary") for o in inst.opportunities}
    prob += pulp.lpSum(o.reward * x[o.id] for o in inst.opportunities)
    for e in inst.conflicts:
        prob += x[e.u] + x[e.v] <= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    sol = np.zeros(inst.n_vars, dtype=int)
    for o in inst.opportunities:
        sol[o.id] = int(round(pulp.value(x[o.id]) or 0))
    return sol

def report(inst: Instance) -> str:
    """Run every baseline and format the comparison table teams aim to beat."""
    g = greedy(inst)
    ls = local_search(inst)
    rows = [
        ("Greedy-by-value (today's heuristic)", g),
        ("Greedy + local search", ls),
    ]
    exact_note = ""
    opt = None
    if inst.n_vars <= 22:
        rows.append(("Brute-force optimum", brute_force(inst)))
        opt = served_value(inst, rows[-1][1])
    else:
        try:
            rows.append(("Exact optimum (ILP)", exact_ilp(inst)))
            opt = served_value(inst, rows[-1][1])
        except Exception as exc:                       # PuLP missing, etc.
            exact_note = (f"\n(exact optimum unavailable: {exc}; "
                          f"decompose by satellite to brute-force it)")

    lines = [f"Baselines for {inst.name}  (n={inst.n_vars} variables)",
             "-" * 60,
             f"{'method':<38}{'value':>7}{'feas':>6}{'gap':>8}"]
    for label, x in rows:
        val = served_value(inst, x)
        feas = "yes" if is_feasible(inst, x) else "NO"
        gap = "" if opt is None else f"{100 * (opt - val) / opt:5.1f}%"
        lines.append(f"{label:<38}{val:>7}{feas:>6}{gap:>8}")
    return "\n".join(lines) + exact_note


