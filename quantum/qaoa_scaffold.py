"""Optional QAOA scaffold (Qiskit) for one scheduling sub-problem.

This is the "optional small QAOA scaffold" mentioned in the use-case submission.
It is intentionally minimal - a *starting point* a team can read in five minutes
and then extend - not a finished solver.  It is the only file in the repo that
needs Qiskit; everything else (data, propagator, baselines) runs without it.

What it does
------------
  1. take a scheduling Instance (ideally a per-satellite sub-problem of 8-15
     variables, from baseline.subproblems_by_sat),
  2. build its QUBO and map it to an Ising Hamiltonian (SparsePauliOp),
  3. run QAOA at depth p with COBYLA on Qiskit's StatevectorEstimator,
  4. sample the optimised circuit and decode the most likely bitstring,
  5. repair it to feasibility and return it as a warm start for the classical
     layer.

Running on real hardware: swap StatevectorEstimator / StatevectorSampler for
qiskit_ibm_runtime's EstimatorV2 / SamplerV2 - the primitive (PUB) API is
identical, so no other change is needed.

The honest framing from the submission applies: at this size a classical solver
is trivial, so QAOA's role here is to produce a high-quality *warm start*, and a
null result (quantum warm start does not beat the classical control) is a valid,
reportable outcome.

Requires: qiskit >= 1.0 (core only - no qiskit-aer needed), numpy, scipy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from eo_tasking.instance import Instance
from eo_tasking.qubo import build_qubo, to_ising, evaluate
from eo_tasking.baseline import local_search, repair


def ising_to_sparsepauliop(h: np.ndarray, J: np.ndarray):
    """Build a Qiskit SparsePauliOp from Ising fields h and couplings J."""
    from qiskit.quantum_info import SparsePauliOp

    n = len(h)
    terms: List[Tuple[str, float]] = []
    for i in range(n):
        if abs(h[i]) > 1e-12:
            z = ["I"] * n
            z[n - 1 - i] = "Z"            # qiskit's little-endian qubit order
            terms.append(("".join(z), float(h[i])))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                z = ["I"] * n
                z[n - 1 - i] = "Z"
                z[n - 1 - j] = "Z"
                terms.append(("".join(z), float(J[i, j])))
    if not terms:
        terms.append(("I" * n, 0.0))
    return SparsePauliOp.from_list(terms)


def solve_qaoa(inst: Instance, p: int = 2, maxiter: int = 150,
               shots: int = 4096, seed: int = 7) -> dict:
    """Run QAOA on one (small) instance and return a warm start + diagnostics."""
    from qiskit import transpile
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.primitives import StatevectorEstimator, StatevectorSampler
    from scipy.optimize import minimize

    n = inst.n_vars
    if n == 0:
        return {"x": np.zeros(0, dtype=int), "value": 0, "feasible": True, "p": p}
    if n > 18:
        raise ValueError(
            f"this scaffold runs statevector QAOA; n={n} > 18 is too large. "
            f"Use baseline.subproblems_by_sat to get 8-15 variable pieces."
        )

    Q = build_qubo(inst)
    h, J, offset = to_ising(Q)
    cost_op = ising_to_sparsepauliop(h, J)

    # Build QAOA and DECOMPOSE it to elementary gates.  A QAOAAnsatz carries a
    # PauliEvolutionGate that the statevector estimator would otherwise re-derive
    # by matrix exponential on every call (~1000x slower).  Transpiling once to
    # rz / rzz / rx rotations is exact here, because the cost terms all commute.
    ansatz = transpile(QAOAAnsatz(cost_operator=cost_op, reps=p),
                       basis_gates=["h", "rx", "rz", "rzz", "cx"],
                       optimization_level=0)
    ansatz_measured = ansatz.measure_all(inplace=False)
    estimator = StatevectorEstimator(seed=seed)
    sampler = StatevectorSampler(seed=seed)

    def energy(theta: np.ndarray) -> float:
        job = estimator.run([(ansatz, cost_op, [theta])])
        return float(job.result()[0].data.evs[0])

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, np.pi, ansatz.num_parameters)
    res = minimize(energy, x0, method="COBYLA", options={"maxiter": maxiter})

    # sample the optimised circuit; pick the lowest-energy feasible-after-repair
    job = sampler.run([(ansatz_measured, res.x)], shots=shots)
    counts = job.result()[0].data.meas.get_counts()

    best = None
    for bitstr, freq in sorted(counts.items(), key=lambda kv: -kv[1]):
        # qiskit returns big-endian strings; reverse to variable order 0..n-1
        x = np.array([int(b) for b in bitstr[::-1]], dtype=int)
        x = local_search(inst, repair(inst, x))   # repair to feasible, then climb
        ev = evaluate(inst, x)
        key = (ev["feasible"], ev["value"])        # prefer feasible, then value
        if best is None or key > (best[1]["feasible"], best[1]["value"]):
            best = (x, ev, freq)

    x_warm, ev, _ = best
    return {
        "x": x_warm,
        "value": ev["value"],
        "feasible": ev["feasible"],
        "energy": ev["energy"],
        "p": p,
        "opt_energy": float(res.fun) + offset,
        "n_qubits": n,
    }


def _demo() -> None:
    """Run QAOA on the largest per-satellite sub-problem and compare baselines."""
    from pathlib import Path
    from eo_tasking.baseline import subproblems_by_sat, greedy, brute_force
    from eo_tasking.qubo import served_value

    root = Path(__file__).resolve().parent.parent
    inst = Instance.load_json(root / "dataset" / "instance.json")
    subs = subproblems_by_sat(inst)
    sat_id = max(subs, key=lambda k: subs[k].n_vars)
    sub = subs[sat_id]
    print(f"QAOA scaffold demo on sub-problem {sub.name}  (n={sub.n_vars} qubits)\n")

    g = served_value(sub, greedy(sub))
    opt = served_value(sub, brute_force(sub)) if sub.n_vars <= 18 else None
    res = solve_qaoa(sub, p=2)

    print(f"  greedy baseline ........ value {g}")
    if opt is not None:
        print(f"  brute-force optimum .... value {opt}")
    print(f"  QAOA hybrid warm start . value {res['value']}  feasible={res['feasible']}")


if __name__ == "__main__":
    _demo()



