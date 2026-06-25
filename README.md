# EO Satellite Tasking & Scheduling - UK Quantum Hackathon 2026 starter kit

A self-contained starter repository for the NQCC use case **"Hybrid
quantum-classical optimisation for Earth observation satellite tasking and
scheduling"** (Nexorion Spacetech). It ships everything a team needs to begin
from hour one:

* **pre-processed orbital data** derived from real public Starlink TLEs, exported
  as ready-to-ingest CSV and JSON;
* the **pre-computed candidate opportunities and conflict pairs** (the decision
  variables and the QUBO couplings);
* a **baseline classical heuristic solver** - the ground truth to beat - and the
  **brute-force optimum** on the small instances;
* the **open-source orbit propagator** that generated the constraints, so teams
  can regenerate or extend the data;
* an **optional QAOA scaffold** in Qiskit.

No orbital mechanics, data hunting, or quantum hardware is needed to start: load
`dataset/instance.json`, build the QUBO, and run the baseline.

## The problem in one paragraph

An agile Earth-observation constellation must decide which satellite images which
ground target in which short time window, to **maximise the total priority of the
requests served**, subject to two binding constraints: a request should be served
**at most once** (single-service), and a satellite **cannot retarget faster than
its slew rate allows** (slew conflict). One binary variable per candidate
opportunity turns this into a Quadratic Unconstrained Binary Optimisation (QUBO)
problem - the natural input for QAOA and quantum annealers. See section 4 of the
use-case submission for the full formulation; the equation is reproduced below.

## Repository layout

```
dataset/                 pre-processed, ready-to-ingest data (no code needed)
  tles.txt               real public Starlink TLEs (the only real input)
  satellites.csv         the 6 satellites + their TLEs
  requests.csv  .json    the 21 civil imaging requests + priorities
  opportunities.csv .json the 35 decision variables (sat, request, window)
  conflicts.csv          the 44 QUBO couplings (single-service + slew)
  instance.json          the whole instance in one file
  subproblems/SAT*.json  per-satellite sub-problems (the QAOA units)
eo_tasking/              open-source Python package
  propagator.py          SGP4 + access / slew geometry
  instance.py            data model + JSON / CSV load & save
  qubo.py                QUBO and Ising construction, solution scoring
  baseline.py            greedy heuristic, local search, brute force, exact ILP
  generate.py            TLEs -> instance (regenerates dataset/)
quantum/
  qaoa_scaffold.py       optional minimal Qiskit QAOA starter
scripts/
  regenerate_data.py     rebuild dataset/ from the TLEs
  run_baseline.py        print the baseline table teams aim to beat
examples/
  quickstart.py          60-second load -> QUBO -> score path
DATA_PROVISION.md        data provision statement + full data dictionary
```

## Quickstart

```bash
pip install -r requirements.txt          # numpy, sgp4, pulp

python examples/quickstart.py            # load instance, build QUBO, score greedy
python scripts/run_baseline.py           # the full baseline table + sub-problems
```

Optional - the QAOA scaffold (core Qiskit only, no qiskit-aer needed):

```bash
pip install -r requirements-quantum.txt  # qiskit, scipy
python quantum/qaoa_scaffold.py          # QAOA on the SAT2 sub-problem
```

## The instance

* 6 satellites (real Starlink TLEs), 21 civil imaging requests
* **35 candidate opportunities** = 35 binary variables
* 44 conflict edges: 30 single-service + 14 slew
* per-satellite sub-problems of 3-9 variables (each a natural QAOA unit)
* epoch 2026-06-24T00:00:00Z, 8-hour horizon, 50 deg off-nadir agility envelope

Urgent disaster targets (flood / wildfire) are clustered into three hotspots
(Mediterranean, California, SE Asia) so they compete for the same satellite
passes - the realistic structure that makes a greedy schedule sub-optimal -
while routine monitoring targets are spread globally.

## The QUBO (the cost function to minimise)

```
H(x) = - sum_w  p_w x_w                          (reward served priorities)
       + A sum_(single-service pairs)  x_w x_w'   (serve each request <= once)
       + B sum_(slew-conflict pairs)   x_w x_w'   (respect satellite agility)
```

with `x_w` in {0, 1} one variable per opportunity, `p_w` its reward, and
penalties `A = B = 11` (one more than the maximum reward, so any single
constraint violation costs more than the value it could unlock). `eo_tasking.qubo`
builds both `Q` (for `H(x) = x^T Q x`) and the equivalent Ising `(h, J, offset)`
that QAOA runs on. **Penalty-weight tuning is part of the exercise** - try
lowering `A`/`B` and watch feasibility break.

## The baseline to beat

`python scripts/run_baseline.py`:

| instance | greedy | + local search | exact optimum | gap |
|---|---|---|---|---|
| full base (n=35) | 90 | 90 | 90 (ILP) | 0.0% |
| SAT0 (n=4) | 24 | 24 | 24 | 0.0% |
| SAT1 (n=8) | 31 | 31 | 31 | 0.0% |
| **SAT2 (n=9)** | **35** | **35** | **40** | **12.5%** |
| SAT3 (n=3) | 18 | 18 | 18 | 0.0% |
| SAT4 (n=6) | 24 | 24 | 24 | 0.0% |
| SAT5 (n=5) | 28 | 28 | 28 | 0.0% |

Read this honestly. On the **composed 35-variable instance the greedy heuristic
is already optimal** (it matches the ILP), because the rich multi-satellite
coverage lets greedy route around most conflicts. The recoverable value lives
**inside individual satellite schedules**: on the 9-variable **SAT2** sub-problem
both greedy *and* local search score 35 against the optimum of 40 - a **12.5%
gap**. That sub-problem is the natural QAOA unit (`dataset/subproblems/SAT2.json`)
and the target a quantum-assisted solver is challenged to recover.

Why greedy is trapped on SAT2: imaging the high-priority Chiang Mai opportunity
(reward 10) uses up a slew window that blocks both Bangkok (8) and Hanoi (7),
whose 8 + 7 = 15 beats 10. Greedy takes the single high-value target; the optimum
takes the compatible pair. Local search stays stuck because escaping needs a
*drop-one-add-two* move, not a 1-for-1 swap - exactly the kind of trap a good
warm start can break.

## Decomposition: the logic map for teams

The submission's three-step recipe is implemented in code:

1. **filter** - `eo_tasking.generate` keeps only feasible (sat, request, window)
   triples (off-nadir within the agility envelope, target sunlit);
2. **map constraints** - single-service pairs and slew-overlap pairs become the
   conflict edges;
3. **build the Hamiltonian** - `eo_tasking.qubo.build_qubo` + `to_ising`.

`baseline.subproblems_by_sat` splits the instance by satellite (the natural
8-to-15-qubit pieces); the classical layer then composes the per-satellite
solutions and resolves the single-service couplings between them.

## The optional QAOA scaffold

`quantum/qaoa_scaffold.py` takes one (small) sub-problem, maps its QUBO to a
`SparsePauliOp`, runs QAOA at depth `p` with COBYLA on Qiskit's statevector
primitives, samples the optimised circuit, decodes the best bitstring, and
repairs it to a feasible warm start for the classical layer. It is a *starting
point* to extend, not a finished solver. To run on real IBM hardware, swap
`StatevectorEstimator` / `StatevectorSampler` for `qiskit_ibm_runtime`'s
`EstimatorV2` / `SamplerV2` - the primitive (PUB) API is identical.

## Regenerating or extending the data

The data is pre-computed; this is only if you want to change the scenario.

```bash
python scripts/regenerate_data.py
```

Edit the target list, horizon, off-nadir envelope, or slew model in
`eo_tasking/generate.py`, or drop in fresh TLEs at `dataset/tles.txt`.

## Honest framing (from the submission)

* QAOA here runs on **statevector primitives** standing in for NISQ hardware;
  its demonstrated value is **warm-start quality**, not speed.
* At this size a classical solver is trivial - the point is a clean, verifiable
  comparison against the brute-force optimum and a classical control.
* A **null or negative result** (the quantum warm start does not beat the
  classical control) is a valid, reportable outcome. The classical local-search
  control is built in precisely so quantum advantage cannot be overstated.

## Data provision statement

> To ensure hackathon readiness, a GitHub repository will be provided containing
> pre-processed orbital data and a baseline classical solver. This removes the
> data-acquisition bottleneck and lets teams focus on quantum algorithm
> implementation from the first hour.

See [DATA_PROVISION.md](DATA_PROVISION.md) for the full data dictionary and
provenance. Satellite orbits are real public TLE data (no personal data); all
priorities and targets are synthetic, civil, and illustrative - no GDPR,
proprietary, sensitive, or defence content.

## Licence

MIT (see [LICENSE](LICENSE)). (c) 2026 Nexorion Spacetech Limited. Orbit data
courtesy of CelesTrak; propagation by the `sgp4` library; quantum tooling by
Qiskit.
