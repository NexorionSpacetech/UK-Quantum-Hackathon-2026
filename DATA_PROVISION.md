# Data Provision Statement & Data Dictionary

## Data provision statement

> To ensure hackathon readiness, a GitHub repository will be provided containing
> pre-processed orbital data and a baseline classical solver. This removes the
> data-acquisition bottleneck and lets teams focus on quantum algorithm
> implementation from the first hour.

This repository **is** that provision. Everything a team needs to start
optimising on hour one is pre-computed in [`dataset/`](dataset/) and ingestible
with nothing but the Python standard library (CSV) or one line of JSON parsing.

## What is real, what is synthetic

| Element | Source | Real / synthetic |
|---|---|---|
| Satellite orbits | Public Two-Line Element sets (Starlink, CelesTrak GP) | **Real.** No personal data; not subject to data-protection law. |
| Ground targets (lat/lon) | Well-known civil cities / regions | Real places, **illustrative** selection. |
| Imaging-request priorities | Assigned by scenario | **Synthetic & illustrative** (urgent disaster vs routine monitoring). |
| Access windows, off-nadir angles | Computed by the SGP4 propagator in `eo_tasking/` | Derived from the real orbits. |
| Slew-conflict / single-service edges | Computed from the windows + a slew model | Derived. |

No personal data, no operator-proprietary data, no sensitive or named-defence
locations are involved, so there are **no GDPR or data-protection concerns**.
The targets are civil throughout (flood, wildfire, environmental monitoring),
consistent with the hackathon's exclusion of defence use cases.

## The base instance at a glance

* 6 satellites (real Starlink TLEs), 21 civil imaging requests
* **35 candidate opportunities** = 35 binary decision variables
* 44 conflict edges: 30 single-service + 14 slew
* Decomposes into per-satellite sub-problems of 3-9 variables (the QAOA units)
* Scenario epoch 2026-06-24T00:00:00Z, 8-hour planning horizon

## Files in `dataset/`

Every table is provided as both **CSV** (spreadsheet / pandas friendly) and,
where structured, **JSON**. `instance.json` is the whole instance in one file.

### `tles.txt`
Raw public Two-Line Element sets, 3 lines per satellite (name, line 1, line 2).
The only real input; everything else is derived from these.

### `satellites.csv`
| column | meaning |
|---|---|
| `id` | scenario id (`SAT0` ... `SAT5`) |
| `name` | TLE object name (e.g. `STARLINK-1008`) |
| `norad_id` | NORAD catalogue number |
| `tle_line1`, `tle_line2` | the TLE this satellite was propagated from |

### `requests.csv` / `requests.json`
| column | meaning |
|---|---|
| `id` | request id (`REQ00` ...) |
| `name` | human-readable civil target |
| `lat_deg`, `lon_deg` | target location (degrees) |
| `category` | `urgent` (disaster response) or `routine` (monitoring) |
| `priority` | reward earned if the request is served once (the QUBO weight) |

### `opportunities.csv` / `opportunities.json`  - the decision variables
| column | meaning |
|---|---|
| `id` | variable index `0 .. n-1` (one binary variable each) |
| `sat_id` | which satellite |
| `request_id` | which request it would serve |
| `t_start`, `t_end` | access-window bounds (ISO-8601 UTC) |
| `t_best` | time of minimum off-nadir angle (best image time) |
| `duration_s` | window length in seconds |
| `off_nadir_deg` | minimum off-nadir (pointing) angle over the window |
| `reward` | priority of the served request (copied for convenience) |
| `los_ecef` | unit line-of-sight vector at `t_best` (used for slew geometry) |

### `conflicts.csv`  - the QUBO couplings (edge list)
| column | meaning |
|---|---|
| `kind` | `single_service` (same request twice) or `slew` (cannot retarget) |
| `u`, `v` | the two opportunity ids that cannot both be scheduled |
| `slack_s` | for slew edges: time gap minus required slew time (negative = the conflict) |

### `instance.json`
The complete instance (satellites + requests + opportunities + conflicts +
penalty weights `A`, `B` + provenance metadata). Load with
`eo_tasking.instance.Instance.load_json` or any JSON reader.

### `subproblems/SAT*.json`
The instance split by satellite (cross-satellite single-service edges dropped).
Each is small enough to brute-force and is the natural unit for one QAOA run.
**`SAT2.json` is the showcase**: 9 variables, where both the greedy baseline and
local search score 35 against the brute-force optimum of 40 - a 12.5% gap, and
the value a quantum-assisted solver is challenged to recover.

## Reproducing / extending the data

The data is pre-computed, so this is optional. To regenerate from the TLEs (or
after editing the targets, horizon, off-nadir envelope, or slew model in
`eo_tasking/generate.py`):

```bash
pip install -r requirements.txt
python scripts/regenerate_data.py
```

To use fresh TLEs, replace `dataset/tles.txt` with any 3-line-per-record TLE
file (e.g. from CelesTrak `GROUP=starlink`) and re-run the command above.
