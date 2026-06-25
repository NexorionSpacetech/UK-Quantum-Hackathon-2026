"""Turn public TLEs into a ready-to-ingest scheduling instance.

Running ``python -m eo_tasking.generate`` (or ``scripts/regenerate_data.py``)
reads ``dataset/tles.txt``, propagates each satellite over the planning horizon,
finds every (satellite, request, window) opportunity, derives the single-service
and slew-conflict edges, and writes the data products into ``dataset/``.

The imaging requests are *synthetic and illustrative* civil targets - urgent
disaster-response sites (flood / wildfire) clustered into a few hotspots, plus
routine environmental-monitoring sites spread globally.  The clustering is the
realistic structure from the submission's flood scenario: a sudden block of
nearby urgent tasks competing for the same satellite passes, which is exactly
what makes a greedy schedule sub-optimal.  No personal, proprietary, sensitive
or defence-related data is used.  Satellite orbits are the only real data and
come from public Two-Line Element sets (no personal data, not subject to
data-protection legislation).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

from sgp4.api import Satrec

from .instance import (Satellite, Request, Opportunity, ConflictEdge, Instance)
from .propagator import (AccessConfig, find_access_windows, slew_time_s)

# --------------------------------------------------------------------------- #
#  Imaging requests - 21 illustrative *civil* targets within the 53-degree
#  Starlink shell.  Urgent disaster sites are clustered into three hotspots
#  (Mediterranean / California / SE-Asia) so they compete for the same passes;
#  routine monitoring sites are spread globally.  priority = reward for serving.
# --------------------------------------------------------------------------- #
REQUESTS: List[Request] = [
    # --- Mediterranean flood / wildfire hotspot (urgent) ------------------ #
    Request("REQ00", "Valencia flood watch (ES)",      39.47,   -0.38, "urgent", 10),
    Request("REQ01", "Barcelona flood watch (ES)",     41.39,    2.17, "urgent",  9),
    Request("REQ02", "Marseille flood watch (FR)",     43.30,    5.37, "urgent",  9),
    Request("REQ03", "Rome flood watch (IT)",          41.90,   12.50, "urgent",  8),
    Request("REQ04", "Naples wildfire watch (IT)",     40.85,   14.27, "urgent",  8),
    Request("REQ05", "Athens wildfire watch (GR)",     37.98,   23.73, "urgent", 10),
    # --- California wildfire hotspot (urgent) ----------------------------- #
    Request("REQ06", "Los Angeles wildfire (US)",      34.05, -118.24, "urgent",  9),
    Request("REQ07", "San Francisco wildfire (US)",    37.77, -122.42, "urgent",  9),
    Request("REQ08", "Sacramento flood watch (US)",    38.58, -121.49, "urgent",  8),
    Request("REQ09", "Las Vegas flood watch (US)",     36.17, -115.14, "urgent",  7),
    # --- SE-Asia haze / flood hotspot (urgent) ---------------------------- #
    Request("REQ10", "Chiang Mai major fire (TH)",     18.79,   98.98, "urgent", 10),
    Request("REQ11", "Bangkok flood watch (TH)",       13.76,  100.50, "urgent",  8),
    Request("REQ12", "Hanoi flood watch (VN)",         21.03,  105.85, "urgent",  7),
    # --- routine environmental monitoring (global, low/medium value) ------ #
    Request("REQ13", "London air-quality (UK)",        51.51,   -0.13, "routine", 3),
    Request("REQ14", "Lagos coastal erosion (NG)",      6.52,    3.38, "routine", 4),
    Request("REQ15", "Mumbai monsoon survey (IN)",     19.08,   72.88, "routine", 4),
    Request("REQ16", "Nairobi agriculture (KE)",       -1.29,   36.82, "routine", 4),
    Request("REQ17", "Cape Town reservoirs (ZA)",     -33.92,   18.42, "routine", 4),
    Request("REQ18", "Buenos Aires delta (AR)",       -34.60,  -58.38, "routine", 3),
    Request("REQ19", "Tokyo bay monitoring (JP)",      35.68,  139.65, "routine", 4),
    Request("REQ20", "Santiago snowpack (CL)",        -33.45,  -70.67, "routine", 3),
]


@dataclass
class GenerateConfig:
    epoch_utc: str = "2026-06-24T00:00:00+00:00"
    horizon_hours: float = 8.0
    max_off_nadir_deg: float = 50.0     # agility envelope (how far it can point)
    require_daylight: bool = True       # optical EO needs a sunlit target
    step_s: float = 15.0                # sampling step for window detection
    slew_rate_deg_s: float = 1.0        # agile-EO body slew rate
    slew_settle_s: float = 20.0         # settle + image-dwell after a slew


def load_tles(path: str | Path) -> List[Satellite]:
    """Parse a 3-line-per-record TLE file into Satellite records (id SAT0..)."""
    lines = [ln.rstrip() for ln in Path(path).read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    sats: List[Satellite] = []
    for k in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[k].strip(), lines[k + 1], lines[k + 2]
        sats.append(Satellite(id=f"SAT{len(sats)}", name=name,
                              norad_id=int(l1[2:7]), tle_line1=l1, tle_line2=l2))
    return sats


def _opps_for_sat(opps: List[Opportunity], sat_id: str) -> List[Opportunity]:
    return [o for o in opps if o.sat_id == sat_id]


def build_instance(tle_path: str | Path,
                   cfg: GenerateConfig | None = None,
                   name: str = "nqcc-eo-base") -> Instance:
    cfg = cfg or GenerateConfig()
    epoch = datetime.fromisoformat(cfg.epoch_utc).astimezone(timezone.utc)
    sats = load_tles(tle_path)
    targets: List[Tuple[float, float]] = [(r.lat_deg, r.lon_deg) for r in REQUESTS]

    acc = AccessConfig(max_off_nadir_deg=cfg.max_off_nadir_deg,
                       require_daylight=cfg.require_daylight,
                       step_s=cfg.step_s)

    # 1. opportunities: one per access window of each satellite over each target
    opportunities: List[Opportunity] = []
    for si, sat in enumerate(sats):
        rec = Satrec.twoline2rv(sat.tle_line1, sat.tle_line2)
        for w in find_access_windows(rec, si, targets, epoch, cfg.horizon_hours, acc):
            req = REQUESTS[w.req_idx]
            opportunities.append(Opportunity(
                id=len(opportunities),
                sat_id=sat.id,
                request_id=req.id,
                t_start=w.t_start.isoformat(),
                t_end=w.t_end.isoformat(),
                t_best=w.t_best.isoformat(),
                duration_s=(w.t_end - w.t_start).total_seconds(),
                off_nadir_deg=round(w.min_off_nadir_deg, 2),
                reward=req.priority,
                los_ecef=tuple(round(c, 6) for c in w.los_best_ecef),
            ))

    # 2a. single-service conflicts: two opportunities serving the same request
    conflicts: List[ConflictEdge] = []
    by_req: dict = {}
    for o in opportunities:
        by_req.setdefault(o.request_id, []).append(o)
    for opps in by_req.values():
        for a, b in combinations(opps, 2):
            conflicts.append(ConflictEdge("single_service", a.id, b.id))

    # 2b. slew conflicts: same satellite, cannot retarget in the time available
    for sat in sats:
        opps = sorted(_opps_for_sat(opportunities, sat.id), key=lambda o: o.t_start)
        for a, b in combinations(opps, 2):
            ta_end = datetime.fromisoformat(a.t_end)
            tb_start = datetime.fromisoformat(b.t_start)
            gap = (tb_start - ta_end).total_seconds()
            need = slew_time_s(a.los_ecef, b.los_ecef,
                               cfg.slew_rate_deg_s, cfg.slew_settle_s)
            slack = gap - need
            if slack < 0:
                conflicts.append(ConflictEdge("slew", a.id, b.id, round(slack, 1)))

    # 3. penalties: any single violation must cost more than the best reward
    max_reward = max(r.priority for r in REQUESTS)
    penalty = float(max_reward + 1)

    return Instance(
        name=name,
        epoch_utc=epoch.isoformat(),
        horizon_hours=cfg.horizon_hours,
        satellites=sats,
        requests=list(REQUESTS),
        opportunities=opportunities,
        conflicts=conflicts,
        penalty_single_service=penalty,
        penalty_slew=penalty,
        meta={
            "source": "public Starlink TLEs (CelesTrak GP)",
            "max_off_nadir_deg": cfg.max_off_nadir_deg,
            "require_daylight": cfg.require_daylight,
            "slew_rate_deg_s": cfg.slew_rate_deg_s,
            "slew_settle_s": cfg.slew_settle_s,
            "step_s": cfg.step_s,
        },
    )


def write_dataset(inst: Instance, dataset_dir: str | Path) -> None:
    d = Path(dataset_dir)
    d.mkdir(parents=True, exist_ok=True)
    inst.save_json(d / "instance.json")
    inst.save_csv(d)
    # per-satellite sub-problems: small, brute-forceable, the natural QAOA units
    from .baseline import subproblems_by_sat
    subdir = d / "subproblems"
    subdir.mkdir(exist_ok=True)
    for sid, sub in subproblems_by_sat(inst).items():
        if sub.n_vars > 0:
            sub.save_json(subdir / f"{sid}.json")
    import json
    (d / "requests.json").write_text(
        json.dumps([r.__dict__ for r in inst.requests], indent=2), encoding="utf-8")
    (d / "opportunities.json").write_text(
        json.dumps([o.__dict__ for o in inst.opportunities], indent=2), encoding="utf-8")


def main() -> None:
    here = Path(__file__).resolve().parent.parent      # repo root
    dataset = here / "dataset"
    inst = build_instance(dataset / "tles.txt")
    write_dataset(inst, dataset)
    print(inst.summary())
    print(f"\nwrote dataset to {dataset}")


if __name__ == "__main__":
    main()


