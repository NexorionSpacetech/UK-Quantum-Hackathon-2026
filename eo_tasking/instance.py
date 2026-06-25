"""Data model for an EO tasking & scheduling instance, with JSON / CSV I/O.

An *instance* is everything a team needs to start optimising:

  * the satellites (with the TLEs they were propagated from),
  * the imaging requests (ground targets + priority weights),
  * the candidate *opportunities* - one binary decision variable each,
  * the *conflict edges* - pairs of opportunities that cannot both be scheduled,
  * the QUBO penalty weights A (single-service) and B (slew conflict).

The instance is plain data. It is produced by ``generate.py`` and is also
shipped pre-computed in ``dataset/``, so it can be loaded with nothing but the
Python standard library.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
#  Core records
# --------------------------------------------------------------------------- #
@dataclass
class Satellite:
    """An agile EO satellite, identified by the TLE it was propagated from."""

    id: str                     # short scenario id, e.g. "SAT0"
    name: str                   # e.g. "STARLINK-1008"
    norad_id: int
    tle_line1: str
    tle_line2: str


@dataclass
class Request:
    """An imaging request for a fixed ground target with a priority weight."""

    id: str                     # e.g. "REQ03"
    name: str                   # human-readable target name
    lat_deg: float
    lon_deg: float
    category: str               # "urgent" (disaster) or "routine" (monitoring)
    priority: int               # reward earned if this request is served once


@dataclass
class Opportunity:
    """A feasible (satellite, request, time-window) triple = one binary variable.

    ``reward`` is the priority of the request it would serve.  ``off_nadir_deg``
    and the pointing unit-vector ``los_ecef`` (line of sight at ``t_best``) are
    kept so the slew geometry between opportunities can be re-derived if wanted.
    """

    id: int                     # variable index 0..n-1
    sat_id: str
    request_id: str
    t_start: str                # ISO-8601 UTC
    t_end: str                  # ISO-8601 UTC
    t_best: str                 # time of minimum off-nadir angle (best image time)
    duration_s: float
    off_nadir_deg: float        # minimum off-nadir angle over the window
    reward: int
    los_ecef: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ConflictEdge:
    """A pair of opportunities that must not both be scheduled.

    ``kind`` is "single_service" (same request served twice) or "slew" (same
    satellite cannot retarget in time).  For slew edges ``slack_s`` is the time
    gap minus the required slew time (negative = infeasible, hence the edge).
    """

    kind: str                   # "single_service" | "slew"
    u: int                      # opportunity id
    v: int                      # opportunity id
    slack_s: Optional[float] = None


# --------------------------------------------------------------------------- #
#  The instance
# --------------------------------------------------------------------------- #
@dataclass
class Instance:
    """A complete scheduling instance plus the metadata to reproduce it."""

    name: str
    epoch_utc: str                              # scenario start (ISO-8601 UTC)
    horizon_hours: float
    satellites: List[Satellite] = field(default_factory=list)
    requests: List[Request] = field(default_factory=list)
    opportunities: List[Opportunity] = field(default_factory=list)
    conflicts: List[ConflictEdge] = field(default_factory=list)
    penalty_single_service: float = 0.0         # "A" in the QUBO
    penalty_slew: float = 0.0                    # "B" in the QUBO
    meta: Dict = field(default_factory=dict)

    # -- convenience lookups ------------------------------------------------- #
    @property
    def n_vars(self) -> int:
        return len(self.opportunities)

    def request_by_id(self, rid: str) -> Request:
        return next(r for r in self.requests if r.id == rid)

    def opportunities_for_sat(self, sat_id: str) -> List[Opportunity]:
        return [o for o in self.opportunities if o.sat_id == sat_id]

    def adjacency(self) -> Dict[int, set]:
        """Undirected conflict adjacency over opportunity ids."""
        adj: Dict[int, set] = {o.id: set() for o in self.opportunities}
        for e in self.conflicts:
            adj[e.u].add(e.v)
            adj[e.v].add(e.u)
        return adj

    # -- JSON ---------------------------------------------------------------- #
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "epoch_utc": self.epoch_utc,
            "horizon_hours": self.horizon_hours,
            "penalty_single_service": self.penalty_single_service,
            "penalty_slew": self.penalty_slew,
            "meta": self.meta,
            "satellites": [asdict(s) for s in self.satellites],
            "requests": [asdict(r) for r in self.requests],
            "opportunities": [asdict(o) for o in self.opportunities],
            "conflicts": [asdict(c) for c in self.conflicts],
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: Dict) -> "Instance":
        return cls(
            name=d["name"],
            epoch_utc=d["epoch_utc"],
            horizon_hours=d["horizon_hours"],
            penalty_single_service=d.get("penalty_single_service", 0.0),
            penalty_slew=d.get("penalty_slew", 0.0),
            meta=d.get("meta", {}),
            satellites=[Satellite(**s) for s in d["satellites"]],
            requests=[Request(**r) for r in d["requests"]],
            opportunities=[
                Opportunity(**{**o, "los_ecef": tuple(o.get("los_ecef", (0, 0, 0)))})
                for o in d["opportunities"]
            ],
            conflicts=[ConflictEdge(**c) for c in d["conflicts"]],
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "Instance":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- CSV (flat tables, for spreadsheet / pandas ingestion) --------------- #
    def save_csv(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        with (d / "satellites.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "name", "norad_id", "tle_line1", "tle_line2"])
            for s in self.satellites:
                w.writerow([s.id, s.name, s.norad_id, s.tle_line1, s.tle_line2])

        with (d / "requests.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "name", "lat_deg", "lon_deg", "category", "priority"])
            for r in self.requests:
                w.writerow([r.id, r.name, r.lat_deg, r.lon_deg, r.category, r.priority])

        with (d / "opportunities.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "sat_id", "request_id", "t_start", "t_end", "t_best",
                        "duration_s", "off_nadir_deg", "reward"])
            for o in self.opportunities:
                w.writerow([o.id, o.sat_id, o.request_id, o.t_start, o.t_end,
                            o.t_best, f"{o.duration_s:.1f}", f"{o.off_nadir_deg:.2f}",
                            o.reward])

        with (d / "conflicts.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["kind", "u", "v", "slack_s"])
            for c in self.conflicts:
                w.writerow([c.kind, c.u, c.v,
                            "" if c.slack_s is None else f"{c.slack_s:.1f}"])

    def summary(self) -> str:
        n_ss = sum(1 for c in self.conflicts if c.kind == "single_service")
        n_slew = sum(1 for c in self.conflicts if c.kind == "slew")
        per_sat = {s.id: len(self.opportunities_for_sat(s.id)) for s in self.satellites}
        lines = [
            f"Instance: {self.name}",
            f"  epoch {self.epoch_utc}  horizon {self.horizon_hours} h",
            f"  satellites .......... {len(self.satellites)}",
            f"  requests ............ {len(self.requests)}",
            f"  opportunities (vars)  {self.n_vars}",
            f"  conflict edges ...... {len(self.conflicts)}"
            f"  (single-service {n_ss}, slew {n_slew})",
            f"  vars per satellite .. {per_sat}",
            f"  penalties ........... A(single-service)={self.penalty_single_service}"
            f"  B(slew)={self.penalty_slew}",
        ]
        return "\n".join(lines)
