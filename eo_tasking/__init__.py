"""Earth-observation satellite tasking & scheduling - hackathon toolkit.

A small, self-contained, open-source package for the UK Quantum Hackathon 2026
use case "Hybrid quantum-classical optimisation for Earth observation satellite
tasking and scheduling" (Nexorion Spacetech).

Modules
-------
instance     dataclasses for the scheduling instance + load/save (JSON & CSV).
propagator   open-source SGP4 orbit propagator + access / slew geometry.
generate     turn public TLEs into a ready-to-ingest scheduling instance.
qubo         build the QUBO / Ising cost function from an instance.
baseline     the classical baselines: greedy heuristic + brute-force optimum.

The data products in ``dataset/`` are produced by ``generate.py`` and can be
ingested directly without running any of this code (see DATA_PROVISION.md).
"""

from .instance import (
    Satellite,
    Request,
    Opportunity,
    ConflictEdge,
    Instance,
)

__all__ = [
    "Satellite",
    "Request",
    "Opportunity",
    "ConflictEdge",
    "Instance",
]

__version__ = "1.0.0"
