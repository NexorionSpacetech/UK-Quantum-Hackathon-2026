"""Open-source orbit propagator + access / slew geometry.

This is the "open-source satellite orbit propagator" referenced in the use-case
submission: it turns public Two-Line Element (TLE) sets into the geometric
constraints of the scheduling problem, so teams never have to touch orbital
mechanics during the event.

Pipeline
--------
    TLE  --SGP4-->  satellite position (TEME)
                    --GMST rotation-->  ECEF
    ground target (lat, lon)  --geodetic-->  ECEF
    look geometry  -->  off-nadir angle, elevation, daylight
    contiguous "off-nadir < theta_max" intervals  -->  access windows
    angle between two windows' boresights / slew-rate  -->  slew transition time

Modelling choices (documented so they can be challenged or refined):
  * Spherical Earth, radius 6378.137 km.  Good to a few km on look angles -
    fine for generating a *scheduling* instance, not for precision pointing.
  * TEME -> ECEF uses GMST rotation about the Z axis only (no precession,
    nutation, or polar motion).  Sub-degree error - immaterial here.
  * Low-precision analytic Sun model (~0.01 deg) for the daylight gate.

Dependencies: numpy, sgp4.  Nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import numpy as np
from sgp4.api import Satrec, jday

R_EARTH_KM = 6378.137
DEG = math.pi / 180.0


# --------------------------------------------------------------------------- #
#  Time / frame helpers
# --------------------------------------------------------------------------- #
def _julian(dt: datetime) -> Tuple[float, float]:
    """Return (jd, fr) for sgp4 from a timezone-aware UTC datetime."""
    dt = dt.astimezone(timezone.utc)
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                dt.second + dt.microsecond * 1e-6)


def gmst_rad(jd: float, fr: float) -> float:
    """Greenwich Mean Sidereal Time (radians) - IAU-1982 series."""
    t = (jd + fr - 2451545.0) / 36525.0
    sec = (67310.54841
           + (876600.0 * 3600.0 + 8640184.812866) * t
           + 0.093104 * t * t
           - 6.2e-6 * t * t * t)
    deg = (sec % 86400.0) / 240.0          # 86400 s == 360 deg  ->  /240 deg/s
    return (deg % 360.0) * DEG


def teme_to_ecef(r_teme: np.ndarray, theta: float) -> np.ndarray:
    """Rotate a TEME position vector to ECEF by the GMST angle ``theta``."""
    c, s = math.cos(theta), math.sin(theta)
    x, y, z = r_teme
    return np.array([c * x + s * y, -s * x + c * y, z])


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
    """Spherical-Earth ground point -> ECEF (km)."""
    lat, lon = lat_deg * DEG, lon_deg * DEG
    r = R_EARTH_KM + alt_km
    return np.array([r * math.cos(lat) * math.cos(lon),
                     r * math.cos(lat) * math.sin(lon),
                     r * math.sin(lat)])


def sun_ecef(jd: float, fr: float, theta: float) -> np.ndarray:
    """Low-precision unit Sun direction in ECEF (good to ~0.01 deg)."""
    n = jd + fr - 2451545.0
    L = math.radians((280.460 + 0.9856474 * n) % 360.0)
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 4e-7 * n)
    sun_teme = np.array([math.cos(lam),
                         math.cos(eps) * math.sin(lam),
                         math.sin(eps) * math.sin(lam)])
    return teme_to_ecef(sun_teme, theta)


# --------------------------------------------------------------------------- #
#  Look geometry
# --------------------------------------------------------------------------- #
@dataclass
class LookGeometry:
    off_nadir_deg: float        # angle of the target off the satellite's nadir
    elevation_deg: float        # target-local elevation of the satellite
    sun_elev_deg: float         # target-local solar elevation (daylight if > 0)
    los_ecef: Tuple[float, float, float]   # unit satellite->target vector (ECEF)


def look_geometry(sat_ecef: np.ndarray, tgt_ecef: np.ndarray,
                  sun_dir_ecef: np.ndarray) -> LookGeometry:
    """Off-nadir / elevation / solar-elevation for one sat-target pair."""
    sat2tgt = tgt_ecef - sat_ecef
    rng = np.linalg.norm(sat2tgt)
    los = sat2tgt / rng

    # off-nadir: angle between the boresight-to-target and the nadir (-sat) dir
    nadir = -sat_ecef / np.linalg.norm(sat_ecef)
    off_nadir = math.degrees(math.acos(float(np.clip(np.dot(los, nadir), -1, 1))))

    # local up at the target = target position unit vector (spherical Earth)
    up = tgt_ecef / np.linalg.norm(tgt_ecef)
    tgt2sat = -los
    elev = math.degrees(math.asin(float(np.clip(np.dot(tgt2sat, up), -1, 1))))
    sun_elev = math.degrees(math.asin(float(np.clip(np.dot(sun_dir_ecef, up), -1, 1))))

    return LookGeometry(off_nadir, elev, sun_elev, tuple(los))


# --------------------------------------------------------------------------- #
#  Propagation + access windows
# --------------------------------------------------------------------------- #
@dataclass
class AccessWindow:
    sat_idx: int
    req_idx: int
    t_start: datetime
    t_end: datetime
    t_best: datetime
    min_off_nadir_deg: float
    los_best_ecef: Tuple[float, float, float]


@dataclass
class AccessConfig:
    max_off_nadir_deg: float = 45.0     # agility envelope (how far it can point)
    require_daylight: bool = True       # optical EO needs a sunlit target
    sun_elev_min_deg: float = 0.0       # daylight threshold at the target
    step_s: float = 15.0                # sampling step for window detection
    min_duration_s: float = 0.0         # discard windows shorter than this


def propagate_ecef(sat: Satrec, when: datetime) -> np.ndarray:
    """SGP4 position of ``sat`` at UTC ``when``, in ECEF (km)."""
    jd, fr = _julian(when)
    err, r, _v = sat.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 error code {err}")
    return teme_to_ecef(np.array(r), gmst_rad(jd, fr))


def find_access_windows(sat: Satrec, sat_idx: int,
                        targets: List[Tuple[float, float]],
                        epoch: datetime, horizon_hours: float,
                        cfg: AccessConfig) -> List[AccessWindow]:
    """All access windows of one satellite over the horizon, one per target pass.

    ``targets`` is a list of (lat_deg, lon_deg).  A window is a contiguous run of
    samples for which the target is inside the off-nadir envelope (and sunlit, if
    required); ``t_best`` is the sample of minimum off-nadir angle in that run.
    """
    n_steps = int(horizon_hours * 3600.0 / cfg.step_s) + 1
    tgt_ecef = [geodetic_to_ecef(lat, lon) for (lat, lon) in targets]

    # pre-compute satellite track + sun direction once per time step
    times: List[datetime] = []
    sats_ecef: List[np.ndarray] = []
    suns_ecef: List[np.ndarray] = []
    for k in range(n_steps):
        when = epoch + timedelta(seconds=k * cfg.step_s)
        jd, fr = _julian(when)
        err, r, _v = sat.sgp4(jd, fr)
        if err != 0:
            continue
        theta = gmst_rad(jd, fr)
        times.append(when)
        sats_ecef.append(teme_to_ecef(np.array(r), theta))
        suns_ecef.append(sun_ecef(jd, fr, theta))

    windows: List[AccessWindow] = []
    for j, tgt in enumerate(tgt_ecef):
        in_pass = False
        w_start = w_best_t = None
        w_best_off = 1e9
        w_best_los = (0.0, 0.0, 0.0)
        prev_t = None
        for t, se, su in zip(times, sats_ecef, suns_ecef):
            g = look_geometry(se, tgt, su)
            visible = (g.off_nadir_deg <= cfg.max_off_nadir_deg
                       and g.elevation_deg > 0.0)
            if cfg.require_daylight:
                visible = visible and g.sun_elev_deg >= cfg.sun_elev_min_deg
            if visible:
                if not in_pass:
                    in_pass, w_start = True, t
                    w_best_off = 1e9
                if g.off_nadir_deg < w_best_off:
                    w_best_off, w_best_t, w_best_los = g.off_nadir_deg, t, g.los_ecef
                prev_t = t
            else:
                if in_pass:
                    _close(windows, sat_idx, j, w_start, prev_t, w_best_t,
                           w_best_off, w_best_los, cfg)
                    in_pass = False
        if in_pass:
            _close(windows, sat_idx, j, w_start, prev_t, w_best_t,
                   w_best_off, w_best_los, cfg)
    return windows


def _close(windows, sat_idx, req_idx, t_start, t_end, t_best,
           best_off, best_los, cfg) -> None:
    dur = (t_end - t_start).total_seconds()
    if dur >= cfg.min_duration_s:
        windows.append(AccessWindow(sat_idx, req_idx, t_start, t_end, t_best,
                                     best_off, best_los))


# --------------------------------------------------------------------------- #
#  Slew transition time between two windows on the same satellite
# --------------------------------------------------------------------------- #
def slew_time_s(los_a: Tuple[float, float, float],
                los_b: Tuple[float, float, float],
                slew_rate_deg_s: float, settle_s: float) -> float:
    """Time to reorient the boresight from look-vector A to B (+ settle)."""
    a, b = np.array(los_a), np.array(los_b)
    ang = math.degrees(math.acos(float(np.clip(np.dot(a, b), -1, 1))))
    return ang / slew_rate_deg_s + settle_s
