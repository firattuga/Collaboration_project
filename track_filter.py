# ---------------------------------------------------------------------
# track_filter.py
# Non-relativistic reconstruction of charged particle trajectories in Bz
# ---------------------------------------------------------------------

from __future__ import annotations
import csv
import math
import numpy as np
from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple, Optional

# =====================================================================
#                         Types & Constants
# =====================================================================

# HitT: (sensor_id, x, y, z, time_ns)
HitT = Tuple[int, float, float, float, float]

# SI / HEP constants
C = 299792458.0                                # m/s
E_CHARGE = 1.602176634e-19                     # C
KG_PER_GEV_C2 = 1.78266192e-27                 # kg per (GeV/c^2)
KG_PER_GEVC2_PER_E = KG_PER_GEV_C2 / E_CHARGE  # 1 (GeV/c^2/e) = 1.11265006e-8 kg/C


# =====================================================================
#                         Reconstructed Loader
# =====================================================================

class RecoTrackFilter:
    """
    Loads reconstructed hits (SensorID, X, Y, Z, time_ns) from CSV.

    Provides:
      - gen_paths: generate all full paths (one hit per sensor; no skipping)
      - f_dzdt  : physics filter (Δz≈const, dt>0, Δt≈const)
      - _fit_circle
      - fit_mq  : m/q from circle + angular velocity ω (with uncertainty)
    """

    # ---------------------------------------------------------
    # Constructor: load CSV
    # ---------------------------------------------------------
    def __init__(self, csv_path: str = "hits.csv") -> None:
        self.csv_path = csv_path
        self.events: Dict[int, List[HitT]] = {}
        self._load_csv(csv_path)

    # ---------------------------------------------------------
    # Load CSV into event → hits dictionary
    # ---------------------------------------------------------
    def _load_csv(self, path: str) -> None:
        buckets = defaultdict(list)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ev  = int(row["EventID"])
                sid = int(row["SensorID"])
                x   = float(row["x_measured"])
                y   = float(row["y_measured"])
                z   = float(row["z_nominal"])
                t   = float(row["time_ns"])
                buckets[ev].append((sid, x, y, z, t))
        self.events = dict(buckets)

    # ---------------------------------------------------------
    # Generate all full paths (one hit per sensor; sensors sorted)
    # ---------------------------------------------------------
    def gen_paths(self, hits: List[HitT]) -> List[List[HitT]]:
        if not hits:
            return []
        by_s = defaultdict(list)
        for h in hits:
            by_s[h[0]].append(h)
        sensors = sorted(by_s.keys())
        lists = [by_s[s] for s in sensors]
        paths: List[List[HitT]] = []
        for combo in product(*lists):
            paths.append(list(combo))
        return paths

    # ---------------------------------------------------------
    # Filter: enforce Δz≈const, dt>0, and Δt≈const across sensors
    # ---------------------------------------------------------
    def f_dzdt(self, paths: List[List[HitT]], dz=0.1,
               dz_tol=1e-6, dt_tol=1e-3) -> List[List[HitT]]:
        kept: List[List[HitT]] = []
        for p in paths:
            if len(p) < 2:
                continue
            ok = True
            dts: List[float] = []
            for i in range(len(p)-1):
                z0, t0 = p[i][3], p[i][4]
                z1, t1 = p[i+1][3], p[i+1][4]
                # constant Δz
                if abs((z1 - z0) - dz) > dz_tol:
                    ok = False
                    break
                # strictly increasing time
                dt = t1 - t0
                if dt <= 0:
                    ok = False
                    break
                dts.append(dt)
            # constant Δt
            if ok and all(abs(dt - dts[0]) <= dt_tol for dt in dts):
                kept.append(p)
        return kept

    # ---------------------------------------------------------
    # Circle fit: algebraic least squares (Kåsa) with centroid shift
    # ---------------------------------------------------------
    def _fit_circle(self, path: List[HitT]):
        pts = np.array([[h[1], h[2]] for h in path], float)
        if len(pts) < 3:
            return None

        # Step 1: shift to centroid for numerical stability
        mx, my = pts.mean(axis=0)
        X = pts[:, 0] - mx
        Y = pts[:, 1] - my

        # Step 2: solve D*x + E*y + F = -(x^2 + y^2)
        A = np.column_stack([X, Y, np.ones(len(pts))])
        b = -(X*X + Y*Y)
        D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]

        xc0 = -D / 2
        yc0 = -E / 2
        R2  = xc0*xc0 + yc0*yc0 - F
        if R2 <= 0:
            return None

        # Step 3: shift center back to original coordinates
        xc = xc0 + mx
        yc = yc0 + my
        R  = math.sqrt(R2)

        return float(xc), float(yc), float(R)

    # ---------------------------------------------------------
    # Compute m/q from circle radius + angular frequency ω
    # ---------------------------------------------------------
    def fit_mq(self, path: List[HitT], B: float, method: str = "median"):
        if B == 0.0 or len(path) < 3:
            return None

        # Circle fit (use all points)
        fit = self._fit_circle(path)
        if fit is None:
            return None
        xc, yc, R = fit

        xs = np.array([h[1] for h in path], float)
        ys = np.array([h[2] for h in path], float)
        tns = np.array([h[4] for h in path], float)  # ns (already ascending)

        # radial residuals as a quality metric for the circle fit
        rs  = np.sqrt((xs - xc)**2 + (ys - yc)**2)
        err = float(np.std(rs))

        # Phase φ(t)
        phi = np.unwrap(np.arctan2(ys - yc, xs - xc))

        # Angular velocity ω (rad/ns)
        if method == "linfit":
            # linear least squares: φ = ω t + φ0
            A = np.vstack([tns, np.ones_like(tns)]).T
            w_ns, b = np.linalg.lstsq(A, phi, rcond=None)[0]
            phi_fit = w_ns * tns + b
            resid = phi - phi_fit
            n = len(tns)
            if n >= 3:
                s2  = float((resid**2).sum()/(n-2))
                Sxx = float(((tns - tns.mean())**2).sum())
                dw_ns = math.sqrt(s2/Sxx) if Sxx > 0 else 0.0
            else:
                dw_ns = 0.0
        else:  # robust median slope from differences
            dphi = np.diff(phi)
            dt   = np.diff(tns)
            wi   = dphi / dt
            w_ns = float(np.median(wi))
            mad  = float(np.median(np.abs(wi - w_ns))) if len(wi) > 1 else 0.0
            dw_ns= 1.4826*mad/math.sqrt(len(wi)) if len(wi) > 1 else 0.0

        # Convert to SI (rad/s)
        w_si  = w_ns * 1e9
        dw_si = dw_ns* 1e9
        if w_si == 0:
            return None

        # m/q and q/m
        qm    = w_si / B          # C/kg
        mq_SI = B / w_si          # kg/C
        relw  = abs(dw_si / w_si)
        mq_SI_err = abs(mq_SI) * relw

        # Convert to GeV/c^2 per e
        mq_GeV     = mq_SI     / KG_PER_GEVC2_PER_E
        mq_GeV_err = mq_SI_err / KG_PER_GEVC2_PER_E

        # Charge sign: positive charge rotates CCW when Bz > 0 (ω > 0)
        q_sign = +1 if (w_si * B) > 0 else -1

        return {
            "xc": xc, "yc": yc, "R": R, "err": err,
            "w_ns": w_ns, "dw_ns": dw_ns,
            "mq_SI": mq_SI, "mq_SI_err": mq_SI_err,
            "mq_GeV": mq_GeV, "mq_GeV_err": mq_GeV_err,
            "qm": qm, "q_sign": q_sign
        }
