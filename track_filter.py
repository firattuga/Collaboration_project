# ---------------------------------------------------------------------
# track_filter.py
# Relativistic reconstruction of charged particle trajectories in Bz
# ---------------------------------------------------------------------

from __future__ import annotations
import csv
import math
import numpy as np
from collections import defaultdict
from itertools import combinations, product
from typing import Dict, List, Tuple, Optional

# =====================================================================
#                         Types & Constants
# =====================================================================

# HitT: (sensor_id, x, y, z, time_ns)
HitT = Tuple[int, float, float, float, float]

# Speed of light in vacuum (m/ns)
c = 0.299792458

# =====================================================================
#                         Reconstructed Track Loader
# =====================================================================

class RecoTrackFilter:
    """
    Loads reconstructed hits from CSV (SensorID, X, Y, Z, time_ns) and
    provides methods to:
      - generate physics-consistent particle paths (gen_paths_phys)
      - fit circles in XY plane (_fit_circle)
      - estimate relativistic particle mass/charge (fit_mq_rel)
    """

    # ---------------------------------------------------------
    # Constructor: load CSV
    # ---------------------------------------------------------
    def __init__(self, csv_path: str = "hits.csv") -> None:
        self.csv_path = csv_path
        self.events: Dict[int, List[HitT]] = {}
        self._load_csv(csv_path)

    # ---------------------------------------------------------
    # Load CSV into a dictionary mapping EventID → list of hits
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
    # Physics-aware path generator
    # ---------------------------------------------------------
    def gen_paths_phys(self, hits: list) -> list[list[Optional[tuple]]]:
        """
        Generate particle trajectories considering 5, 4, or 3 sensor hits.
        - The first hit sets the initial velocity along z
        - Subsequent hits are checked for time consistency
        - Missing sensor hits are filled with None
        - Paths that are subsets of longer paths are removed
        """
        if not hits:
            return []

        # Group hits by sensor
        by_sensor = {}
        for h in hits:
            by_sensor.setdefault(h[0], []).append(h)

        all_sensors = sorted(by_sensor.keys())
        max_sensors = len(all_sensors)
        final_paths = []

        # ------------------------
        # Check time consistency along z
        # ------------------------
        def is_time_consistent(path):
            h0 = path[0]
            z1, t1 = h0[3], h0[4]
            vz = z1 / t1 if t1 != 0 else float('inf')
            for prev, h in zip(path[:-1], path[1:]):
                z_prev, t_prev = prev[3], prev[4]
                z_next, t_next = h[3], h[4]
                dt_pred = (z_next - z_prev) / vz
                t_pred = t_prev + dt_pred
                if abs(t_next - t_pred) > 1e-3:
                    return False
            return True

        # ------------------------
        # Build paths from 5 to 3 hits
        # ------------------------
        for n in range(min(5, max_sensors), 2, -1):
            sensor_combos = list(combinations(all_sensors, n))
            for sensors in sensor_combos:
                hit_lists = [by_sensor[s] for s in sensors]
                for perm in product(*hit_lists):
                    path = list(perm)
                    if not is_time_consistent(path):
                        continue
                    full_path = []
                    for s in all_sensors:
                        if s in sensors:
                            idx = sensors.index(s)
                            full_path.append(path[idx])
                        else:
                            full_path.append(None)
                    final_paths.append(full_path)

        # ------------------------
        # Remove short paths that are subsets of longer paths
        # ------------------------
        def is_subset(p_short, p_long):
            for hs, hl in zip(p_short, p_long):
                if hs is None:
                    continue
                if hl is None or hs != hl:
                    return False
            return True

        filtered_paths = []
        for p in final_paths:
            if any(
                len([h for h in other if h is not None]) > len([h for h in p if h is not None])
                and is_subset(p, other)
                for other in final_paths
            ):
                continue
            filtered_paths.append(p)

        return filtered_paths

    # ---------------------------------------------------------
    # Circle fit in XY plane (Kåsa method)
    # ---------------------------------------------------------
    def _fit_circle(self, path: List[HitT]):
        """
        Fit a circle to (x,y) hits and estimate cyclotron frequency omega.

        Returns:
            (xc, yc, R, omega_ns)
            omega_ns is in rad/ns
        """
        if len(path) < 3:
            return None

        pts = np.array([[h[1], h[2]] for h in path], float)
        tns = np.array([h[4] for h in path], float)

        # Shift to centroid for numerical stability
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

        # Compute cyclotron frequency from phase slope
        phi = np.unwrap(np.arctan2(pts[:, 1] - yc, pts[:, 0] - xc))
        dt = np.diff(tns)
        if np.any(dt <= 0):
            return None
        dphi = np.diff(phi)
        omega_ns = float(np.median(dphi / dt))

        return float(xc), float(yc), float(R), omega_ns

    # ---------------------------------------------------------
    # Relativistic m/q estimation
    # ---------------------------------------------------------
    def fit_mq(self, path: List[HitT], B: float):
        """
        Compute relativistic m/q from hits along a path in a uniform B field.

        Returns a dictionary with:
            xc, yc, R       : circle center and radius
            omega           : cyclotron frequency (rad/ns)
            m_over_q        : relativistic mass/charge
            q_sign          : +1 or -1
            err             : standard deviation of radial distances (fit quality)
        """
        path = [h for h in path if h is not None]
        if B == 0.0 or len(path) < 3:
            return None

        # Fit circle
        fit = self._fit_circle(path)
        if fit is None:
            return None
        xc, yc, R, omega = fit

        xs = np.array([h[1] for h in path], float)
        ys = np.array([h[2] for h in path], float)
        rs = np.sqrt((xs - xc)**2 + (ys - yc)**2)
        err = float(np.std(rs))

        # Transverse velocity
        v_trans = omega * R

        # z-velocity from first and last hits
        z_first, t_first = path[0][3], path[0][4]
        z_last,  t_last  = path[-1][3], path[-1][4]
        dt_z = t_last - t_first
        if dt_z <= 0:
            return None
        v_z = (z_last - z_first) / dt_z

        # Total velocity
        v_tot = np.sqrt(v_trans**2 + v_z**2)
        if v_tot >= c:
            return None

        # Relativistic gamma
        gamma = 1.0 / np.sqrt(1.0 - (v_tot / c)**2)

        # Relativistic mass/charge
        m_over_q = (B * gamma) / omega

        # Charge sign from rotation direction
        q_sign = +1 if (omega * B) > 0 else -1

        return {
            "xc": xc,
            "yc": yc,
            "R": R,
            "w_ns": omega,
            "m_over_q": m_over_q,
            "q_sign": q_sign,
            "err": err
        }



def get_mq(tracks: dict) -> dict:
    """
    Extract only the relativistic m/q for each reconstructed particle track.

    Parameters
    ----------
    tracks : dict
        Output of `reconstruct_hits`. Keys are event IDs, values are dicts
        containing 'path' and 'fit'.

    Returns
    -------
    mq_dict : dict
        Dictionary mapping event_id -> m/q value.
    """
    mq_dict = {}

    for event_id, track in tracks.items():
        fit = track.get("fit")
        if fit is None:
            continue
        mq_dict[event_id] = fit["m_over_q"]

    return mq_dict