import numpy as np
import pandas as pd
from track_filter import RecoTrackFilter

# ----------------------------------------------------------
# 1. Reconstruct individual particle tracks
# ----------------------------------------------------------
def reconstruct_hits(
    csv_path: str,
    Bz: float,
    max_err: float = 1e-2,
    source_tol: float = 0.01,
    n_backtrack: int = 200
):
    """
    Reconstruct all valid particle tracks per event, discarding tracks
    that do not originate from the source (0,0,0) based on backtracked trajectory.

    Parameters
    ----------
    csv_path : str
        Path to the hits CSV file.
    Bz : float
        Magnetic field along z (T).
    max_err : float, optional
        Maximum allowed fit error for a valid track.
    source_tol : float, optional
        Maximum distance from the origin for a particle to be considered originating from the source.
    n_backtrack : int, optional
        Number of points to generate when backtracking particle trajectory.

    Returns
    -------
    tracks : dict
        Dictionary of reconstructed tracks:
            reconstructed_id -> dict with keys:
                - event_id: original event ID
                - path: list of hits
                - fit: output of fit_mq

    data : pd.DataFrame
        Contains reconstructed particle properties:
            Reconstructed ID | EventID | SensorID | x | y | t | mq
    """

    reco = RecoTrackFilter(csv_path)
    tracks = {}
    reconstructed_id = 0

    # CSV storage
    rids, eids, sids, xs, ys, ts, mqs = [], [], [], [], [], [], []

    for event_id, hits in reco.events.items():

        candidate_paths = reco.gen_paths_phys(hits)

        for path in candidate_paths:
            fit = reco.fit_mq(path, B=Bz)
            if fit is None or fit["err"] >= max_err:
                continue

            filtered_path = [h for h in path if h is not None]
            if len(filtered_path) == 0:
                continue

            # --------------------------------------------------
            # Backtrack particle to check origin
            # --------------------------------------------------
            track_dict = {"path": filtered_path, "fit": fit}
            traj = backtrack_particle_trajectory(track_dict, Bz, t_min=0.0, n_points=n_backtrack)
            start_point = traj[0]  # first point in backtracked trajectory

            if np.linalg.norm(start_point) > source_tol:
                continue  # discard particle not originating from source

            # Save hits for CSV
            for hit in filtered_path:
                rids.append(reconstructed_id)
                eids.append(event_id)
                sids.append(hit[0])
                xs.append(hit[1])
                ys.append(hit[2])
                ts.append(hit[4])
                mqs.append(fit["m_over_q"])

            # Save reconstructed track
            tracks[reconstructed_id] = {
                "event_id": event_id,
                "path": filtered_path,
                "fit": fit
            }

            reconstructed_id += 1

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------
    data = pd.DataFrame({
        "Reconstructed ID": rids,
        "EventID": eids,
        "SensorID": sids,
        "x": xs,
        "y": ys,
        "t": ts,
        "mq": mqs
    })
    data.set_index("Reconstructed ID", inplace=True)

    return tracks, data


def backtrack_particle_trajectory(
    track: dict,
    Bz: float,
    t_min: float = 0.0,
    n_points: int = 200
):
    """
    Reconstruct full 3D helix and backtrack particle in time.

    Parameters
    ----------
    track : dict
        One entry from `tracks[event_id]`
    Bz : float
        Magnetic field along z
    t_min : float
        How far back in time to extrapolate
    n_points : int
        Number of trajectory points

    Returns
    -------
    traj : (N, 3) ndarray
        Reconstructed 3D trajectory
    """

    path = track["path"]
    fit = track["fit"]

    # --------------------------------------------------
    # Extract hits
    # --------------------------------------------------
   
    xs = np.array([h[1] for h in path])
    ys = np.array([h[2] for h in path])
    zs = np.array([h[3] for h in path])
    ts = np.array([h[4] for h in path])

    # --------------------------------------------------
    # Fit parameters
    # --------------------------------------------------
    xc, yc, R = fit["xc"], fit["yc"], fit["R"]
    omega = fit["w_ns"]          # rad / ns

    # --------------------------------------------------
    # Infer initial phase from first hit
    # --------------------------------------------------
    phi0 = np.arctan2(ys[0] - yc, xs[0] - xc) - omega * ts[0]

    # --------------------------------------------------
    # Infer vz from z(t)
    # --------------------------------------------------
    vz, z0 = np.polyfit(ts, zs, deg=1)

    # --------------------------------------------------
    # Time range (backwards)
    # --------------------------------------------------
    t_max = ts.max()
    t_vals = np.linspace(t_min, t_max, n_points)

    # --------------------------------------------------
    # Helical trajectory
    # --------------------------------------------------
    x = xc + R * np.cos(phi0 + omega * t_vals)
    y = yc + R * np.sin(phi0 + omega * t_vals)
    z = z0 + vz * t_vals

    return np.column_stack([x, y, z])



# ----------------------------------------------------------
# 2. Characterise each particle by m/q
# ----------------------------------------------------------

def get_mq(reco: "RecoTrackFilter", B: float) -> dict:
    """
    Compute relativistic m/q for each path in each event.

    Parameters
    ----------
    reco : RecoTrackFilter
        Initialized track filter with loaded events
    B : float
        Magnetic field along z (Tesla or compatible units)

    Returns
    -------
    mq_all : dict
        Dictionary mapping event_id -> list of m/q values for each path
    """
    mq_all = {}

    for event_id, hits in reco.events.items():
        paths = reco.gen_paths_phys(hits)
        mq_list = []

        for path in paths:
            fit = reco.fit_mq(path, B)
            if fit is not None:
                mq_list.append(fit["m_over_q"])

        if mq_list:  # only store if there is at least one valid path
            mq_all[event_id] = mq_list

    return mq_all