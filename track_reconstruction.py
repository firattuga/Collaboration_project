import numpy as np
from track_filter import RecoTrackFilter

# ----------------------------------------------------------
# 1. Reconstruct individual particle tracks
# ----------------------------------------------------------

def reconstruct_hits(
    csv_path: str,
    Bz: float,
    max_err: float = 1e-2
):
    """
    Reconstruct one best particle track per event.

    Returns:
        tracks: dict[event_id] -> dict with keys:
            - path: list of hits
            - fit : output of fit_mq
    """

    reco = RecoTrackFilter(csv_path)
    tracks = {}

    for event_id, hits in reco.events.items():
        # Physics filter
        good_paths = reco.gen_paths_phys(hits)

        best = None
        best_err = np.inf

        # Fit each path and keep the best one
        for path in good_paths:
            fit = reco.fit_mq(path, B=Bz)
            if fit is None:
                continue

            if fit["err"] < best_err and fit["err"] < max_err:
                best_err = fit["err"]
                filtered_path = [h for h in path if h is not None]
                best = {"path": filtered_path, "fit": fit}

        if best is not None:
            tracks[event_id] = best

    return tracks


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