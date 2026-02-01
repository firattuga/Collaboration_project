import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm, colors

from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

cmap = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)+ list(plt.get_cmap("tab20c").colors)

# ----------------------------------------------------------
# 1. Plot hits with source and sensors
# ----------------------------------------------------------

def plot_hits_xy_merged(
    tracks: dict,
    source=(0.0, 0.0),
    show_sensors=True
):
    """
    Plot reconstructed tracks in the x–y plane (all sensors merged),
    using distinct colors per particle.
    Returns fig, ax for testing.
    """

    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot particles
    for idx, (reco_id, data) in enumerate(tracks.items()):
        color = cmap[idx % len(cmap)]

        path = data["path"]
        fit = data["fit"]

        xs = np.array([h[1] for h in path])
        ys = np.array([h[2] for h in path])

        ax.scatter(xs, ys, s=40, color=color, label=str(reco_id))

        xc, yc, R = fit["xc"], fit["yc"], fit["R"]
        phi = np.linspace(0, 2 * np.pi, 400)
        ax.plot(xc + R * np.cos(phi), yc + R * np.sin(phi),
                linestyle="--", color=color, alpha=0.6)

    if show_sensors:
        square = patches.Rectangle((-0.5, -0.5), 1, 1,
                                   edgecolor="orange",
                                   facecolor="none")
        ax.add_patch(square)

    ax.scatter(source[0], source[1], marker="*", s=200, c="red", label="Source")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Reconstructed particle hits (x–y)")
    ax.axis("equal")
    ax.grid(True)

    fig.legend(title="Particle ID", bbox_to_anchor=(1.02, 1), loc='upper right',
               ncols=3, fontsize="xx-small")

    plt.tight_layout()

    return fig, ax



def plot_hits_xy_sensorwise(
        tracks: dict,
        show_sensors: bool = True
):
    """
    Plot reconstructed particle hits in the x–y plane, separated by sensor.

    This function creates a fixed 2×3 grid of subplots (one per sensor,
    with one unused panel removed) and overlays reconstructed hit positions
    for all particles. Each particle is drawn with a unique color across
    all sensors, and a single global legend maps colors to particle IDs.

    Parameters
    ----------
    tracks : dict
        Dictionary of reconstructed particles indexed by particle (or
        reconstruction) ID. Each entry must contain:
            - "path": list of hits, where each hit is a tuple
              (sensor_id, x, y, z, time).
    show_sensors : bool, optional
        If True, draw the 1×1 m sensor outline centered at (0, 0)
        on each subplot. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object containing all subplots.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Array of axes corresponding to the sensor subplots
        (shape 2×3, with the last axis removed).

    Notes
    -----
    - Sensor indices are assumed to run from 0 to 4.
    - Colors are assigned per particle and remain consistent across sensors.
    - The global legend shows particle IDs only once, avoiding duplicates.
    - This function does not call `plt.show()` to allow flexible use
      in scripts, notebooks, and automated tests.
    """

    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(20, 12),
    )

    legend_handles = {}

    for reconstructed_id, data in tracks.items():
        path = data["path"]

        sids = np.array([h[0] for h in path])
        xs = np.array([h[1] for h in path])
        ys = np.array([h[2] for h in path])

        for i in range(5):
            mask = sids == i
            if np.any(mask):
                sc = axs[i // 3][i % 3].scatter(
                    xs[mask],
                    ys[mask],
                    s=40,
                    label=f"{reconstructed_id}",
                    color=cmap[reconstructed_id % len(cmap)]
                )

                # Save one handle per particle (avoid duplicates)
                if reconstructed_id not in legend_handles:
                    legend_handles[reconstructed_id] = sc

    # Sensor outlines and styling
    for i in range(5):
        ax = axs[i // 3][i % 3]
        ax.set_title(f"Sensor {i}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)


        if show_sensors:
            square = patches.Rectangle(
                (-0.5, -0.5), 1, 1,
                edgecolor="orange",
                facecolor="none"
            )
            ax.add_patch(square)

    # Remove unused subplot
    axs[1][2].remove()

    # Global legend
    fig.legend(
        legend_handles.values(),
        [f"{rid}" for rid in legend_handles.keys()],
        ncols=3,
        loc="lower right",
        title="Particle ID",
        frameon=True,
        fontsize="xx-small"
    )

    return fig, axs


def animate_hits_by_time(
    tracks: dict,
    dt: float = 1,
    interval: int = 100,
    show_sensors: bool = True
):
    """
    Animate reconstructed detector hits in the x–y plane as a function of time.

    This function creates a sensor-wise animation (2×3 subplot grid, one
    subplot per sensor) in which reconstructed particle hits appear once
    their hit time is reached. Each particle is assigned a unique, consistent
    color across all sensors. A single global legend maps colors to particle IDs.

    Hits accumulate over time: once a hit appears, it remains visible for
    all subsequent frames.

    Parameters
    ----------
    tracks : dict
        Dictionary of reconstructed particles indexed by particle (or
        reconstruction) ID. Each entry must contain:
            - "path": list of hits, where each hit is a tuple
              (sensor_id, x, y, z, time).
    dt : float, optional
        Time step between animation frames in nanoseconds.
        Default is 1 ns.
    interval : int, optional
        Delay between animation frames in milliseconds.
        Default is 100 ms.
    show_sensors : bool, optional
        If True, draw the 1×1 m sensor outline centered at (0, 0)
        on each subplot. Default is True.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Matplotlib animation object. The caller is responsible for keeping
        a reference to this object and either displaying it with `plt.show()`
        or saving it using `anim.save()`.

    Notes
    -----
    - Sensor indices are assumed to run from 0 to 4.
    - The animation uses fixed time bins defined by `dt`.
    - Hits are revealed when `hit_time <= current_frame_time`.
    - This function does not call `plt.show()` to remain compatible with
      scripts, notebooks, and automated testing environments.
    """

    fig, axs = plt.subplots(nrows=2,ncols=3, figsize=(20, 12))

    # ------------------------------------------------------
    # Assign colors to events
    # ------------------------------------------------------
    reconstructed_ids = sorted(tracks.keys())
    cmap = plt.get_cmap("tab20") # categorical colormap

    particle_colors = {
        rid: cmap(i % cmap.N) for i, rid in enumerate(reconstructed_ids)
    }
    legend_handles = [
    Line2D([0], [0], marker=".", linestyle="",
           color=particle_colors[rid], label=f"{rid}")
    for rid in reconstructed_ids
    ]

    fig.legend(
        handles=legend_handles,
        ncols=3,
        loc='lower right',
        frameon=True,
        fontsize="xx-small",
        title="Particle ID"
    )

    # ------------------------------------------------------
    # Collect all hits (KEEP reconstructed_id)
    # ------------------------------------------------------
    hits = []
    for reconstructed_id, data in tracks.items():
        for h in data["path"]:
            hits.append({
                "ID": reconstructed_id,
                "sid": h[0],
                "x": h[1],
                "y": h[2],
                "z": h[3],
                "t": h[4]
            })

    hits.sort(key=lambda h: h["t"])

    t_max = max(h["t"] for h in hits)
    frame_times = np.arange(0, t_max + dt, dt)

    # ------------------------------------------------------
    # Prepare scatter plots (one per sensor)
    # ------------------------------------------------------
    scatters = []
    for i in range(5):
        sc = axs[i//3][i%3].scatter([], [], s=20)
        scatters.append(sc)

        axs[i//3][i%3].set_title(f"Sensor {i}")
        axs[i//3][i%3].set_xlabel("x")
        axs[i//3][i%3].set_ylabel("y")

        if show_sensors:
            square = patches.Rectangle(
                (-0.5, -0.5), 1, 1,
                edgecolor="orange",
                facecolor="none"
            )
            axs[i//3][i%3].add_patch(square)

    # ------------------------------------------------------
    # Animation callbacks
    # ------------------------------------------------------
    def init():
        for sc in scatters:
            sc.set_offsets(np.empty((0, 2)))
        return scatters

    def update(frame):
        t_now = frame_times[frame]

        for i in range(5):
            xs, ys, cs = [], [], []

            for h in hits:
                if h["sid"] == i and h["t"] <= t_now:
                    xs.append(h["x"])
                    ys.append(h["y"])
                    cs.append(particle_colors[h["ID"]])

            if xs:
                scatters[i].set_offsets(np.column_stack([xs, ys]))
                scatters[i].set_color(cs)
            else:
                scatters[i].set_offsets(np.empty((0, 2)))

            axs[i//3][i%3].set_title(f"Sensor {i}  (t ≤ {t_now:.0f} ns)")

        return scatters

    axs[1][2].remove()
    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_times),
        init_func=init,
        interval=interval,
        blit=False
    )

    plt.tight_layout()

    return anim


def draw_detector_plane(ax, z, size=1.0, color="orange", alpha=0.15):
    """
    Draw a square detector plane in 3D space.

    The detector plane is centered at (x=0, y=0) and lies in a plane of
    constant z. It is rendered both as an outline and as a semi-transparent
    filled surface.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        A Matplotlib 3D axis object on which the detector plane is drawn.

    z : float
        z-coordinate of the detector plane.

    size : float, optional
        Side length of the square detector plane (in the same units as x and y).
        Default is 1.0.

    color : str or tuple, optional
        Color used to draw the detector outline and surface.
        Default is "orange".

    alpha : float, optional
        Transparency of the filled detector surface.
        Must be between 0 (fully transparent) and 1 (fully opaque).
        Default is 0.15.

    Notes
    -----
    - The detector plane is centered at (0, 0, z).
    - The outline is drawn using `ax.plot`, while the filled surface is drawn
      using `ax.plot_trisurf`.
    - This function does not modify axis limits or aspect ratios.

    Examples
    --------
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection="3d")
    >>> draw_detector_plane(ax, z=1.2, size=1.0)
    >>> plt.show()
    """

    half = size / 2

    x = [-half, half, half, -half, -half]
    y = [-half, -half, half, half, -half]
    z = [z] * 5

    ax.plot(x, y, z, color=color, alpha=0.8)
    ax.plot_trisurf(
        x[:4], y[:4], [z[0]] * 4,
        color=color, alpha=alpha
    )

def plot_trajectories_3d(
    trajectories: dict,
    hits: dict
):
    """
    Plot reconstructed 3D particle trajectories together with detector hits,
    the particle source, and detector planes.

    Parameters
    ----------
    trajectories : dict
        particle_id -> array_like of shape (N, 3)
        Reconstructed particle trajectories.

    hits : dict
        particle_id -> list of (x, y, z)
        Detector hits associated with each particle.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure containing the 3D plot.
    """

    particle_ids = sorted(trajectories.keys())

    # Use GLOBAL categorical colormap
    colors = {
        pid: cmap[i % len(cmap)]
        for i, pid in enumerate(particle_ids)
    }

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # --------------------------------------------------
    # Source
    # --------------------------------------------------
    ax.scatter(
        0.0, 0.0, 0.0,
        c="red",
        marker="*",
        s=150,
        label="Source"
    )

    # --------------------------------------------------
    # Detector planes
    # --------------------------------------------------
    detector_zs = [1.0, 1.1, 1.2, 1.3, 1.4]

    for i, z in enumerate(detector_zs):
        draw_detector_plane(ax, z)
        ax.text(0.55, 0.55, z, f"D{i}", color="orange")

    # --------------------------------------------------
    # Trajectories + hits
    # --------------------------------------------------
    for pid in particle_ids:
        traj = np.asarray(trajectories[pid])
        xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]

        ax.plot(
            xs, ys, zs,
            color=colors[pid],
            label=f"{pid}"
        )

        if pid in hits and len(hits[pid]) > 0:
            hx, hy, hz = zip(*hits[pid])
            ax.scatter(
                hx, hy, hz,
                s=20,
                color=colors[pid]
            )

    # --------------------------------------------------
    # Styling
    # --------------------------------------------------
    ax.view_init(20, -120, vertical_axis="y")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D reconstructed particle trajectories")

    ax.set_box_aspect([1, 1, 1.4])
    ax.grid(False)

    plt.tight_layout()
    return fig


def animate_trajectories_3d(
    trajectories: dict,
    hits: dict,
    interval: int = 20
):
    """
    Animate 3D particle trajectories using precomputed trajectories.
    Hits appear when particles cross the detector planes, and disappear
    when the animation restarts.

    Parameters
    ----------
    trajectories : dict
        reconstructed_id -> (N,3) array of particle positions
    hits : dict
        reconstructed_id -> list of (x, y, z) detector hits
    interval : int
        Frame delay in milliseconds

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object.
    """

    from matplotlib.animation import FuncAnimation
    import numpy as np

    reconstructed_ids = sorted(trajectories.keys())
    n_frames = min(len(traj) for traj in trajectories.values())

    # Use global cmap (tab20 + tab20b + tab20c)
    global cmap
    colors = {rid: cmap[i % len(cmap)] for i, rid in enumerate(reconstructed_ids)}

    # -----------------------------
    # Figure setup
    # -----------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Animated 3D particle trajectories")
    ax.view_init(20, -120, vertical_axis="y")
    ax.set_box_aspect([1, 1, 1.4])

    # Source
    ax.scatter(0, 0, 0, c="red", marker="*", s=150, label="Source")

    # Detector planes
    detector_zs = [1.0, 1.1, 1.2, 1.3, 1.4]
    for i, z in enumerate(detector_zs):
        draw_detector_plane(ax, z)
        ax.text(0.55, 0.55, z, f"D{i}", color="orange")

    # -----------------------------
    # Prepare hit scatters
    # -----------------------------
    hit_scatters = {rid: [] for rid in reconstructed_ids}
    for rid in reconstructed_ids:
        for (x, y, z) in hits.get(rid, []):
            sc = ax.scatter([], [], [], s=20, color=colors[rid])
            hit_scatters[rid].append({"x": x, "y": y, "z": z, "scatter": sc})

    # -----------------------------
    # Track lines + moving markers
    # -----------------------------
    lines = {}
    markers = {}

    for rid in reconstructed_ids:
        line, = ax.plot([], [], [], lw=2, color=colors[rid], label=f"{rid}")
        marker = ax.scatter([], [], [], s=20, color=colors[rid])
        lines[rid] = line
        markers[rid] = marker

    # -----------------------------
    # Animation callbacks
    # -----------------------------
    def init():
        # Reset trajectories
        for rid in reconstructed_ids:
            lines[rid].set_data([], [])
            lines[rid].set_3d_properties([])
            markers[rid]._offsets3d = ([], [], [])
            # Reset all hits
            for hit in hit_scatters[rid]:
                hit["scatter"]._offsets3d = ([], [], [])
        return list(lines.values()) + list(markers.values()) + [
            hit["scatter"] for hs in hit_scatters.values() for hit in hs
        ]

    def update(frame):
        for rid in reconstructed_ids:
            traj = np.asarray(trajectories[rid])
            x, y, z = traj[frame]

            # Trail
            lines[rid].set_data(traj[:frame, 0], traj[:frame, 1])
            lines[rid].set_3d_properties(traj[:frame, 2])

            # Moving particle
            markers[rid]._offsets3d = ([x], [y], [z])

            # Reveal hits after crossing
            for hit in hit_scatters[rid]:
                if z >= hit["z"]:
                    hit["scatter"]._offsets3d = (
                        [hit["x"]],
                        [hit["y"]],
                        [hit["z"]],
                    )
                else:
                    hit["scatter"]._offsets3d = ([], [], [])

        return (
            list(lines.values())
            + list(markers.values())
            + [hit["scatter"] for hs in hit_scatters.values() for hit in hs]
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=interval,
        blit=False,
        repeat=True  # ensures animation loops
    )

    plt.tight_layout()
    return anim

def plot_measured_hits_xy_sensorwise(
    csv_path: str,
    show_sensors: bool = True
):
    """
    Plot measured detector hits in the x–y plane, grouped by sensor.

    This function reads a CSV file containing measured detector hits and
    produces a grid of subplots (one per sensor). Signal and noise hits are
    distinguished visually, independent of any event or particle
    reconstruction.

    Signal hits are defined as rows with ``HitID != -1``, while noise hits
    have ``HitID == -1``.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing measured hits. The file must contain
        at least the following columns:

        - ``SensorID`` : int
            Sensor index (expected values: 0–4)
        - ``HitID`` : int
            Hit identifier (-1 indicates noise)
        - ``x_measured`` : float
            Measured x position
        - ``y_measured`` : float
            Measured y position

    show_sensors : bool, optional
        If True, draw the square sensor outline for each subplot
        (default: True).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object.

    axs : numpy.ndarray of matplotlib.axes.Axes
        Array of subplot axes arranged in a 2×3 grid. The last (unused)
        subplot is removed.

    Notes
    -----
    - All sensors are plotted with identical axis limits to allow direct
      visual comparison.
    - A single global legend is added to distinguish signal and noise hits.
    - The plot is event-agnostic and does not require reconstructed tracks.

    Examples
    --------
    >>> fig, axs = plot_measured_hits_xy_sensorwise("hits.csv")
    >>> plt.show()
    """


    df = pd.read_csv(csv_path)

    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(20, 12)
    )

    # --------------------------------------------------
    # Plot hits per sensor
    # --------------------------------------------------
    for sid in range(5):
        ax = axs[sid // 3][sid % 3]

        df_s = df[df["SensorID"] == sid]
        df_noise = df_s[df_s["HitID"] == -1]
        df_signal = df_s[df_s["HitID"] != -1]

        ax.scatter(
            df_signal["x_measured"],
            df_signal["y_measured"],
            s=30,
            alpha=0.7
        )

        ax.scatter(
            df_noise["x_measured"],
            df_noise["y_measured"],
            s=30,
            alpha=0.2,
            color="red"
        )

        ax.set_title(f"Sensor {sid}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)

        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.55, 0.55)

        if show_sensors:
            square = patches.Rectangle(
                (-0.5, -0.5), 1, 1,
                edgecolor="orange",
                facecolor="none"
            )
            ax.add_patch(square)

        from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0],
               marker='o',
               color='w',
               markerfacecolor='C0',
               markersize=8,
               alpha=0.7,
               label='Signal'),
        Line2D([0], [0],
               marker='o',
               color='w',
               markerfacecolor='red',
               markersize=8,
               alpha=0.2,
               label='Noise')
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        title="Hit type",
        frameon=True
    )

    # Remove unused subplot
    axs[1][2].remove()

    return fig, axs
