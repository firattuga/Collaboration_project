import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm, colors

from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

cmap = list(cm.get_cmap("tab20").colors) + list(cm.get_cmap("tab20b").colors)+ list(cm.get_cmap("tab20c").colors)

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
    """

    fig, ax = plt.subplots(figsize=(7, 7))

    # --------------------------------------------------
    # Plot particles
    # --------------------------------------------------
    for idx, (reco_id, data) in enumerate(tracks.items()):
        color = cmap[idx]

        path = data["path"]
        fit = data["fit"]

        xs = np.array([h[1] for h in path])
        ys = np.array([h[2] for h in path])

        # Hits
        ax.scatter(
            xs,
            ys,
            s=40,
            color=color,
            label=str(reco_id)
        )

        # Fitted circle
        xc, yc, R = fit["xc"], fit["yc"], fit["R"]
        phi = np.linspace(0, 2 * np.pi, 400)

        ax.plot(
            xc + R * np.cos(phi),
            yc + R * np.sin(phi),
            linestyle="--",
            color=color,
            alpha=0.6
        )

    # Sensors
    if show_sensors:
        square = patches.Rectangle(
            (-0.5, -0.5), 1, 1,
            edgecolor="orange",
            facecolor="none"
        )
        ax.add_patch(square)

    # Source
    ax.scatter(source[0], source[1], marker="*", s=200, c="red", label="Source")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Reconstructed particle hits (x–y)")
    ax.axis("equal")
    ax.grid(True)

    # Global legend
    fig.legend(title="Particle ID",bbox_to_anchor=(1.02, 1), loc='upper right', ncols=3, fontsize="xx-small")

    plt.tight_layout()
    plt.show()


def plot_hits_xy_sensorwise(
        tracks: dict,
        show_sensors=True
):
    """
    Plot reconstructed hits in the x–y plane, one subplot per sensor,
    with a single global legend.
    """

    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(20, 12),
    )

    legend_handles = {}

    for event_id, data in tracks.items():
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
                    label=f"{event_id}",
                    color=cmap[event_id % len(cmap)]
                )

                # Save one handle per particle (avoid duplicates)
                if event_id not in legend_handles:
                    legend_handles[event_id] = sc

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
        [f"{eid}" for eid in legend_handles.keys()],
        ncols=3,
        loc="lower right",
        title="Particle ID",
        frameon=True,
        fontsize="xx-small"
    )

    plt.show()


def animate_hits_by_time(
    tracks: dict,
    dt=1,         # ns
    interval=100,
    show_sensors=True
):
    """
    Animate detector hits using fixed time bins.
    Each particle is shown with a different color.
    """

    fig, axs = plt.subplots(nrows=2,ncols=3, figsize=(20, 12))

    # ------------------------------------------------------
    # Assign colors to events
    # ------------------------------------------------------
    event_ids = sorted(tracks.keys())
    cmap = plt.get_cmap("tab20") # categorical colormap

    event_colors = {
        eid: cmap(i % cmap.N) for i, eid in enumerate(event_ids)
    }
    legend_handles = [
    Line2D([0], [0], marker=".", linestyle="",
           color=event_colors[eid], label=f"Event {eid}")
    for eid in event_ids
    ]

    fig.legend(
        handles=legend_handles,
        ncols=3,
        loc='lower right',
        frameon=True,
        fontsize="xx-small"
    )

    # ------------------------------------------------------
    # Collect all hits (KEEP event_id)
    # ------------------------------------------------------
    hits = []
    for event_id, data in tracks.items():
        for h in data["path"]:
            hits.append({
                "event": event_id,
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
                    cs.append(event_colors[h["event"]])

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
    #plt.show()

    return anim


def draw_detector_plane(ax, z, size=1.0, color="orange", alpha=0.15):
    """
    Draw a square detector plane centered at (0,0) at given z.
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
    Plot 3D particle trajectories, detector hits, source and detector planes.

    Parameters
    ----------
    trajectories : dict
        event_id -> (N,3) array of particle positions
    hits : dict
        event_id -> list of (x,y,z) detector hits
    """

    event_ids = sorted(trajectories.keys())
    cmap = plt.get_cmap("tab20")
    colors = {eid: cmap(i % cmap.N) for i, eid in enumerate(event_ids)}

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
    for eid in event_ids:
        traj = trajectories[eid]
        xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]

        ax.plot(
            xs, ys, zs,
            color=colors[eid],
            label=f"Event {eid}"
        )

        if eid in hits:
            hx, hy, hz = zip(*hits[eid])
            ax.scatter(
                hx, hy, hz,
                s=20,
                color=colors[eid]
            )

    # --------------------------------------------------
    # Labels & styling
    # --------------------------------------------------
    ax.view_init(20, -120, vertical_axis="y")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D reconstructed particle tracks")

    #fig.legend(bbox_to_anchor=(1.1, 1), loc='upper right',ncols=3,fontsize="xx-small")
    ax.set_box_aspect([1, 1, 1.4])
    ax.grid(False)

    plt.tight_layout()
    plt.show()

def animate_trajectories_3d(
    trajectories: dict,
    hits: dict,
    interval: int = 20
):
    """
    Animate 3D particle trajectories using precomputed trajectories.

    Parameters
    ----------
    trajectories : dict
        event_id -> (N,3) array of particle positions
    hits : dict
        event_id -> list of (x,y,z) detector hits
    interval : int
        Frame delay in ms
    """

    from matplotlib.animation import FuncAnimation

    event_ids = sorted(trajectories.keys())
    cmap = plt.get_cmap("tab20")

    colors = {eid: cmap(i % cmap.N) for i, eid in enumerate(event_ids)}
    n_frames = min(len(traj) for traj in trajectories.values())

    # --------------------------------------------------
    # Figure setup
    # --------------------------------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Animated 3D particle trajectories")
    ax.view_init(20, -120, vertical_axis="y")
    ax.set_box_aspect([1, 1, 1.4])

    # --------------------------------------------------
    # Source
    # --------------------------------------------------
    ax.scatter(0, 0, 0, c="red", marker="*", s=150, label="Source")

    # --------------------------------------------------
    # Detector planes
    # --------------------------------------------------
    detector_zs = [1.0, 1.1, 1.2, 1.3, 1.4]
    for i, z in enumerate(detector_zs):
        draw_detector_plane(ax, z)
        ax.text(0.55, 0.55, z, f"D{i}", color="orange")

    # --------------------------------------------------
    # Prepare hit scatters (hidden initially)
    # --------------------------------------------------
    hit_scatters = {eid: [] for eid in event_ids}

    for eid in event_ids:
        for (x, y, z) in hits.get(eid, []):
            sc = ax.scatter([], [], [], s=20, color=colors[eid])
            hit_scatters[eid].append({
                "x": x,
                "y": y,
                "z": z,
                "scatter": sc
            })

    # --------------------------------------------------
    # Track lines + moving markers
    # --------------------------------------------------
    lines = {}
    markers = {}

    for eid in event_ids:
        line, = ax.plot([], [], [], lw=2, color=colors[eid], label=f"Event {eid}")
        marker = ax.scatter([], [], [], s=20, color=colors[eid])

        lines[eid] = line
        markers[eid] = marker

    #fig.legend(bbox_to_anchor=(1.1, 1), loc='upper right',ncols=3,fontsize="xx-small")

    # --------------------------------------------------
    # Animation callbacks
    # --------------------------------------------------
    def init():
        for eid in event_ids:
            lines[eid].set_data([], [])
            lines[eid].set_3d_properties([])
            markers[eid]._offsets3d = ([], [], [])
        return list(lines.values()) + list(markers.values())

    def update(frame):
        for eid in event_ids:
            traj = trajectories[eid]
            x, y, z = traj[frame]

            # Trail
            lines[eid].set_data(traj[:frame, 0], traj[:frame, 1])
            lines[eid].set_3d_properties(traj[:frame, 2])

            # Moving particle
            markers[eid]._offsets3d = ([x], [y], [z])

            # Reveal hits after crossing
            for hit in hit_scatters[eid]:
                if z >= hit["z"]:
                    hit["scatter"]._offsets3d = (
                        [hit["x"]],
                        [hit["y"]],
                        [hit["z"]]
                    )

        return (
            list(lines.values())
            + list(markers.values())
            + [h["scatter"] for hs in hit_scatters.values() for h in hs]
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=interval,
        blit=False
    )

    plt.tight_layout()
    #plt.show()

    return anim

def plot_measured_hits_xy_sensorwise(
    csv_path: str,
    show_sensors=True
):
    """
    Plot measured hits from hits.csv in the x–y plane,
    one subplot per sensor, event-agnostic.
    """

    df = pd.read_csv(csv_path)

    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(20, 12),
        constrained_layout=True
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
        ax.axis("equal")
        ax.grid(True)

        #ax.set_xlim(-0.55, 0.55)
        #ax.set_ylim(-0.55, 0.55)

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

    plt.show()
