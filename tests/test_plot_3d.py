import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from plotting import (
    plot_trajectories_3d,
    animate_trajectories_3d,
)


def test_plot_trajectories_3d_runs(dummy_trajectories, dummy_hits_3d):
    plot_trajectories_3d(dummy_trajectories, dummy_hits_3d)

    fig = plt.gcf()
    ax = fig.axes[0]

    # Lines + source marker
    assert len(ax.lines) > 0

    plt.close(fig)


def test_animate_trajectories_3d(dummy_trajectories, dummy_hits_3d):
    anim = animate_trajectories_3d(
        dummy_trajectories,
        dummy_hits_3d,
        interval=10
    )

    assert isinstance(anim, FuncAnimation)
