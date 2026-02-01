import matplotlib.pyplot as plt
from pyscripts.plotting import plot_hits_xy_merged, plot_hits_xy_sensorwise
from matplotlib.collections import PathCollection

def test_plot_hits_xy_merged_runs(dummy_tracks):
    fig, ax = plot_hits_xy_merged(dummy_tracks, show_sensors=True)

    assert len(fig.axes) == 1
    assert ax in fig.axes

    collections = [
        c for c in ax.collections
        if isinstance(c, PathCollection)
    ]
    assert len(collections) > 0

    # Optional: ensure at least one point
    assert collections[0].get_offsets().shape[0] > 0


def test_plot_hits_xy_sensorwise_runs(dummy_tracks):
    plot_hits_xy_sensorwise(dummy_tracks, show_sensors=True)

    fig = plt.gcf()
    axes = fig.axes

    # 5 sensors → at least 5 axes (1 removed)
    assert len(axes) >= 5

    plt.close(fig)
