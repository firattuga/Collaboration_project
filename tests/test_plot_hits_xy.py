import matplotlib.pyplot as plt
from pyscripts.plotting import plot_hits_xy_merged, plot_hits_xy_sensorwise


def test_plot_hits_xy_merged_runs(dummy_tracks):
    plot_hits_xy_merged(dummy_tracks, show_sensors=True)

    fig = plt.gcf()
    ax = fig.axes[0]

    # At least one scatter collection
    assert len(ax.collections) > 0

    plt.close(fig)


def test_plot_hits_xy_sensorwise_runs(dummy_tracks):
    plot_hits_xy_sensorwise(dummy_tracks, show_sensors=True)

    fig = plt.gcf()
    axes = fig.axes

    # 5 sensors → at least 5 axes (1 removed)
    assert len(axes) >= 5

    plt.close(fig)
