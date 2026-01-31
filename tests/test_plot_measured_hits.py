import pandas as pd
import tempfile
from pyscripts.plotting import plot_measured_hits_xy_sensorwise


def test_plot_measured_hits_xy_sensorwise_runs():
    df = pd.DataFrame({
        "EventID": [0, 0, 0],
        "SensorID": [0, 1, 2],
        "HitID": [0, -1, 1],
        "x_measured": [0.1, 0.2, -0.1],
        "y_measured": [0.0, 0.1, -0.2],
        "z_nominal": [1.0, 1.1, 1.2],
        "time_ns": [1.0, 2.0, 3.0],
    })

    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        df.to_csv(f.name, index=False)
        plot_measured_hits_xy_sensorwise(f.name, show_sensors=True)
