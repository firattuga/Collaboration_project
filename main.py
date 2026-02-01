# main.py
import numpy as np
import csv
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    # Fall back to non-interactive backend when TkAgg is unavailable
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle import create_random_particle
from sensor import Sensor
import track_reconstruction as reco
import plotting


# ADD-ON imports
from time_gate_addon import time_gate_filter_csv, compare_reco_quality

# ---------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------
num_events = 10
avg_particles = 5
avg_noise_hits = 5
b_field_z = 2.0
output_file = "hits.csv"
particle_id = 0

# ---------------------------------------------------------
# Setup Sensors
# ---------------------------------------------------------
sensors = []
for i in range(5):
    z_pos = 1.0 + (i * 0.1)
    sensor = Sensor(sensor_id=i, z_position=z_pos)
    sensors.append(sensor)

print("Starting simulation...")

# ---------------------------------------------------------
# Write hits.csv
# ---------------------------------------------------------
with open(output_file, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "EventID", "SensorID", "HitID",
        "x_measured", "y_measured", "z_nominal",
        "time_ns", "ParticleID", "m_over_q"
    ])

    for event_id in range(num_events):

        # 1) Signal loop
        n_particles = np.random.poisson(avg_particles)
        for i in range(n_particles):
            particle = create_random_particle(particle_id=i + particle_id)

            for sensor in sensors:
                hit = sensor.detect_hit(particle, b_field_z)
                if hit is not None:
                    writer.writerow([
                        event_id,
                        hit.sensor_id,
                        hit.hit_id,
                        f"{hit.x:.6f}",
                        f"{hit.y:.6f}",
                        f"{hit.z:.4f}",
                        f"{hit.time:.4f}",
                        particle.id,
                        particle.mass / particle.charge
                    ])

        particle_id += n_particles

        # 2) Noise loop
        for sensor in sensors:
            n_noise = np.random.poisson(avg_noise_hits)
            for _ in range(n_noise):
                noise_x = np.random.uniform(-sensor.half_width, sensor.half_width)
                noise_y = np.random.uniform(-sensor.half_height, sensor.half_height)
                noise_time = np.random.uniform(0, 25)

                writer.writerow([
                    event_id,
                    sensor.id,
                    -1,
                    f"{noise_x:.6f}",
                    f"{noise_y:.6f}",
                    f"{sensor.z_position:.4f}",
                    f"{noise_time:.4f}",
                    -1,
                    -1
                ])

        # Clear sensors for next event
        for sensor in sensors:
            sensor.clear()

print(f"Simulation complete. Hits saved to {output_file}")


# =========================================================
# 1) ORIGINAL: plots + reconstruction + animations (raw hits)
# =========================================================
print("\n================ ORIGINAL PIPELINE (RAW) ================\n")

# Measured hits plot (raw)
plotting.plot_measured_hits_xy_sensorwise(output_file)
plt.show()

# Reconstruction (raw)
tracks_raw = reco.reconstruct_hits(csv_path=output_file, Bz=b_field_z)

# Backtrack trajectories (raw)
trajectories_raw = {
    rid: reco.backtrack_particle_trajectory(track, Bz=b_field_z, t_min=0.0)
    for rid, track in tracks_raw.items()
}
hits_raw = {
    rid: [(h[1], h[2], h[3]) for h in track["path"]]
    for rid, track in tracks_raw.items()
}

# Plots/animations (raw)
plotting.plot_hits_xy_merged(tracks_raw, source=(0.0, 0.0))
plotting.plot_hits_xy_sensorwise(tracks_raw)

hits_anim_raw = plotting.animate_hits_by_time(tracks_raw, dt=1, interval=100)
plt.show()

plotting.plot_trajectories_3d(trajectories_raw, hits_raw)

traj_anim_raw = plotting.animate_trajectories_3d(trajectories_raw, hits_raw)
plt.show()


# =========================================================
# 2) ADD-ON: time-gate filter + reconstruction + plots
# =========================================================
print("\n================ ADD-ON PIPELINE (TIME-GATED) ================\n")

filtered_csv = time_gate_filter_csv(
    input_csv=output_file,
    output_csv="hits_timegated.csv",
    dt_window_ns=1.0
)

plt.show()  # show add-on filter diagnostic plots

# Measured hits plot (filtered)
plotting.plot_measured_hits_xy_sensorwise(filtered_csv)
plt.show()

# Reconstruction (filtered)
tracks_filtered = reco.reconstruct_hits(csv_path=filtered_csv, Bz=b_field_z)

# Compare quality (raw vs filtered) – add-on plots
compare_reco_quality(tracks_raw, tracks_filtered)
plt.show()

# Backtrack trajectories (filtered)
trajectories_filtered = {
    rid: reco.backtrack_particle_trajectory(track, Bz=b_field_z, t_min=0.0)
    for rid, track in tracks_filtered.items()
}
hits_filtered = {
    rid: [(h[1], h[2], h[3]) for h in track["path"]]
    for rid, track in tracks_filtered.items()
}

# Plots/animations (filtered)
plotting.plot_hits_xy_merged(tracks_filtered, source=(0.0, 0.0))
plotting.plot_hits_xy_sensorwise(tracks_filtered)

hits_anim_filtered = plotting.animate_hits_by_time(tracks_filtered, dt=1, interval=1000)
plt.show()

plotting.plot_trajectories_3d(trajectories_filtered, hits_filtered)

traj_anim_filtered = plotting.animate_trajectories_3d(trajectories_filtered, hits_filtered)
plt.show()
