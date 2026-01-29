# main.py
import numpy as np 
import csv
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from particle import create_random_particle
from sensor import Sensor
import track_reconstruction as reco
import plotting

num_events = 10
avg_particles = 5
avg_noise_hits = 5
b_field_z = 2.0
output_file = "hits.csv"

# Setup Sensors
sensors = []
for i in range(5):
    z_pos = 1.0 + (i * 0.1) 
    sensor = Sensor(sensor_id=i, z_position=z_pos)
    sensors.append(sensor)

print("Starting simulation...")

with open(output_file, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["EventID", "SensorID", "HitID", "x_measured", "y_measured", "z_nominal", "time_ns"])

    for event_id in range(num_events):
        
        # 1. Signal Loop
        n_particles = np.random.poisson(avg_particles)
        for i in range(n_particles):
            particle = create_random_particle(particle_id=i)
            
            # FIXED Indentation here
            for sensor in sensors:
                hit = sensor.detect_hit(particle, b_field_z)
                if hit is not None:
                    writer.writerow([
                        event_id,
                        hit.sensor_id,
                        hit.hit_id, # Is HitID assumed to be unknown when doing reconstruction?
                        f"{hit.x:.6f}",
                        f"{hit.y:.6f}",
                        f"{hit.z:.4f}",
                        f"{hit.time:.4f}"
                    ])

        # 2. Noise Loop
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
                    f"{noise_time:.4f}"
                ])
                
        # Clear sensors for next event
        for sensor in sensors:
            sensor.clear()

print(f"Simulation complete. Hits saved to {output_file}")



    # =================================================================
    #                           reconstruction
    # =================================================================
# Reconstruction of hits and calculation of particle trajectories
tracks = reco.reconstruct_hits(csv_path=output_file, Bz=b_field_z)
print(f"Reconstructed hits: {len(tracks)}")
trajectories = {
    eid: reco.backtrack_particle_trajectory(track, Bz=b_field_z, t_min=0.0)
    for eid, track in tracks.items()
}
hits = {
    eid: [(h[1], h[2], h[3]) for h in track["path"]]
    for eid, track in tracks.items()
}

# Plotting and animating detector hits and particle trajectories
plotting.plot_hits_xy_merged(tracks, source=(0.0, 0.0))
plotting.plot_hits_xy_sensorwise(tracks)

hits_animation=plotting.animate_hits_by_time(tracks, dt=1,interval=100)
#hits_animation.save('hits_animation.gif', writer = 'pillow', fps = 10)
plt.show()

plotting.plot_trajectories_3d(trajectories, hits)

trajectories_animation=plotting.animate_trajectories_3d(trajectories, hits)
#trajectories_animation.save('trajectories_animation.gif', writer = 'pillow', fps = 10)
plt.show()