# sensor.py
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# Type checking import to avoid circular dependency
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .particle import Particle

@dataclass
class Hit:
    """
    Represents a single hit measurement recorded by a sensor.
    Stores both the measured (smeared) data and the true Monte Carlo values.

    Attributes:
        hit_id     : unique hit identifier
        sensor_id  : ID of the sensor that recorded this hit
        particle_id: ID of the originating particle

        x, y, z    : measured hit position (with detector resolution effects)
        time       : timestamp of the hit (ns or compatible units)

        x_true, y_true, z_true : true (unsmeared) hit position from simulation
    """
    hit_id: int
    sensor_id: int
    particle_id: int
    
    # Measured (Data)
    x: float
    y: float
    z: float
    time: float
    
    # Truth (MC)
    x_true: float
    y_true: float
    z_true: float

class Sensor:
    """
    Models a planar detector sensor positioned at a fixed z-coordinate.
    Responsible for detecting particle hits that intersect its active area.

    Attributes:
        id          : sensor identifier
        z_position  : z-coordinate (plane position in detector)
        half_width  : half of the sensor width (x direction)
        half_height : half of the sensor height (y direction)
        resolution  : spatial resolution (Gaussian sigma for smearing)
        hits        : list of Hit objects recorded by this sensor
        _hit_counter: internal counter for assigning unique hit IDs
    """
    def __init__(self, sensor_id: int, z_position: float, width: float = 1.0, height: float = 1.0, 
                 resolution: float = 25e-6):
        self.id = sensor_id
        self.z_position = z_position
        self.half_width = width / 2
        self.half_height = height / 2  # FIXED typo 'heigh'
        self.resolution = resolution
        
        self.hits: List[Hit] = [] # list of recorded hits
        self._hit_counter = 0 # unique ID counter for hits

    def clear(self):
        """
        Reset the sensor by clearing all recorded hits and resetting the counter.
        Useful between simulation events or runs.
        """
        self.hits = []
        self._hit_counter = 0 

    def detect_hit(self, particle: "Particle", b_field_z: float = 0) -> Optional[Hit]:
        """
        Simulate the detection of a particle crossing the sensor plane.

        Steps:
          1. Propagate the particle to the sensor's z-plane.
          2. Check if it intersects the active sensor area (within width/height).
          3. Apply Gaussian smearing to emulate measurement resolution.
          4. Create and store a Hit object for both the sensor and particle.

        Args:
            particle   : Particle object to propagate
            b_field_z  : magnetic field along z (Tesla or compatible units)

        Returns:
            Hit object if the particle crosses the sensor, otherwise None.
        """
        # 1. Propagate
        particle_position = particle.propagation_to_z(self.z_position, b_field_z)
        
        if particle_position is None:
            return None
        
        x_true, y_true, z_true = particle_position

        # 2. Check Area (FIXED typo 'halff_height')
        if (-self.half_width <= x_true <= self.half_width and
            -self.half_height <= y_true <= self.half_height):
            
            # 3. Smear
            x_measured = x_true + np.random.normal(0, self.resolution)
            y_measured = y_true + np.random.normal(0, self.resolution)

            new_hit = Hit(
                hit_id = self._hit_counter,
                sensor_id = self.id,
                particle_id = particle.id,
                x = x_measured,
                y = y_measured,
                z = self.z_position,
                time = particle.time,
                x_true = x_true,
                y_true = y_true,
                z_true = z_true
            )

            # FIXED: append 'new_hit', not 'hit'
            self.hits.append(new_hit)
            self._hit_counter += 1
            particle.true_hits.append(new_hit)
            
            return new_hit
            
        return None
