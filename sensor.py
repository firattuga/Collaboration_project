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
    """Single electron hit measurement on a sensor."""
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
    def __init__(self, sensor_id: int, z_position: float, width: float = 1.0, height: float = 1.0, 
                 resolution: float = 25e-6):
        self.id = sensor_id
        self.z_position = z_position
        self.half_width = width / 2
        self.half_height = height / 2  # FIXED typo 'heigh'
        self.resolution = resolution
        
        self.hits: List[Hit] = []
        self._hit_counter = 0 

    def clear(self):
        self.hits = []
        self._hit_counter = 0 

    def detect_hit(self, particle: "Particle", b_field_z: float = 0) -> Optional[Hit]:
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
