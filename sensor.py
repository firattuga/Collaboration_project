"""
Defines the Sensor class for planar detector elements.
"""
import numpy as np
from typing import List, Optional, Tuple
from .particle import Particle
from .hit import Hit

class Sensor:

    """
    Represents a planar detector sensor with resolution 25μm.
    
    A sensor is a flat detector plane at a fixed z-position that records
    the (x, y) position of particles passing through it.
    
    Attributes:
        id (int): Sensor identifier
        z_position (float): Z-coordinate of sensor plane in meters
        resolution (float): Spatial resolution (Gaussian σ) in meters
        x_range (Tuple[float, float]): Active area x-limits in meters
        y_range (Tuple[float, float]): Active area y-limits in meters
        hits (List[Hit]): All hits recorded by this sensor
    """

    def __init__(self, sensor_id: int, z_position:float , resolution: float = 25e-6, area:Tuple[float,float] = (1.0, 1.0)):

        """
        Initialize a sensor.
        
        Args:
            sensor_id: Unique identifier for this sensor
            z_position: Distance from source in meters
            resolution: Spatial resolution (1σ) in meters (default: 25 μm)
            area: Active area (width_x, width_y) in meters (default: 1m × 1m)
        """

        self.id = sensor_id
        self.z_position = z_position
        self.resolution = resolution # in meters
        # Define area centered at (0,0)
        self.x_range = (-area[0]/2, area[0]/2)
        self.y_range = (-area[1]/2, area[1]/2)
        self.hits : List[Hit] = []

    def detect_hit(self, particle: Particle) -> Optional[Hit]:
        """
        Detect if a particle hits this sensor and record the measurement.
        
        The particle trajectory is propagated to the sensor's z-position.
        If it falls within the active area, a hit is recorded with
        Gaussian measurement noise applied.
        
        Args:
            particle: The particle to check
            
        Returns:
            Hit object if detected, None otherwise
        """
        particle_position = particle.propagate(self.z_position)
        if (self.x_range[0] <= particle_position[0] <= self.x_range[1] and
            self.y_range[0] <= particle_position[1] <= self.y_range[1]):
            # Add noise from resolution
            x_measured = particle_position[0] + np.random.normal(0, self.resolution)
            y_measured = particle_position[1] + np.random.normal(0, self.resolution)
            hit = Hit(sensor_id = self.id, z=self.z_position, y=y_measured, x=x_measured)
            self.hits.append(hit)
            return hit
        return None 
    
    def clear_hits(self):
        '''Clears all recorded hits'''
        self.hits = []