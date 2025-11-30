
'''Defines the Particle class and generates a random direction for the particle.

Attributes:
    id (int): Unique identifier for the particle.
    origin (np.ndarray): The starting position of the particle in 3D space in meters (always assumed to be 0,0,0).
    direction (np.ndarray): The (unit) direction vector of the particle in 3D space.
    true_hits (List[Hit]): List of true hits associated with the particle.
'''

import numpy as np
from typing import List
import math
from .hit import Hit

class Particle:
    def __init__(self, particle_id: int, origin: np.ndarray = np.array([0,0,0])):
        """
        Initialize a particle with random direction.
        
        Args:
            particle_id: Unique identifier for tracking
            origin: Starting position (default: origin)
        """
        self.id = particle_id
        self.origin = origin
        self.direction = self.generate_random_direction()
        self.true_hits : List[Hit] = []

    def generate_random_direction(self) -> np.ndarray:
        """
        Generate a random direction vector within a cone.
        
        The cone has a maximum opening angle of 30 degrees (π/6 radians)
        from the z-axis. This models particles from a collimated source.
        
        Returns:
            Normalized direction vector [dx, dy, dz]
        """        
        theta_max = np.pi / 6
        #any degree between 0 and theta_max is equally probable
        theta = np.random.uniform(0, theta_max)
        phi = np.random.uniform(0, 2 * np.pi)
        # Convert spherical to Cartesian coordinates
        direction = np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta)
        ])
        #Normalize to ensure unit length (sometimes numerical errors may cause the vector to not be unit length)
        direction = direction / np.linalg.norm(direction)  
        return direction
    
    def propagate(self, distance: float) -> np.ndarray:
        """
        Propagate the particle along its direction for a given distance.
        
        Uses straight-line propagation (no magnetic field).
        Distance is used instead of time since we only care about geometry,
        not the particle's speed.
        
        Args:
            distance: Distance to propagate in meters
            
        Returns:
            Position [x, y, z] at the given distance
        """
        return self.origin + self.direction * distance