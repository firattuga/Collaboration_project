# particle.py
import numpy as np
from typing import List, Optional
import math

# Use TYPE_CHECKING to avoid circular import with sensor.py
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .sensor import Hit

# --- CONSTANTS ---
c = 0.299792458 

PARTICLE_DB = {
    "electron": (-1, 0.000511),
    "muon":     (-1, 0.10566),
    "pion":     ( 1, 0.13957),
    "proton":   ( 1, 0.93827),
    "alpha":    ( 2, 3.7273),
    "mystery":  (-1, 500.0)
}

class Particle:
    def __init__(self, particle_id: int, charge: float, mass: float,
                 momentum: np.ndarray, time: float = 0.0):
        self.id = particle_id
        self.charge = charge
        self.mass = mass
        self.momentum = momentum
        self.position = np.array([0.0, 0.0, 0.0]) 
        self.time = time
        self.true_hits: List['Hit'] = []

    def get_energy(self) -> float:
        p_magnitude = np.linalg.norm(self.momentum)
        return np.sqrt(p_magnitude**2 + self.mass**2)
        
    def get_velocity(self) -> np.ndarray:
        energy = self.get_energy()
        return (self.momentum / energy) * c 
        
    def propagation_to_z(self, z_target: float, b_field_z: float = 0) -> Optional[np.ndarray]:
        delta_z = z_target - self.position[2]
        v = self.get_velocity()

        # SAFETY CHECK: Avoid division by zero
        if v[2] == 0 or (delta_z > 0 and v[2] < 0):
            return None

        dt = delta_z / v[2] 

        if b_field_z == 0:
            self.position += v * dt
        else:
            energy = self.get_energy()
            omega = (c**2 * b_field_z * self.charge) / energy 

            d_phi = -omega * dt
            
            px, py = self.momentum[0], self.momentum[1]
            
            # Rotate Momentum
            cos_dphi = math.cos(d_phi)
            sin_dphi = math.sin(d_phi)
            self.momentum[0] = px * cos_dphi - py * sin_dphi
            self.momentum[1] = px * sin_dphi + py * cos_dphi

            # Update Position
            vx, vy = v[0], v[1]
            dx = (vx * sin_dphi - vy * (1 - cos_dphi)) / omega
            dy = (vy * sin_dphi + vx * (1 - cos_dphi)) / omega
            
            self.position[0] += dx
            self.position[1] += dy
            self.position[2] = z_target 

        self.time += dt
        return self.position # <--- ADDED THIS

def create_random_particle(particle_id: int) -> Particle:
    name = np.random.choice(list(PARTICLE_DB.keys()))
    charge, mass = PARTICLE_DB[name]
    
    p_mag = np.random.uniform(1.0, 10.0) 
    theta = np.random.uniform(0, np.pi / 6)
    phi = np.random.uniform(0, 2 * np.pi)
    
    px = p_mag * math.sin(theta) * math.cos(phi)
    py = p_mag * math.sin(theta) * math.sin(phi)
    pz = p_mag * math.cos(theta)
    
    return Particle(particle_id, charge, mass, np.array([px, py, pz]))