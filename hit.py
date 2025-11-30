"""
Defines the Hit class for representing detector measurements.
"""

from dataclasses import dataclass

@dataclass
class Hit:
    """
    Represents a single hit measurement on a sensor.
    
    Attributes:
        sensor_id (int): ID of the sensor that recorded this hit
        z (float): Z-position of the hit (along beam axis) in meters
        x (float): X-position of the hit (measured) in meters
        y (float): Y-position of the hit (measured) in meters
    """
    sensor_id: int
    z: float
    y: float
    x: float