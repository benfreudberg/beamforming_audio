from __future__ import annotations
import math
import numpy as np

class Node:
    """Base class for anything with a 3D position (mics, ears, sources, targets)."""
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def distance_to(self, other: Node) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def azel(self, degrees: bool = False) -> tuple[float, float]:
        """
        Return (azimuth, elevation) of this point relative to the origin.

        Convention:
        - azimuth φ = atan2(y, x) in [-pi, pi]
        - elevation θ = atan2(z, sqrt(x^2 + y^2)) in [-pi/2, pi/2]

        If degrees=True, returns (φ, θ) in degrees.

        Raises:
        ValueError if the node is exactly at the origin.
        """
        x, y, z = self.x, self.y, self.z
        if x == 0.0 and y == 0.0 and z == 0.0:
            raise ValueError("Azimuth/elevation undefined at the origin.")

        r_xy = math.hypot(x, y)           # sqrt(x^2 + y^2)
        az = math.atan2(y, x)             # [-pi, pi]
        el = math.atan2(z, r_xy)          # [-pi/2, pi/2]

        if degrees:
            return math.degrees(az), math.degrees(el)
        return az, el

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, z={self.z})"
