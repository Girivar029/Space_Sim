import math
import numpy as np
from gravity import BodyProperties, GravityConfig, GravityIntegrator, G, AU
from typing import List, Tuple, Dict, Optional
from enum import Enum
from dataclasses import dataclass

class OrbitType(Enum):
    CIRCULAR = "circular"
    ELLIPTICAL = "elliptical"
    PARABOLIC = "parabolic"
    HYPERBOLIC = "hyperbolic"

@dataclass
class OrbitConfig:
    eccentricity_tolerence: float = 1e-6
    max_orbit_iterations: int = 1000
    orbit_convergence_tolerence: float = 1e-10
    use_adaptive_step: bool = True
    adaptive_step_min: float