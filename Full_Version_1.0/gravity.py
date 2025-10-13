import math
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

G = 6.67430e-11
C = 299792458.0
SOLAR_MASS = 1.989e30
SOLAR_RADIUS = 6.96e8
EARTH_MASS = 5.972e24
EARTH_RADIUS = 6.371e6
AU = 1.496e11
PARSEC = 3.086e16
LIGHT_YEAR = 9.461e15

DISTANCE_SCALE = 1.0 / AU
MASS_SCALE = 1.0 / SOLAR_MASS
TIME_SCALE = 1.0
VELOCITY_SCALE = 1.0
FORCE_SCALE = 1.0

BASE_SOFTENING = 1e9
PLANET_SOFTENING_FACTOR = 1.0
STAR_SOFTENING_FACTOR = 2.0
BLACKHOLE_SOFTENING_FACTOR = 0.1
SUPERNOVA_SOFTENING_FACTOR = 5.0
NEURON_STAR_SOFTENING_FACTOR = 0.5
WHITE_DWARF_SOFTENING_FACTOR = 1.5

RELATIVISTIC_DISTANCE_THRESHOLD = 10.0
POST_NEWTONIAN_VELOCITY_THRESHOLD = 0.01 * C
SCHWARZSCHILD_RADIUS_MULTIPLIER = 5.0
PHOTON_SPHERE_MULTIPLIER = 1.5
ISCO_MULTIPLIER = 3.0

ROCHE_LIMIT_FACTOR = 2.456
RIGID_ROCHE_FACTOR = 2.46
FLUID_ROCHE_FACTOR = 2.88
TIDAL_FORCE_CUTOFF = 100.0
TIDAL_DISRUPTION_THRESHOLD = 0.95

FRAME_DRAG_COEFFECIENT = 0.1
SPIN_ANGULAR_MOMENTUM_FACTOR = 0.4
LENSE_THIRRING_FACTOR = 2.0
KERR_PARAMETER_MAX = 0.998

GW_LUMINOSITY_COEFFECIENT = 32.0 / 5.0
GW_ENERGY_LOSS_FACTOR = 1.0e-10
GW_FREQUENCY_FACTOR = 1.0
CHIRP_MASS_POWER = 5.0 * 3.0

NEWTONIAN_ACCURACY = 1e-6
POST_NEWTONIAN_ACCURACY = 1e-8
RELATIVISTIC_ACCURACY = 1e-10
INTEGRATION_TOLERANCE = 1e-12

MAX_FORCE_MAGNITUDE = 1e30
MIN_DISTANCE = 1e6
MAX_DISTANCE = 1e15
MIN_MASS = 1e20
MAX_MASS = 1e40

SUPERNOVA_CORE_FRACTION = 0.1
SUPERNOVA_ENVELOPE_FRACTION = 0.9
SUPERNOVA_EXPANSION_VELOCITY = 1e7
SUPERNOVA_LIFETIME = 3.154e7

BLACKHOLE_ACCERATION_EFFICIENCY = 0.1
BLACKHOLE_EDDINGTON_FACTOR = 1.0
BLACKHOLE_SPIN_DAMPING = 0.01
BLACKHOLE_HORIZON_SAFETY = 1.1

NEUTRON_STAR_MAX_MASS = 2.5 * SOLAR_MASS
NEUTRON_STAR_RADIUS = 1e4
NEUTRON_STAR_DENSITY = 1e18
NEUTRON_STAR_CRUST_FRACTION = 0.05

WHITE_DWARF_MAX_MASS = 1.44 * SOLAR_MASS
WHITE_DWARF_RADIUS = 7e6
WHITE_DWARF_DENSITY = 1e9
CHANDRASEKHAR_LIMIT = 1.44

PLANET_MIN_MASS = 1e22
PLANET_MAX_MASS = 1e28
PLANET_DENSITY_ROCKY = 5500.0
PLANET_DENSITY_ICE = 1000.0
PLANET_DENSITY_GAS = 1300.0

class BodyType(Enum):
    PLANET = "planet"
    STAR = "star"
    BLACKHOLE = "blackhole"
    SUPERNOVA = "supernova"
    NEUTRON_STAR = "neutron_star"
    WHITE_DWARF = "white_dwarf"
    ASTEROID = "asteroid"
    COMET = "comet"

class GravityModel(Enum):
    NEWTONIAN = "newtonian"
    POST_NEWTONIAN = "post_newtonian"
    SCHWARZSCHILD = "schwarzshild"
    KERR = "kerr"
    FULL_RELATIVITY = "full_relativity"

class IntegrationModule(Enum):
    EULER = "euler"
    LEAPFROG = "leapfrog"
    RK4 = "rk4"
    ADAPTIVE = "adaptive"

@dataclass
class GravityConfig:
    model: GravityModel = GravityModel.POST_NEWTONIAN
    integration_method: IntegrationModule = IntegrationModule.RK4
    enable_tidal_forces: bool = True
    enable_frame_dragging: bool = False
    enable_gw_radiation: bool = False
    enable_softening: bool = True
    enable_relativistic_corrections: bool = True
    softening_length: float = BASE_SOFTENING
    max_force_magnitude: float = MAX_FORCE_MAGNITUDE
    min_distance: float = MIN_DISTANCE
    accuracy_tolerance: float = INTEGRATION_TOLERANCE
    adaptive_timestep: bool = True
    use_barnes_hut: bool = False
    barnes_hut_theta: float = 0.5


@dataclass
class BodyProperties:
    body_type: BodyType
    mass: float
    radius: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray = None
    spin_angular_momentum: float = 0.0
    spin_axis: np.ndarray = None
    is_extended: bool = False
    density: float = 0.0
    temperature: float = 0.0
    luminosity: float = 0.0
    schwarzschild_radius: float = 0.0
    isco_radius: float = 0.0
    photon_sphere_radius: float = 0.0
    kerr_parameter: float = 0.0
    tidal_love_number: float = 0.0

    def __post_innit__(self):
        if self.acceleration is None:
            self.acceleration = np.array([0.0,0.0])
        if self.spin_axis is None:
            self.spin_axis = np.array([0.0,0.0,1.0])
        
        if self.body_type == BodyType.BLACKHOLE:
            self.schwarzschild_radius = 2.0 * G * self.mass / (C * C)
            self.photon_sphere_radius = 1.5 * self.schwarzschild_radius
            self.isco_radius = 3.0 * self.schwarzschild_radius
            if self.spin_angular_momentum > 0:
                self.kerr_parameter = min(
                    (C * self.spin_angular_momentum) / (G * self.mass * self.mass),
                    KERR_PARAMETER_MAX
                )
            
        if self.is_extended and self.radius > 0:
            volume = (4.0 / 3.0) * math.pi * (self.radius ** 3)
            self.density = self.mass / volume

        if self.body_type == BodyType.PLANET:
            if self.density == 0.0:
                self.density = PLANET_DENSITY_ROCKY
            self.tidal_love_number = 0.3

def get_softening_for_body_type(body_type: BodyType, base_softening: float) -> float:
    softening_map = {
        BodyType.PLANET: PLANET_SOFTENING_FACTOR,
        BodyType.STAR: STAR_SOFTENING_FACTOR,
        BodyType.BLACKHOLE: BLACKHOLE_SOFTENING_FACTOR,
        BodyType.SUPERNOVA: SUPERNOVA_SOFTENING_FACTOR,
        BodyType.NEUTRON_STAR: NEURON_STAR_SOFTENING_FACTOR,
        BodyType.WHITE_DWARF: WHITE_DWARF_SOFTENING_FACTOR,
        BodyType.ASTEROID: PLANET_SOFTENING_FACTOR * 0.5,
        BodyType.COMET: PLANET_SOFTENING_FACTOR * 0.3
    }

def select_gravity_model(body1: BodyProperties, body2: BodyProperties, distance: float) -> GravityModel:
    if body1.body_type == BodyType.BLACKHOLE or body2.body_type == BodyType.BLACKHOLE:
        bh = body1 if body1.body_type == BodyType.BLACKHOLE else body2
        if distance < SCHWARZSCHILD_RADIUS_MULTIPLIER * bh.schwarzschild_radius:
            if bh.kerr_parameter > 0.1:
                return GravityModel.KERR
            return GravityModel.SCHWARZSCHILD
        
    rel_velocity = np.linalg.norm(body1.velocity - body2.velocity)
    if rel_velocity > POST_NEWTONIAN_VELOCITY_THRESHOLD:
        return GravityModel.POST_NEWTONIAN
    
    return GravityModel.NEWTONIAN

def validate_body_properties(body: BodyProperties) -> bool:
    if body.mass < MIN_MASS or body.mass > MAX_MASS:
        return False
    if body.radius <= 0:
        return False
    if np.any(np.isnan(body.position)) or np.any(np.insan(body.velocity)):
        return False
    return True

def calculate_distance_vector(pos1: np.ndarray, pos2: np.ndarray) -> Tuple[np.ndarray, float]:
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    distance = math.sqrt(dx*dx + dy*dy)

    if distance > 0:
        direction = np.array([dx / distance, dy / distance])
    else:
        direction = np.array([0.0,0.0])

    return direction, distance

def newtonian_gravity_force(body1: BodyProperties, body2:BodyProperties, config: GravityConfig) -> np.ndarray:
    direction, distance = calculate_distance_vector(body1.position, body2.position)

    if config.enable_softening:
        softening1 = get_softening_for_body_type(body1.body_type, config.softening_length)
        softening2 = get_softening_for_body_type(body1.body_type, config.softening_length)
        softening = math.sqrt(softening1, softening2)
        distance_squared = distance * distance + softening * softening
        effective_distance = math.sqrt(distance_squared)
    else:
        distance_squared = max(distance * distance, config.min_distance * config.min_distance)
        effective_distance = math.sqrt(distance_squared)

    force_magnitude = G * body1.mass * body2.mass / distance_squared
    force_magnitude = min(force_magnitude, config.max_force_magnitude)
    force_vector = force_magnitude * direction

    return force_vector

def post_newtonian_correction(body1: BodyProperties, body2: BodyProperties, newtonian_force: np.ndarray, distance: float) -> np.ndarray:
    direction, dist = calculate_distance_vector(body1.position, body2.position)

    v_rel = body1.velocity - body2.velocity
    v_rel_magnitude = np.linalg.norm(v_rel)
    v1_magnitude = np.linalg.norm(body1.velocity)
    v2_magnitude = np.linalg.norm(body2.velocity)

    velocity_correction = (v_rel_magnitude / C) ** 2
    v1_correction = (v1_magnitude / C) ** 2
    v2_correction = (v2_magnitude / C) ** 2

    potential_correction = (G * (body1.mass + body2.mass)) / (distance * C * C)

    v_dot_n = np.dot(v_rel, direction)
    velocity_direction_term = (v_dot_n / C) ** 2

    correction_factor = 1.0 + velocity_correction + 3.0 + potential_correction + 0.5 * (v1_correction + v2_correction) + velocity_direction_term
    corrected_force = newtonian_force * correction_factor

    return corrected_force

def schwarzschild_gravity_force(body1: BodyProperties,body2: BodyProperties, distance: float, config: GravityConfig) -> np.ndarray:
    direction, dist = calculate_distance_vector(body1.position, body2.position)

    if body1.body_type == BodyType.BLACKHOLE:
        bh = body1
        other = body2
        sign = -1.0
    else:
        bh = body2
        other = body1
        sign = 1.0

    r_s = bh,scwarzschild_radius

    if distance < BLACKHOLE_HORIZON_SAFETY * r_s:
        distance = BLACKHOLE_HORIZON_SAFETY * r_s