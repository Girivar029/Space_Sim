#The goal of this piece of code is to simulate the best possible physics without putting a lot of strain on the computer.
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

FRAME_DRAG_COEFFICIENT = 0.1
SPIN_ANGULAR_MOMENTUM_FACTOR = 0.4
LENSE_THIRRING_FACTOR = 2.0
KERR_PARAMETER_MAX = 0.998

GW_LUMINOSITY_COEFFICIENT = 32.0 / 5.0
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
#Break1(after a long time of coding)

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

    def __post_init__(self):
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
    if np.any(np.isnan(body.position)) or np.any(np.isnan(body.velocity)):
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

    r_s = bh.schwarzschild_radius


#break 2 - starting day 5
    if distance < BLACKHOLE_HORIZON_SAFETY * r_s:
        distance = BLACKHOLE_HORIZON_SAFETY * r_s

    newtonian_term = G * bh.mass * other.mass / (distance * distance)

    relativistic_factor = 1.0 / (1.0 - r_s / distance)

    if distance < PHOTON_SPHERE_MULTIPLIER * r_s:
        instability_factor = 1.0 + math.exp(-(distance - PHOTON_SPHERE_MULTIPLIER * r_s) / r_s)
        relativistic_factor *= instability_factor

    if distance < ISCO_MULTIPLIER * r_s:
        isco_correction = 1.0 + 0.5 * (ISCO_MULTIPLIER * r_s / distance - 1.0)
        relativistic_factor *= isco_correction

    force_magnitude = newtonian_term * relativistic_factor
    force_magnitude = min(force_magnitude, config.max_force_magnitude)
    force_vector = sign * force_magnitude * direction

    return force_vector

def calculate_tidal_force(body1: BodyProperties, body2: BodyProperties, distance: float) -> Tuple[np.ndarray, float]:
    direction, dist = calculate_distance_vector(body1.position, body2.position)

    if not body1.is_extended or body1.radius == 0:
        return np.array([0.0,0.0]),0.0
    
    if distance > TIDAL_FORCE_CUTOFF * body1.radius:
        return np.array([0.0,0.0]),0.0
    
    tidal_coefficient = 2.0 * G * body2.mass * body1.radius / (distance ** 3)
    tidal_stress = tidal_coefficient * body1.mass

    perpendicular = np.array([-direction[1], direction[0]])
    tidal_force = tidal_stress * perpendicular

    return tidal_force, tidal_stress

def calculate_roche_limit(body1: BodyProperties, body2:BodyProperties) -> float:
    if body1.density > 0 and body2.density > 0:
        if body1.is_extended:
            roche_limit = body2.radius * FLUID_ROCHE_FACTOR * (body2.density / body1.density) ** (1.0/3.0)
        else:
            roche_limit = body2.radius * RIGID_ROCHE_FACTOR * (body2.density / body1.density) ** (1.0*3.0)
    else:
        roche_limit = body2.radius * ROCHE_LIMIT_FACTOR * (body2.mass / body1.mass) ** (1.0*3.0)

    return roche_limit

def check_tidal_disruption(body1: BodyProperties, body2: BodyProperties, distance: float) -> Tuple[bool, float]:
    roche_limit = calculate_roche_limit(body1,body2)
    disruption_ratio = distance / roche_limit

    is_disrupted = disruption_ratio < TIDAL_DISRUPTION_THRESHOLD

    return is_disrupted, disruption_ratio

def calculate_tidal_heating(body1:BodyProperties, body2: BodyProperties, distance: float, orbital_eccentricity: float) -> float:
    if not body1.is_extended:
        return 0.0
    
    tidal_force, tidal_stress = calculate_tidal_force(body1,body2,distance)

    love_number = body1.tidal_love_number
    dissipation_factor = 0.01

    tidal_heating_power = love_number * dissipation_factor * (tidal_stress ** 2) * body1.radius ** 3 / body1.mass

    eccentricity_factor = 1.0 + 7.5 * orbital_eccentricity ** 2
    tidal_heating_power *= eccentricity_factor

    return tidal_heating_power

def extended_body_gravity_correction(body1: BodyProperties, body2: BodyProperties, distance: float, newtonian_force: np.ndarray) -> np.ndarray:
    if not body2.is_extended:
        return newtonian_force
    
    if distance < body2.radius:
        interior_mass_fraction = (distance / body2.radius) ** 3
        corrected_force = newtonian_force * interior_mass_fraction
        return corrected_force
    
    quadrupple_correction = (body2.radius / distance) ** 2
    j2_coeffecient = 0.01

    direction, dist = calculate_distance_vector(body1.position, body2.position)

    quadrupple_term = j2_coeffecient * quadrupple_correction * newtonian_force

    return newtonian_force + quadrupple_term

def multi_body_perturbation(primary_body: BodyProperties, secondary_body: BodyProperties, pertubing_bodies: List[BodyProperties], distance: float) -> np.ndarray:
    total_pertubation = np.array([0.0,0.0])

    for perturber in pertubing_bodies:
        if perturber is primary_body or perturber is secondary_body:
            continue

        dist_to_secondary = np.linalg.norm(secondary_body.position - perturber.position)
        dist_to_primary = np.linalg.norm(primary_body.position - perturber.position)


        if dist_to_secondary < distance * 0.1:
            direction_to_secondary, d = calculate_distance_vector(perturber.position, secondary_body.position)
            perturbation_magnitude = G * perturber.mass / (dist_to_secondary ** 2)
            total_pertubation += perturbation_magnitude * direction_to_secondary

    return total_pertubation

def calculate_gravitational_potential(body: BodyProperties, position: np.ndarray, all_bodies: List[BodyProperties]) -> float:
    total_potential = 0.0

    for other_body in all_bodies:
        if other_body is body:
            continue

        direction, distance = calculate_distance_vector(position, other_body.position)

        if distance > 0:
            potential = -G * other_body.mass / distance
            total_potential += distance

    return total_potential

def calculate_escape_velocity(body: BodyProperties, distance: float) -> float:
    escape_vel = math.sqrt(2.0 * G * body.mass / distance)
    return escape_vel

def calculate_orbital_velocity(central_body: BodyProperties, distance:float) -> float:
    orbital_vel = math.sqrt(G * central_body.mass / distance)
    return orbital_vel

def calculate_hill_sphere_radius(body: BodyProperties, central_body: BodyProperties, orbital_distance: float) -> float:
    mass_ratio = body.mass / central_body.mass
    hill_radius = orbital_distance * (mass_ratio / 3.0) ** (1.0/3.0)
    return hill_radius

def check_sphere_of_influence(body1: BodyProperties, body2: BodyProperties, distance: float, central_body: BodyProperties, orbital_distance: float) -> bool:
    if body2 is central_body:
        return False
    
    hill_radius = calculate_hill_sphere_radius(body2, central_body, orbital_distance)
    return distance < hill_radius

def calculate_gravity_gradient(body: BodyProperties, position: np.ndarray, all_bodies: List[BodyProperties]) -> np.ndarray:
    delta = 1e6
    gradient = np.array([0.0,0.0])

    pos_x_plus = position + np.array([delta, 0.0])
    pos_x_minus = position - np.array([delta,0.0])

    potential_x_plus = calculate_gravitational_potential(body, pos_x_plus, all_bodies)
    potential_x_minus = calculate_gravitational_potential(body, pos_x_minus, all_bodies)

    gradient[0] = (potential_x_plus - potential_x_minus) / (2.0 * delta)

    pos_y_plus = position + np.array([delta, 0.0])
    pos_y_minus = position + np.array([delta, 0.0])

    potential_y_plus = calculate_gravitational_potential(body, pos_y_plus, all_bodies)
    potential_y_minus = calculate_gravitational_potential(body, pos_y_minus, all_bodies)

    gradient[1] = (potential_y_plus - potential_y_minus) / (2.0 * delta)

    return gradient

def calculate_frame_dragging_effect(body1: BodyProperties, body2: BodyProperties, distance:float) -> np.ndarray:
    if body2.body_type not in [BodyType.BLACKHOLE, BodyType.NEUTRON_STAR]:
        return np.array[[0.0,0.0]]
    
    if body2.spin_angular_momentum == 0:
        return np.array([0.0,0.0])
    
    direction, dist = calculate_distance_vector(body1.position, body2.position)

    omega_lt = (LENSE_THIRRING_FACTOR * G * body2.spin_angular_momentum) / (C * C * distance ** 3)

    perpendicular = np.array([-direction[1], direction[0]])

    spin_axis_projection = np.dot(body2.spin_axis[:2], perpendicular)

    frame_drag_force = body1.mass * omega_lt * FRAME_DRAG_COEFFICIENT * perpendicular * spin_axis_projection

    return frame_drag_force

def calculate_kerr_metric_correction(body1: BodyProperties, body2: BodyProperties, distance: float) -> np.ndarray:
    if body2.body_type != BodyType.BLACKHOLE or body2.kerr_parameter == 0:
        return np.array([0.0,0.0])
    
    direction, dist = calculate_distance_vector(body1.position, body2.position)

    r_s = body2.schwarzschild_radius
    a = body2.kerr_parameter * r_s

    r_plus = 0.5 * r_s * (1.0 + math.sqrt(1.0 - body2.kerr_parameter ** 2))

    if distance < 1.2 * r_plus:
        distance = 1.2 * r_plus

    ergosphere_factor = 1.0
    if distance < 2.0 * r_s:
        ergosphere_factor = 1.0 + (a / distance) ** 2

    kerr_correction = ergosphere_factor * (1.0 + body2.kerr_parameter ** 2 * r_s ** 2 / (4.0 * distance ** 2))

    base_force = G * body1.mass * body2.mass / (distance * distance)
    corrected_force = base_force * kerr_correction

    return corrected_force * direction

def calculate_gravitational_wave_energy_loss(body1: BodyProperties, body2: BodyProperties, distance: float, orbital_velocity: float) -> np.ndarray:
    if distance > 1e11:
        return 0.0
    
    mu = (body1.mass * body2.mass) / (body1.mass + body2.mass)
    M = body1.mass + body2.mass

    omega = orbital_velocity / distance
    gw_luminosity = GW_LUMINOSITY_COEFFICIENT * (G ** 4  / C ** 5) * (mu ** 2) * (M ** 3) / (distance ** 5)
    return gw_luminosity

def calculate_gravitational_wave_recoil(body1: BodyProperties, body2:BodyProperties, distance: float, orbital_velocity: float) -> np.ndarray:
    energy_loss = calculate_gravitational_wave_energy_loss(body1, body2, distance, orbital_velocity)

    if energy_loss == 0:
        return np.array([0.0,0.0])
    
    rel_velocity = body1.velocity - body2.velocity
    if np.linalg.norm(rel_velocity) > 0:
        recoil_direction = -rel_velocity / np.linalg.norm(rel_velocity)
    else:
        return np.array([0.0,0.0])
    
    recoil_magnitude = GW_ENERGY_LOSS_FACTOR * energy_loss / C
    recoil_vector = recoil_magnitude * recoil_direction
    return recoil_vector

def calculate_chirp_mass(body1: BodyProperties, body2: BodyProperties):
    m1 = body1.mass
    m2 = body2.mass
    chirp_mass = ((m1 * m2) ** ( 3.0 / 5.0) / ((m1+m2) ** (1.0/5.0)))
    return chirp_mass

def calculate_merger_time(body1: BodyProperties,body2: BodyProperties, distance: float, eccentricity: float) -> float:
    chirp_mass = calculate_chirp_mass(body1,body2)

    beta = (64.0 / 5.0) * (G **3) * chirp_mass ** 3 / (C ** 5)

    if eccentricity < 0.01:
        merger_time = (distance ** 4) / (4.0 * beta)
    else:
        c0 = distance * (1.0 - eccentricity ** 2)
        e0 = eccentricity
        merger_time = (c0 ** 4 / beta) * (1.0 / (1.0 - e0 ** 2)) ** (7.0/2.0)
    return merger_time

def supernova_remnant_gravity(body1: BodyProperties, body2: BodyProperties, distance: float, time_since_explosion: float) -> np.ndarray:
    direction, dist = calculate_distance_vector(body1.position,body2.position)

    if body2.body_type != BodyType.SUPERNOVA:
        return newtonian_gravity_force(body1, body2, GravityConfig())
    
    core_mass = body2.mass * SUPERNOVA_CORE_FRACTION
    envelope_mass = body2.mass * core_mass / (distance * distance)

    expansion_radius = SUPERNOVA_EXPANSION_VELOCITY * time_since_explosion

    core_force = G * body1.mass * core_mass / (distance * distance)

    if distance > expansion_radius:
        envelope_contribution = envelope_mass
    else:
        shell_mass_fraction = (distance / expansion_radius) ** 3
        envelope_contribution = envelope_mass * shell_mass_fraction

    envelope_force = G * body1.mass * envelope_contribution / (distance * distance)
    total_force_magnitude = core_force + envelope_force
    total_force_magnitude = min(total_force_magnitude, MAX_FORCE_MAGNITUDE)

    return total_force_magnitude * direction

def calculate_radiation_pressure_force(body1: BodyProperties, body2: BodyProperties, distance: float) -> np.ndarray:
    if body2.luminosity == 0:
        return np.array([0.0,0.0])
    
    direction, dist = calculate_distance_vector(body1.position, body2.position)
    cross_section = math.pi * body1.radius ** 2
    radiation_intensity = body2.luminosity / (4.0 * math.pi * distance ** 2)
    radiation_force_magnitude = (radiation_intensity * cross_section) / C
    return radiation_force_magnitude * direction

def white_dwarf_gravity_correction(body1: BodyProperties, body2: BodyProperties, distance:float) -> np.ndarray:
    if body2.body_type != BodyType.WHITE_DWARF:
        return newtonian_gravity_force(body1, body2, GravityConfig())
    
    direction, dist = calculate_distance_vector(body1.position, body2.position)

    if body2.mass > CHANDRASEKHAR_LIMIT * SOLAR_MASS:
        instability_factor = 1.0 + (body2.mass - CHANDRASEKHAR_LIMIT * SOLAR_MASS) / (CHANDRASEKHAR_LIMIT * SOLAR_MASS)
    else:
        instability_factor = 1.0

    base_force = G * body1.mass * body2.mass / (distance * distance)

    return base_force * instability_factor * direction
#Day 2 Over - Hardly 2hrs and not very productive

def neutron_star_gravity_correction(body1: BodyProperties, body2: BodyProperties, distance: float) -> np.ndarray:
    if body2.body_type != BodyType.NEUTRON_STAR:
        return newtonian_gravity_force(body1, body2, GravityConfig())
    
    direction, dist = calculate_distance_vector(body1.position, body2.position)
    compactness = G * body2.mass / (body2.radius * C * C)
    relativistic_correction = 1.0 / (1.0 / 2.0 * compactness)

    base_force = G * body1.mass * body2.mass / (distance * distance)

    if distance < 3.0 * body2.radius:
        promximity_factor = 1.0 + 0.5 * (body2.radius / distance)
        relativistic_correction *= promximity_factor

    corrected_force = base_force * relativistic_correction

    return corrected_force * direction

def calculate_accretion_disk_torque(body1: BodyProperties, body2: BodyProperties, distance: float) -> np.ndarray:
    if body2.body_type not in [BodyType.BLACKHOLE, BodyType.NEUTRON_STAR, BodyType.WHITE_DWARF]:
        return np.array([0.0,0.0])
    if distance > 100.0 * body2.radius:
        return np.array([0.0,0.0])
    
    direction, dist = calculate_distance_vector(body1.position, body2.position)
    perpendicular = np.array([-direction[1], direction[0]])

    accertion_rate = BLACKHOLE_ACCERATION_EFFICIENCY * body2.mass / (1e8 * 365.25 * 24 *3600)

    angular_momentum_transfer = accertion_rate * distance * math.sqrt(G * body2.mass * distance)

    torque_force = angular_momentum_transfer / (body1.mass * distance)

    return torque_force * perpendicular

def calculate_eddington_limit(body: BodyProperties) -> float:
    eddington_luminosity = (4.0 * math.pi * G * body.mass * C) / (0.4 * 6.65e-29)
    return eddington_luminosity

def check_eddington_accretion(body: BodyProperties) -> float:
    if body.luminosity == 0:
        return False
    
    eddington_limit = calculate_eddington_limit(body)
    return body.luminosity > BLACKHOLE_EDDINGTON_FACTOR * eddington_limit

def calculate_dynamical_friction(body1: BodyProperties, background_density: float, velocity_dispersion: float) -> np.ndarray:
    if np.linalg.norm(body1.velocity) == 0:
        return np.array([0.0,0.0])
    
    velocity = np.linalg.norm(body1.velocity)

    coulomb_log = math.log(1.0 + (velocity / velocity_dispersion) ** 2)

    friction_coefficient = 4.0 * math.pi * G * G * body1.mass * background_density * coulomb_log / (velocity ** 3)

    friction_force_magnitude = friction_coefficient * body1.mass * velocity

    friction_direction = -body1.velocity / velocity

    return friction_force_magnitude * friction_direction

def three_body_lagrange_points(body1: BodyProperties, body2: BodyProperties, distance: float) -> List[np.ndarray]:
    direction, dist = calculate_distance_vector(body1.position, body2.position)
    perpendicular = np.array([-direction[1], direction[0]])

    mass_ratio = body2.mass / (body1.mass + body2.mass)
    barycenter = body1.position + mass_ratio * distance * direction

    l1_dirstance = direction * (mass_ratio / 3.0) ** (1.0/3.0)
    l1 = barycenter - l1_dirstance * direction

    l2_dirstance = direction * (mass_ratio / 3.0) ** (1.0/3.0)
    l2 = barycenter + l2_dirstance * direction

    l3_dirstance = direction * (mass_ratio / 3.0) ** (1.0/3.0)
    l3 = body1.position - l3_dirstance * direction

    l4_offset = distance * math.sqrt(3.0) / 2.0
    l4 = barycenter + l4_offset * perpendicular

    l5_offset = distance * math.sqrt(3.0) / 2.0
    l5 = barycenter - l5_offset * perpendicular

    return [l1,l2,l3,l4,l5]