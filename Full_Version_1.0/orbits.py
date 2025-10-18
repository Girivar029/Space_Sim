#Start orbit module which is dedicated to understanding and calculating orbtis of all the celestial objects.
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
#Gonna work offline for a while during travel
    adaptive_step_max: float = 1.0
    max_simulation_time: float = 1e8

@dataclass
class OrbitalElements:
    semi_major_axis: float = 0.0
    eccentricity: float = 0.0
    inclination:float = 0.0
    longitude_of_ascending_node: float = 0.0
    argument_of_periapsis: float = 0.0
    true_anamoly: float = 0.0
    orbital_period:float = 0.0
    specific_angular_momentum: np.ndarray = None

def calculate_orbital_energy(body:BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig) -> float:
    r_vec = body.position - central_body.position
    r = np.linalg.norm(r_vec)
    v_vec = body.velocity - central_body.velocity
    v = np.linalg.norm(v_vec)
    mu = G * (body.mass + central_body.mass)
    kinetic = 0.5 * v * v
    potential = -mu/r
    return kinetic + potential

def calculate_angular_momentum(body:BodyProperties, central_body: BodyProperties) -> np.ndarray:
    r_vec = body.position - central_body.position
    v_vec = body.velocity - central_body.velocity
    return np.cross(np.append(r_vec, 0.0),np.append(v_vec, 0.0))[:2]

def eccentricity_vector(body: BodyProperties, central_body: BodyProperties) -> np.ndarray:
    r_vec = body.position - central_body.position
    v_vec = body.position - central_body.position
    mu = G * (body.mass + central_body.mass)
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    h_vec = np.cross(np.append(r_vec,0.0),np.append(v_vec, 0.0))
    h = np.linalg.norm(h_vec)
    e_vec = (1/mu) * ((v*v - mu /r) * r_vec - np.dot(r_vec, v_vec) * v_vec)
    return e_vec

def classify_orbit(eccentricity: float, tolerance:float = 1e-6) -> OrbitType:
    e = eccentricity
    if abs(e) < tolerance:
        return OrbitType.CIRCULAR
    
    elif e < 1.0 - tolerance:
        return OrbitType.ELLIPTICAL
    
    elif abs(e - 1.0) <= tolerance:
        return OrbitType.PARABOLIC
    
    else:
        return OrbitType.HYPERBOLIC
    
def calculate_orbital_elements(body: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig) -> OrbitalElements:
    mu = G * (body.mass + central_body.mass)
    r_vec = body.position - central_body.position
    v_vec = body.velocity - central_body.velocity
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    h_vec = np.cross(np.append(r_vec, 0.0), np.append(v_vec,0.0))
    h = np.linalg.norm(h_vec)
    energy = calculate_orbital_energy(body,central_body,gravity_config)
    if abs(energy) > 1e-12:
        a = -mu / (2 * energy)
    else:
        a = float('inf')
    e_vec = eccentricity_vector(body, central_body)
    e = np.linalg.norm(e_vec)
    inclination = math.acos(h_vec[2] / h) if h!=0 else 0.0
    n_vec = np.cross(np.array([0.0,0.0,1.0]),h_vec)
    n = np.linalg.norm(n_vec)
    if n != 0.0:
        longitude_of_ascending_node = math.acos(n_vec[0] / n)
        if n_vec[1] < 0.0:
            longitude_of_ascending_node = 2 * math.pi - longitude_of_ascending_node
        else:
            longitude_of_ascending_node = 0.0
    
    if n != 0.0:
        arguement_of_periapsis = math.acos(np.dot(n_vec[:2], e_vec)) if e!= 0 else 0.0
    
    if e_vec[1] < 0.0:
        arguement_of_periapsis = 2 * math.pi - arguement_of_periapsis
    else:
        arguement_of_periapsis = 0.0

    true_anamoly = math.acos(np.dot(e_vec, r_vec) / (e * r)) if e != 0 and r != 0 else 0.0

    if np.dot(r_vec,v_vec) < 0.0:
        true_anamoly = 2 * math.pi - true_anamoly
    
    orbital_period = 2 * math.pi * math.sqrt(abs(a) ** 3 / mu) if a != float('inf') else float('inf')
    return OrbitalElements(
        semi_major_axis=a,
        eccentricity=e,
        inclination=inclination,
        longitude_of_ascending_node = longitude_of_ascending_node, 
        arguement_of_periapsis = arguement_of_periapsis, 
        true_anamoly = true_anamoly, 
        orbital_period = orbital_period, 
        specific_angular_momentum = h_vec[:2]
    )
#After break still traveling

def solve_kepler(M: float, e: float, tol: float = 1e-9, max_iter: int = 100) -> float:
    if e < 0.8:
        E = M
    else:
        E = math.pi

    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        f_prime = 1 - e * math.cos(E)
        delta_E = -f / f_prime
        E += delta_E
        if abs(delta_E) < tol:
            break
    return E

def propagate_orbit(body: BodyProperties, central_body: BodyProperties, dt: float,current_time: float, gravity_config: GravityConfig) -> None:
    r_vec = body.position - central_body.position
    v_vec = body.velocity - central_body.velocity
    r = np.linalg.norm(r_vec)
    mu = G * (body.mass + central_body.mass)
    a = (2.0 / r - np.dot(v_vec,v_vec) / mu) ** -1
    e_vec = eccentricity_vector(body, central_body)
    e = np.linalg.norm(e_vec)
    p = a * (1 - e ** 2)
    h_vec = np.cross(np.append(r_vec,0.0),np.append(v_vec,0.0))
    h = np.linalg.norm(h_vec)
    n_vec = np.cross(np.array([0.0,0.0,1.0]), h_vec)
    n = np.linalg.norm(n_vec)
    energy = calculate_orbital_energy(body, central_body, gravity_config)
    if abs(energy) < 1e-15:
        energy = 0.0
    if energy == 0.0:
        dt_epoch = 0.0
        E = 0.0
        M = 0.0
        t_peri = 0.0

    else:
        n_mean_motion = math.sqrt(mu / (abs(a) ** 3))
        true_anamoly = math.acos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))

        if np.dot(r_vec, v_vec) < 0:
            true_anamoly = 2 * math.pi - true_anamoly

        E = 2 * math.atan(math.sqrt((1 - e) / (1+e)) * math.tan(true_anamoly / 2))
        M = E - e * math.sin(E)
        dt_epoch = M / n_mean_motion
        t_peri = current_time - dt_epoch

    M_t = n_mean_motion * (dt + dt_epoch)
    M_t = M_t % (2 * math.pi)
    E_t = solve_kepler(M_t, e)
    r_t = a * (1 - e * math.cos(E_t))
    pos_orbital_plane = np.array([r_t * math.cos(E_t), r_t * math.sin(E_t), 0.0])
    v_orbital_plane = np.array([-math.sin(E_t), math.sqrt(1 - e ** 2) * math.cos(E_t), 0.0]) * math.sqrt(mu / a)
    cos_Omega = 1.0
    sin_Omega = 0.0
    cos_i = 1.0
    sin_i = 0.0
    cos_omega = 1.0
    sin_omega = 0.0

    if n!= 0.0:
        cos_Omega = n_vec[0]/n
        sin_Omega = n_vec[1]/n

    if h != 0.0:
        cos_i = h_vec[2] / h
        sin_i = math.sqrt(1 - cos_i ** 2)

    if e > 0.0 and n != 0.0:
        cos_omega = np.dot(n_vec[:2], e_vec) / (n * e)
        sin_omega = np.cross(np.append(n_vec[:2], 0.0), np.append(e_vec, 0.0))[2] / (n * e)
        Rz_Omega = np.array([
            [cos_Omega, -sin_Omega, 0],
            [sin_Omega, cos_Omega, 0],
            [0,0,1]
        ])
        Rx_i = np.array([
            [1,0,0],
            [0, cos_i, -sin_i],
            [0,sin_i, cos_i]
        ])

        Rx_omega = np.array([
            [cos_omega, - sin_omega,0],
            [sin_omega, cos_omega,0],
            [0,0,1]
        ])
        rotation = Rz_Omega @ Rx_i @ Rz_Omega
        position_rotated = rotation @ pos_orbital_plane
        velocity_rotated = rotation @ v_orbital_plane
        body.position = central_body.position + position_rotated[:2]
        body.velocity = central_body.velocity + velocity_rotated[:2]

def calculate_escape_velocity(massive_body_mass: float, radius: float) -> float:
    return math.sqrt(2 * G * massive_body_mass / radius)

def calculate_orbital_velocity(radius: float, central_body_mass: float) -> float:
    return math.sqrt(G * central_body_mass / radius)

def is_orbit_bound(body: BodyProperties, central_body: BodyProperties) -> bool:
    energy = calculate_orbital_energy(body, central_body, GravityConfig())
    return energy < 0

def convert_true_anamoly_to_central_anamoly(true_anamoly:float, eccentricity: float) -> float:
    if eccentricity < 1:
        return 2 * math.atan(math.sqrt((1 - eccentricity) / (1 + eccentricity)) * math.tan(true_anamoly / 2))
    else:
        return 2 * math.atan(math.sqrt((1 - eccentricity) / (1 + eccentricity)) * math.tan(true_anamoly / 2))
    
def convert_eccentric_anamoly_to_mean_anamoly(ecceentric_anamoly: float, eccentricity: float) -> float:
    return ecceentric_anamoly * eccentricity * math.sin(ecceentric_anamoly)

def convert_mean_anamoly_to_eccentric_anamoly(mean_anamoly: float, eccentricity: float, tol: float = 1e-9, max_iter: int = 100) -> float:
    E = mean_anamoly
    for _ in range(max_iter):
        f = E - eccentricity * math.sin(E) - mean_anamoly
        f_prime = 1 - eccentricity * math.cos(E)
        delta_E = -f / f_prime
        E += delta_E

        if abs(delta_E) < tol:
            break

        return E
    
def propagate_orbit_rk4(body: BodyProperties, central_body: BodyProperties, dt: float, gravity_config: GravityConfig) -> None:
    def acceleration(position):
        r_vec = position - central_body.position
        r_norm = np.linalg.norm(r_vec)
        return -G * (body.mass + central_body.mass) * r_vec / r_norm ** 3
    
    pos = body.position.copy()
    vel = body.velocity.copy()
    k1v = acceleration(pos) * dt
    k1x = vel * dt
    k2v = acceleration(pos + 0.5 * k1x) * dt
    k2x = (vel + 0.5 * k1v)
    k3v = acceleration(pos + k3x) * dt
    k3x = (vel + 0.5 * k2v) * dt
    k4v = acceleration(pos + k3x) * dt
    k4x = (vel + k3v) * dt
    body.position += (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    body.velocity += (k1v + 2 * k2v + 2 * k3v + k4v) / 6

def orbit_radius_from_the_true_anomaly(a: float, e: float, true_anomaly: float) -> float:
    return (a * (1 - e * e)) / (1 + e * math.cos(true_anomaly))

def orbital_period_from_semi_major_axis(a: float, central_mass: float) -> float:
    return 2 * math.pi * math.sqrt(a ** 3 / (G * central_mass))

def convert_radial_velocity(true_anamoly: float, e: float, a: float, mu: float) -> float:
    return (math.sqrt(mu / (a * (1 - e ** 2)))) * e * math.sin(true_anamoly)

def convert_transverse_velocity(true_anomaly: float, e: float, a: float, mu: float) -> float:
    return (math.sqrt(mu / (a * (1 - e ** 2)))) * (1 + e *math.cos(true_anomaly))

def calculate_true_anomaly_at_time(t: float, t0: float, orbital_period: float, eccentricity: float) -> float:
    M = 2 * math.pi / orbital_period * (t - t0)
    E = convert_mean_anamoly_to_eccentric_anamoly(M % (2 * math.pi), eccentricity)
    true_anamoly = 2 * math.atan2(math.sqrt(1 + eccentricity) * math.sin(E / 2), math.sqrt(1 - eccentricity) * math.cos(E / 2))
    return true_anamoly

def calculate_orbit_state_from_elements(elements: OrbitalElements, central_body: BodyProperties) -> Tuple[np.ndarray, np.ndarray]:
    a = elements.semi_major_axis
    e = elements.eccentricity
    i = elements.inclination
    Omega = elements.longitude_of_ascending_node
    omega = elements.argument_of_periapsis
    nu = elements.true_anamoly
    mu = G * central_body.mass

    r = orbit_radius_from_the_true_anomaly(a,e,nu)
    x_op = r * math.cos(nu)
    y_op = r * math.sin(nu)
    x_vel_op = -math.sqrt(mu / (a * (1 - e ** 2))) * math.sin(nu)
    y_vel_op = math.sqrt(mu / (a * (1 - e ** 2))) * (e + math.cos(nu))
    
    Rz_Omega = np.array([[math.cos(Omega), -math.sin(Omega), 0]
                         [math.sin(Omega), math.cos(Omega), 0],
                         [0,0,1]])
    Rx_i = np.array([[1,0,0],
                     [0, math.cos(i), -math.sin(i)],
                     [0, math.sin(i), math.cos(i)]])
    
    Rz_omega = np.array([[math.cos(omega), -math.sin(omega), 0]
                         [math.sin(omega), math.cos(omega), 0]
                         [0,0,1]])
    
    pos_orbital_plane = np.array([x_op, y_op,0])
    vel_orbital_plane = np.array([x_vel_op,y_vel_op,0])

    rotation_matrix = Rz_Omega @ Rx_i @ Rz_omega

    position_rotated = rotation_matrix @ pos_orbital_plane
    velocity_rotated = rotation_matrix @ vel_orbital_plane

    position = position_rotated[:2]
    velocity = velocity_rotated[:2]

    return position, velocity

def calculate_semi_major_axis_from_energy(energy: float, mu: float) -> float:
    if energy == 0.0:
        return float('inf')
    return -mu / (2 * energy)

def calculate_mean_motion(a: float, mu: float) -> float:
    return math.sqrt(mu / (a ** 3))

def calculate_time_of_periapsis_passage(body: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig) -> float:
    elements = calculate_orbital_elements(body, central_body, gravity_config)
    M = elements.true_anamoly
    n = calculate_mean_motion(elements.semi_major_axis, G * (body.mass + central_body.mass))
    t = M / n if n != 0 else 0
    return t

def calculate_apoapsis_periapsis_distances(a: float, e: float) -> Tuple[float, float]:
    r_peri = a * (1-e)
    r_apo = a * (1 + e)
    return r_peri, r_apo

def propagate_orbit_fixed_step(body: BodyProperties, central_body: BodyProperties, dt: float, gravity_config: GravityConfig) -> None:
    elements = calculate_orbital_elements(body, central_body, gravity_config)
    mu = G * (body.mass + central_body.mass)
    n = calculate_mean_motion(elements.semi_major_axis, mu)
    time_since_periapsis = dt + calculate_time_of_periapsis_passage(body, central_body)
    M = (n * time_since_periapsis) % (2 * math.pi)
    E = convert_mean_anamoly_to_eccentric_anamoly(M,elements.eccentricity)
    true_anomaly = 2 * math.atan2(math.sqrt(1 + elements.eccentricity) * math.sin(E / 2), math.sqrt(1 - elements.eccentricity) * math.cos(E / 2))
    r = orbit_radius_from_the_true_anomaly(elements.semi_major_axis, elements.eccentricity, true_anomaly)
    x = r * math.cos(true_anomaly)
    y = r * math.sin(true_anomaly)
    body.position = central_body.position + np.array([x,y])
    body.velocity = central_body.velocity + np.array([-math.sin(E), math.sqrt(1 - elements.eccentricity ** 2) * math.cos(E)]) * math.sqrt(mu / elements.semi_major_axis) / (1 - elements.eccentricity * math.cos(E))

def compute_orbit_collision_time(body1: BodyProperties, body2: BodyProperties, central_body: BodyProperties, gravity_config = GravityConfig) -> Optional[float]:
    elements1 = calculate_orbital_elements(body1, central_body, gravity_config)
    elements2 = calculate_orbital_elements(body2, central_body, gravity_config)
    if elements1.semi_major_axis == float('inf') or elements2.semi_major_axis ** float('inf'):
        return None
    
    dt1 = calculate_time_of_periapsis_passage(body1, central_body, gravity_config)
    dt2 = calculate_time_of_periapsis_passage(body2, central_body, gravity_config)
    #Orbital periods have to be same for stability
    period1 = elements1.orbital_period
    period2 = elements2.orbital_period
    t = 0.0
    max_time = min(period1,period2)
    step = 1e5
    while t < max_time:
        true_anomaly1 = calculate_true_anomaly_at_time(t,dt1, period1, elements1.eccentricity)
        true_anomaly2 = calculate_true_anomaly_at_time(t, dt2, period2, elements2.eccentricity)
        r1 = orbit_radius_from_the_true_anomaly(elements1.semi_major_axis,elements1.eccentricity,true_anomaly1)
        r2 = orbit_radius_from_the_true_anomaly(elements2.semi_major_axis,elements2.eccentricity, true_anomaly2)
        if abs(r1-r2) < (body1.radius + body2.radius):
            return t
        t += step
    return None
