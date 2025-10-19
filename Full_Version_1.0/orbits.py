#Start orbit module which is dedicated to understanding and calculating orbtis of all the celestial objects.
import math
import numpy as np
from gravity import BodyProperties, GravityConfig, GravityIntegrator, G, AU
from typing import List, Tuple, Dict, Optional
from enum import Enum
from dataclasses import dataclass

#adding missing functions
class BodyType(Enum):
    STAR = 1
    PLANET = 2
    BLACKHOLE = 3
    SUPERNOVA = 4
    NEUTRON_STAR = 5
    WHITE_DWARF = 6
    ASTEROID = 7
    COMET = 8

def calculate_orbit_transfer_delta_v(initial_body: BodyProperties, target_orbit_radius: float, central_body: BodyProperties) -> float:
    r1 = np.linalg.norm(initial_body.position - central_body.position)
    r2 = target_orbit_radius
    mu = G * central_body.mass
    v1 = math.sqrt(mu / r1)
    v_transfer1 = math.sqrt(mu * (2 / r1 - 1 / ((r1 + r2) / 2)))
    v_transfer2 = math.sqrt(mu * (2 / r2 - 1 / ((r1 + r2) / 2)))
    delta_v1 = abs(v_transfer1 - v1)
    delta_v2 = abs(math.sqrt(mu / r2) - v_transfer2)
    return delta_v1 + delta_v2
#done



class OrbitType(Enum):
    CIRCULAR = "circular"
    ELLIPTICAL = "elliptical"
    PARABOLIC = "parabolic"
    HYPERBOLIC = "hyperbolic"

@dataclass
class OrbitConfig:
    use_adaptive_step: bool
    adaptive_step_min: float
    adaptive_step_max: float
    eccentricity_tolerance: float = 1e-6
    max_orbit_iterations: int = 1000
    orbit_convergence_tolerance: float = 1e-10
    max_simulation_time: float = 1e8
#Working offline

@dataclass
class OrbitalElements:
    semi_major_axis: float = 0.0
    eccentricity: float = 0.0
    inclination:float = 0.0
    longitude_of_ascending_node: float = 0.0
    argument_of_periapsis: float = 0.0
    true_anomaly: float = 0.0
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
    h_vec = np.cross(np.append(r_vec, 0.0), np.append(v_vec, 0.0))
    h = np.linalg.norm(h_vec)
    energy = calculate_orbital_energy(body, central_body, gravity_config)
    if abs(energy) > 1e-12:
        a = -mu / (2 * energy)
    else:
        a = float('inf')
    e_vec = eccentricity_vector(body, central_body)
    e = np.linalg.norm(e_vec)
    inclination = math.acos(h_vec[2] / h) if h != 0 else 0.0
    n_vec = np.cross(np.array([0.0, 0.0, 1.0]), h_vec)
    n = np.linalg.norm(n_vec)
    if n != 0.0:
        longitude_of_ascending_node = math.acos(n_vec[0] / n)
        if n_vec[1] < 0.0:
            longitude_of_ascending_node = 2 * math.pi - longitude_of_ascending_node
    else:
        longitude_of_ascending_node = 0.0
    if n != 0.0 and e != 0:
        argument_of_periapsis = math.acos(np.dot(n_vec[:2], e_vec) / (n * e))
        if e_vec[1] < 0.0:
            argument_of_periapsis = 2 * math.pi - argument_of_periapsis
    else:
        argument_of_periapsis = 0.0
    if e != 0 and r != 0:
        true_anomaly = math.acos(np.dot(e_vec, r_vec) / (e * r))
    else:
        true_anomaly = 0.0
    if np.dot(r_vec, v_vec) < 0.0:
        true_anomaly = 2 * math.pi - true_anomaly
    orbital_period = 2 * math.pi * math.sqrt(abs(a) ** 3 / mu) if a != float('inf') else float('inf')
    return OrbitalElements(
        semi_major_axis=a,
        eccentricity=e,
        inclination=inclination,
        longitude_of_ascending_node=longitude_of_ascending_node,
        argument_of_periapsis=argument_of_periapsis,
        true_anomaly=true_anomaly,
        orbital_period=orbital_period,
        specific_angular_momentum=h_vec[:2]
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
        true_anomaly = math.acos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))

        if np.dot(r_vec, v_vec) < 0:
            true_anomaly = 2 * math.pi - true_anomaly

        E = 2 * math.atan(math.sqrt((1 - e) / (1+e)) * math.tan(true_anomaly / 2))
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

def convert_true_anomaly_to_central_anomaly(true_anomaly:float, eccentricity: float) -> float:
    if eccentricity < 1:
        return 2 * math.atan(math.sqrt((1 - eccentricity) / (1 + eccentricity)) * math.tan(true_anomaly / 2))
    else:
        return 2 * math.atan(math.sqrt((1 - eccentricity) / (1 + eccentricity)) * math.tan(true_anomaly / 2))
    
def convert_eccentric_anomaly_to_mean_anomaly(ecceentric_anomaly: float, eccentricity: float) -> float:
    return ecceentric_anomaly * eccentricity * math.sin(ecceentric_anomaly)

def convert_mean_anomaly_to_eccentric_anomaly(mean_anomaly: float, eccentricity: float, tol: float = 1e-9, max_iter: int = 100) -> float:
    E = mean_anomaly
    for _ in range(max_iter):
        f = E - eccentricity * math.sin(E) - mean_anomaly
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

def convert_radial_velocity(true_anomaly: float, e: float, a: float, mu: float) -> float:
    return (math.sqrt(mu / (a * (1 - e ** 2)))) * e * math.sin(true_anomaly)

def convert_transverse_velocity(true_anomaly: float, e: float, a: float, mu: float) -> float:
    return (math.sqrt(mu / (a * (1 - e ** 2)))) * (1 + e *math.cos(true_anomaly))

def calculate_true_anomaly_at_time(t: float, t0: float, orbital_period: float, eccentricity: float) -> float:
    M = 2 * math.pi / orbital_period * (t - t0)
    E = convert_mean_anomaly_to_eccentric_anomaly(M % (2 * math.pi), eccentricity)
    true_anomaly = 2 * math.atan2(math.sqrt(1 + eccentricity) * math.sin(E / 2), math.sqrt(1 - eccentricity) * math.cos(E / 2))
    return true_anomaly

def calculate_orbit_state_from_elements(elements: OrbitalElements, central_body: BodyProperties) -> Tuple[np.ndarray, np.ndarray]:
    a = elements.semi_major_axis
    e = elements.eccentricity
    i = elements.inclination
    Omega = elements.longitude_of_ascending_node
    omega = elements.argument_of_periapsis
    nu = elements.true_anomaly
    mu = G * central_body.mass
    r = orbit_radius_from_the_true_anomaly(a, e, nu)

    # Ensure denominator is positive and nonzero for sqrt
    denom = a * (1 - e ** 2)
    epsilon = 1e-12
    if denom <= epsilon or mu <= 0:
        vmag = 0.0
    else:
        vmag = math.sqrt(mu / denom)

    x_op = r * math.cos(nu)
    y_op = r * math.sin(nu)
    x_vel_op = -vmag * math.sin(nu)
    y_vel_op = vmag * (e + math.cos(nu))

    # Axis rotation (only if needed: you can comment this out for simple coplanar cases)
    Rz_Omega = np.array([
        [math.cos(Omega), -math.sin(Omega), 0],
        [math.sin(Omega),  math.cos(Omega), 0],
        [0, 0, 1]
    ])
    Rx_i = np.array([
        [1, 0, 0],
        [0, math.cos(i), -math.sin(i)],
        [0, math.sin(i), math.cos(i)]
    ])
    Rz_omega = np.array([
        [math.cos(omega), -math.sin(omega), 0],
        [math.sin(omega),  math.cos(omega), 0],
        [0, 0, 1]
    ])

    pos_orbital_plane = np.array([x_op, y_op, 0])
    vel_orbital_plane = np.array([x_vel_op, y_vel_op, 0])

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
    M = elements.true_anomaly
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
    time_since_periapsis = dt + calculate_time_of_periapsis_passage(body, central_body, gravity_config)
    M = (n * time_since_periapsis) % (2 * math.pi)
    E = convert_mean_anomaly_to_eccentric_anomaly(M, elements.eccentricity)

    e_clamped = min(max(elements.eccentricity, 0.0), 1.0)
    sqrt_term1 = math.sqrt(1 + e_clamped)
    sqrt_term2 = math.sqrt(max(0.0, 1 - e_clamped))

    true_anomaly = 2 * math.atan2(sqrt_term1 * math.sin(E / 2), sqrt_term2 * math.cos(E / 2))
    r = orbit_radius_from_the_true_anomaly(elements.semi_major_axis, e_clamped, true_anomaly)
    x = r * math.cos(true_anomaly)
    y = r * math.sin(true_anomaly)
    body.position = central_body.position + np.array([x, y])
    body.velocity = central_body.velocity + np.array([-math.sin(E), math.sqrt(max(0.0, 1 - e_clamped**2)) * math.cos(E)]) * math.sqrt(mu / elements.semi_major_axis) / (1 - elements.eccentricity * math.cos(E))

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

def generate_orbit_points(elements: OrbitalElements, central_body: BodyProperties, num_points: int = 360) -> np.ndarray:
    a = elements.semi_major_axis
    e = elements.eccentricity
    i = elements.inclination
    Omega = elements.longitude_of_ascending_node
    omega = elements.argument_of_periapsis
    mu = G * central_body.mass
    angles = np.linspace(0, 2 * math.pi, num_points, endpoint=False)
    orbit_points = []
    Rz_Omega = np.array([[math.cos(Omega), -math.sin(Omega),0],
                         [math.sin(Omega), math.cos(Omega),0],
                         [0,0,1]])
    Rx_i = np.array([[1,0,0],
                     [0, math.cos(i), - math.sin(i)],
                     [0, math.sin(i), math.cos(i)]])
    
    Rz_omega = np.array([[math.cos(omega), - math.sin(omega),0],
                         [math.sin(omega), math.cos(omega),0],
                         [0,0,1]])
    rotation_matrix = Rz_Omega @ Rx_i @ Rz_omega
    for nu in angles:
        r = orbit_radius_from_the_true_anomaly(a,e,nu)
        pos = np.array([r * math.cos(nu), r * math.sin(nu),0])
        pos_rotated = rotation_matrix @ pos
        orbit_points.append(central_body.position + pos_rotated[:2])
        return np.array(orbit_points)

def orbit_to_ephemeris(elements: OrbitalElements, central_body: BodyProperties, epoch: float, dt: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    points = []
    velocities = []
    mu = G * central_body.mass
    for k in range(n_steps):
        t = epoch + k * dt
        period = orbital_period_from_semi_major_axis(elements.semi_major_axis,central_body.mass)
        nu = calculate_true_anomaly_at_time(t, 0, period, elements.eccentricity, nu)
        pos, vel = calculate_orbit_state_from_elements(elements, central_body)
        points.append(pos)
        velocities.append(vel)
    return np.array(points), np.array(velocities)

def find_orbital_resonance(body1: BodyProperties, body2: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig) -> Optional[float]:
    elements1 = calculate_orbital_elements(body1,central_body,gravity_config)
    elements2 = calculate_orbital_elements(body2, central_body, gravity_config)
    period1 = elements1.orbital_period
    period2 = elements2.orbital_period
    if period1 == float('inf') or period2 == float('inf'):
        return None
    
    ratio = period1 / period2
    best_match = None
    best_error = float('inf')
    for p in range(1,10):
        for q in range(1,10):
            if q == 0:
                continue

            error = abs((p/q) - ratio)
            if error < best_error:
                best_error = error
                best_match = (p,q)
    return best_match

def compute_orbit_intersection(body1: BodyProperties, body2: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig, tol = 1e-2) -> Optional[Tuple[np.ndarray,float]]:
    elements1 = calculate_orbital_elements(body1, central_body, gravity_config)
    elements2 = calculate_orbital_elements(body2, central_body, gravity_config)
    a1, e1 = elements1.semi_major_axis, elements1.eccentricity
    a2, e2 = elements2.semi_major_axis, elements2.eccentricity
    points1 = generate_orbit_points(elements1, central_body, num_points=360)
    points2 = generate_orbit_points(elements2, central_body, num_points=360)
    for pt1 in points1:
        for pt2 in points2:
            dist = np.linalg.norm(pt1 - pt2)
            if dist < tol:
                return pt1, dist
    return None

def compute_transfer_delta_v(body: BodyProperties, central_body: BodyProperties, target_radius: float, gravity_config: GravityConfig) -> Tuple[float, float]:
    r1 = np.linalg.norm(body.position - central_body.position)
    v1 = np.linalg.norm(body.velocity - central_body.velocity)
    v_circ1 = calculate_orbital_velocity(r1,central_body.mass)
    v_circ2 = calculate_orbital_velocity(target_radius, central_body.mass)
    v_trans_a = math.sqrt(2*G*central_body.mass*target_radius/(r1*(r1+target_radius)))
    v_trans_b = math.sqrt(2*G*central_body*r1/(target_radius*(r1+target_radius)))
    delta_v1 = abs(v_trans_a - v_circ1)
    delta_v2 = abs(v_circ2 - v_trans_b)
    return delta_v1, delta_v2

def hohmann_transfer_planet(body: BodyProperties, central_body: BodyProperties, final_radius: float, gravity_config: GravityConfig, outpit_steps = 128) -> Dict:
    r1 = np.linalg.norm(body.position - central_body.position)
    r2 = final_radius
    elements = calculate_orbital_elements(body, central_body, gravity_config)
    mu = G * central_body.mass
    a_trans = (r1 + r2) / 2
    period_trans = 2 * math.pi * math.sqrt(a_trans**3/mu)
    points = []
    velocities = []
    for k in range(outpit_steps):
        nu = 2*math.pi*k/outpit_steps
        r = orbit_radius_from_the_true_anomaly(a_trans,0,nu)
        x = r * math.cos(nu)
        y = r * math.sin(nu)
        pos = central_body.position + np.array([x,y])
        vel_mag = math.sqrt(mu * (2/r-1/a_trans))
        vel = np.array([-math.sin(nu), math.cos(nu)]) * vel_mag
        points.append(pos)
        velocities.append(pos)
    
    delta_v1, delta_v2 = compute_transfer_delta_v(body, central_body, r1,gravity_config)
    return {
        'transfer_points': np.array(points),
        'transfer_velocitites': np.array(velocities),
        'a_transfer': a_trans,
        'period_transfer': period_trans,
        'delta_v1': delta_v1,
        'delta_v2': delta_v2
    }

def apply_orbit_maneuver(body: BodyProperties, maneuver_vector:np.ndarray) -> None:
    body.velocity += maneuver_vector

def orbit_simulation_step(bodies: OrbitalElements, central_body: BodyProperties, gravity_config: GravityConfig, dt: float, method: str = "rk4"):
    for body in bodies:
        if method == "euler":
            propagate_orbit_fixed_step(body, central_body, dt, gravity_config)
        elif method == "rk4":
            propagate_orbit_rk4(body, central_body, dt, gravity_config)
        else:
            propagate_orbit_fixed_step(body, central_body, dt, gravity_config)

def is_ecliptic_crossing(body: BodyProperties, central_body: BodyProperties,gravity_config: GravityConfig) -> bool:
    elements = calculate_orbital_elements(body,central_body ,gravity_config)
    return abs(elements.inclination) < 1e-5

def compute_orbit_distance(body: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig, n_points: int = 256) -> np.ndarray:
    elements = calculate_orbital_elements(body,central_body,gravity_config)
    angles = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    radii = [orbit_radius_from_the_true_anomaly(elements.semi_major_axis, elements.eccentricity, v) for v in angles]
    return np.array(radii)

def check_orbit_period_error(body: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig) -> float:
    elements = calculate_orbital_elements(body, central_body, gravity_config)
    calculated_period = orbital_period_from_semi_major_axis(elements.semi_major_axis, central_body.mass)
    actual_period = elements.orbital_period
    return abs(calculated_period - actual_period)

def check_orbit_circularity(body: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig, tolerance = 1e-5) -> bool:
    elements = calculate_orbital_elements(body, central_body, gravity_config)
    return abs(elements.eccentricity) < tolerance

def compute_next_periapsis(body: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig, t_current: float) -> float:
    elements = calculate_orbital_elements(body, central_body, gravity_config)
    period = elements.orbital_period
    M = elements.true_anomaly
    n = calculate_mean_motion(elements.semi_major_axis, G * (body.mass + central_body.mass))
    t_peri = t_current - M / n if n != 0 else t_current
    return t_peri + period

def mean_longitude(elements: OrbitalElements) -> float:
    return (elements.longitude_of_ascending_node + elements.argument_of_periapsis + elements.true_anomaly) % (2 * math.pi)

def anomaly_difference(elements1: OrbitalElements, elements2: OrbitalElements) -> float:
    ml1 = mean_longitude(elements1)
    ml2 = mean_longitude(elements2)
    diff = abs((ml1 - ml2 + math.pi) % (2 * math.pi) - math.pi)
    return diff

def mutual_inclination(elements1: OrbitalElements, elements2: OrbitalElements) -> float:
    i1 = elements1.inclination
    i2 = elements2.inclination
    Omega1 = elements1.longitude_of_ascending_node
    Omega2 = elements2.longitude_of_ascending_node
    return math.acos(
        math.cos(i1) * math.cos(i2) + 
        math.sin(i1) * math.sin(i2) * math.cos(Omega1 - Omega2)
    )

def laplace_coefficient(s: float, alpha: float, m: int = 0, max_terms: int = 100) -> float:
    coeff = 0.0
    for k in range(max_terms):
        coeff += math.comb(s + k,k) * (alpha ** k) / (k + m + 1)
    return coeff

def secular_apsidal_precession_rate(elements: OrbitalElements, other_elements: OrbitalElements, mass_perturber: float, central_mass: float) -> float:
    a = elements.semi_major_axis
    a_p = other_elements.semi_major_axis
    e = elements.eccentricity
    n = math.sqrt(G * central_mass / a **3)
    alpha = a / a_p
    return n * (mass_perturber / central_mass) * alpha * laplace_coefficient(1.5, alpha)

def mean_motion(elements: OrbitalElements, central_mass: float) -> float:
    return math.sqrt(G * central_mass / elements.semi_major_axis ** 3)

def longitude_of_periapsis(elements: OrbitalElements) -> float:
    return (elements.longitude_of_ascending_node + elements.argument_of_periapsis) % (2 * math.pi)

def spherical_to_cartesian(r: float, theta: float,phi: float) -> np.ndarray:
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return np.array([x, y, z])

def inclination_difference(elements1: OrbitalElements, elements2: OrbitalElements) -> float:
    return abs(elements1.inclination - elements2.inclination)

def ascending_node_seperation(elements: OrbitalElements) -> float:
    return elements.semi_major_axis * (1 - elements.eccentricity)

def orbit_energy_difference(elements1: OrbitalElements, elements2: OrbitalElements, mu: float) -> float:
    a1 = elements1.semi_major_axis
    a2 = elements2.semi_major_axis
    E1 = -mu / (2 * a1)
    E2 = -mu / (2 * a2)
    return abs(E1 -E2)

def advance_orbit_by_anomaly(body: BodyProperties, central_body: BodyProperties, anomaly_change: float, gravity_config: GravityConfig) -> None:
    elements = calculate_orbital_elements(body, central_body, gravity_config)
    new_true_anomaly = (elements.true_anomaly + anomaly_change) % (2 * math.pi)
    position, velocity = calculate_orbit_state_from_elements(OrbitalElements(
        semi_major_axis=elements.semi_major_axis,
        eccentricity=elements.eccentricity,
        inclination=elements.inclination,
        longitude_of_ascending_node= elements.longitude_of_ascending_node,
        argument_of_periapsis=elements.argument_of_periapsis,
        true_anomaly=new_true_anomaly,
        orbital_period=elements.orbital_period,
        specific_angular_momentum=elements.specific_angular_momentum
        ),
        central_body
        )

    body.position = position
    body.velocity = velocity

def barycenter(bodies: List[BodyProperties]) -> np.ndarray:
    total_mass = sum(b.mass for b in bodies)
    return sum((b.mass * b.position for b in bodies), np.zeros_like(bodies[0].position)) / total_mass

def total_angular_momentum(bodies: List[BodyProperties], reference: Optional[np.ndarray] = None) -> np.ndarray:
    if reference is None:
        reference = barycenter(bodies)
    total_L = np.zeros(3)
    for b in bodies:
        r_rel = np.append(b.position - reference, 0)
        v_rel = np.append(b.velocity,0)
        total_L += b.mass * np.cross(r_rel, v_rel)
    return total_L

def phase_space_trajectory(bodies: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig, duration: float, dt: float) -> Dict[str, np.ndarray]:
    num_steps = int(duration // dt)
    phase_trajectories = {i: [] for i in range(len(bodies))}
    for step in range(num_steps):
        for i, body in enumerate(bodies):
            propagate_orbit_fixed_step(body, central_body, dt, gravity_config)
            phase_trajectories[i].append(np.concatenate([body.position, body.velocity]))

    return {k: np.array(v) for k, v in phase_trajectories.item()}

def tot_energy(bodies: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig) -> float:
    return sum(b.mass * b.velocity for b in bodies)

def orbit_drift_analysis(body: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig, dt: float, total_time: float) -> Dict[str, float]:
    elements_inital = calculate_orbital_elements(body, central_body, gravity_config)
    positions = []
    times = []
    for t in np.arange(0, total_time, dt):
        propagate_orbit_fixed_step(body, central_body, dt, gravity_config)
        positions.append(body.position.copy())
        times.append(t)
    elements_final = calculate_orbital_elements(body, central_body, gravity_config)
    drift_a = abs(elements_final.semi_major_axis - elements_inital.semi_major_axis)
    drift_e = abs(elements_final.eccentricity - elements_inital.eccentricity)
    drift_i = abs(elements_final.inclination - elements_inital.inclination)
    return {"drift_a": drift_a,"drift_e": drift_e,"drift_i": drift_i}

def orbit_alignment_metric(elements1:OrbitalElements, elements2: OrbitalElements) -> float:
        i1  = elements1.inclination
        i2 = elements2.inclination
        Omega1 = elements1.longitude_of_ascending_node
        Omega2 = elements2.longitude_of_ascending_node
        return math.acos(
            math.cos(i1) * math.cos(i2) + 
            math.sin(i1) * math.sin(i2) * math.cos(Omega1 - Omega2)
        )

def laplace_coefficient(s: float, alpha: float,m: int = 0, max_terms: int = 100) -> float:
    coeff = 0.0
    for k in range(max_terms):
        coeff += math.comb(s + k, k) * (alpha ** k) / (k + m +1)
    return coeff

def secular_apsidal_precession_rate(elements: OrbitalElements, other_elements: OrbitalElements, mass_perturber: float, central_mass: float) -> float:
    a = elements.semi_major_axis
    a_p = other_elements.semi_major_axis
    e = elements.eccentricity
    n = math.sqrt(G * central_mass / a ** 3)
    alpha = a / a_p
    return n * (mass_perturber / central_mass) * alpha * laplace_coefficient(1.5, alpha)

def mean_motion(elements: OrbitalElements, central_mass: float) -> float:
    return math.sqrt(G * central_mass / elements.semi_major_axis ** 3)

def orbit_apocenter_pericenter(elements: OrbitalElements) -> Tuple[np.ndarray, np.ndarray]:
    a = elements.semi_major_axis
    e = elements.eccentricity
    r_peri = a * (1 - e)
    r_apo = a * (1 + e)
    x_peri = r_peri
    y_peri = 0.0
    x_apo = r_apo
    y_apo = 0.0
    return np.array([x_peri, y_peri]), np.array([x_apo, y_apo])

def orbit_state_series(body: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig, duration: float, dt: float):
    num_steps = int(duration // dt)
    positions, velocities, energies = [],[],[]
    for _ in range(num_steps):
        propagate_orbit_fixed_step(body, central_body, dt, gravity_config)
        positions.append(body.position.copy())
        velocities.append(body.velocity.copy())
        energies.append(calculate_orbital_energy(body,central_body, gravity_config))
        return {"positions": positions,"velocities": velocities,"energies":energies}

def longitude_of_periapsis(elements: OrbitalElements) -> float:
    return (elements.longitude_of_ascending_node + elements.argument_of_periapsis) % (2 * math.pi)

def spherical_to_cartesian(r: float, theta: float, phi: float) -> np.ndarray:
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return np.array([x,y,z])

def inclination_difference(elements1: OrbitalElements, elements2: OrbitalElements) -> float:
    return abs(elements1.inclination - elements2.inclination)

def ascending_node_seperation(elements1: OrbitalElements, elements2: OrbitalElements) -> float:
    return abs(elements1.longitude_of_ascending_node - elements2.longitude_of_ascending_node) % (2 * math.pi)

def periapsis_distance(elements: OrbitalElements) -> float:
    return elements.semi_major_axis * (1 - elements.eccentricity)

def apoapsis_distance(elements: OrbitalElements) -> float:
    return elements.semi_major_axis * (1 + elements.eccentricity)

def orbit_energy_differnce(elements1: OrbitalElements, elements2: OrbitalElements, mu:float) -> float:
    a1 = elements1.semi_major_axis
    a2 = elements2.semi_major_axis
    E1 = -mu / (2 * a1)
    E2 = -mu / (2 * a2)
    return abs(E1 - E2)

def advance_orbit_by_anomaly(body: BodyProperties, central_body: BodyProperties, anomaly_change: float, gravity_config: GravityConfig) -> None:
    elements = calculate_orbital_elements(body, central_body, gravity_config)
    new_true_anomaly = (elements.true_anomaly + anomaly_change) % (2 * math.pi)
    position, velocity = calculate_orbit_state_from_elements(OrbitalElements(
        semi_major_axis=elements.semi_major_axis,
        eccentricity=elements.eccentricity,
        inclination=elements.inclination,
        longitude_of_ascending_node=elements.longitude_of_ascending_node,
        argument_of_periapsis=elements.argument_of_periapsis,
        true_anomaly = new_true_anomaly,
        orbital_period=elements.orbital_period,
        specific_angular_momentum=elements.specific_angular_momentum
    ),
    central_body)
    body.position = position
    body.velocity =velocity

def barycenter(bodies: List[BodyProperties]) -> np.ndarray:
    total_mass = sum(b.mass for b in bodies)
    return sum((b.mass * b.position for b in bodies),
               np.zeros_like(bodies[0].position)) / total_mass

def total_angular_momentum(bodies: List[BodyProperties], refernce: Optional[np.ndarray] = None) -> np.ndarray:
    if refernce is None:
        refernce = barycenter(bodies)

    total_L = np.zeros(3)
    for b in bodies:
        r_rel = np.append(b.position - refernce, 0)
        v_rel = np.append(b.velocity, 0)
        total_L += b.mass * np.cross(r_rel,v_rel)
        return total_L
    
def phase_space_trajectory(bodies: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig, duration: float, dt: float) -> Dict[str, np.ndarray]:
    num_steps = int(duration // dt)
    phase_trajectories = {i: [] for i in range(len(bodies))}
    for step in range(num_steps):
        for i, body in enumerate(bodies):
            propagate_orbit_fixed_step(body, central_body, dt, gravity_config)
            phase_trajectories[i].append(np.concatenate([body.position, body.velocity]))
    return {k: np.array(v) for k, v in phase_trajectories.items()}

def tot_energy(bodies: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig) -> float:
    return sum(calculate_orbital_energy(b, central_body, gravity_config) for b in bodies)

def tot_momentum(bodies: List[BodyProperties]) -> np.ndarray:
    return sum(b.mass * b.velocity for b in bodies)

def orbit_drift_analysis(body: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig, dt: float, total_time: float) -> Dict[str, float]:
    elements_initial = calculate_orbital_elements(body, central_body, gravity_config)
    positions = []
    times = []
    for t in np.arange(0, total_time, dt):
        propagate_orbit_fixed_step(body, central_body, dt, gravity_config)
        positions.append(body.position.copy())
        times.append(t)
    elements_final = calculate_orbital_elements(body, central_body, gravity_config)
    drift_a = abs(elements_final.semi_major_axis - elements_initial.semi_major_axis)
    drift_e = abs(elements_final.eccentricity - elements_initial.eccentricity)
    drift_i = abs(elements_final.inclination - elements_initial.inclination)
    return {"drift_a": drift_a, "drift_e": drift_e, "drift_i": drift_i}

def orbit_alignment_metric(elements1: OrbitalElements, elements2: OrbitalElements) -> float:
    inc1 = elements1.inclination
    inc2 = elements2.inclination
    Omega1 = elements1.longitude_of_ascending_node
    Omega2 = elements2.longitude_of_ascending_node
    metric = (math.cos(inc1) * math.cos(inc2) +
              math.sin(inc1) * math.sin(inc2) *
              math.cos(Omega1 - Omega2))
    return metric

def random_orbital_elements(mu: float, semi_major_axis_range: Tuple[float, float], eccentricity_range: Tuple[float, float],
                           inclination_range: Tuple[float, float], omega_range: Tuple[float, float], Omega_range: Tuple[float, float], nu_range: Tuple[float, float]) -> OrbitalElements:
    a = np.random.uniform(*semi_major_axis_range)
    e = np.random.uniform(*eccentricity_range)
    i = np.random.uniform(*inclination_range)
    omega = np.random.uniform(*omega_range)
    Omega = np.random.uniform(*Omega_range)
    nu = np.random.uniform(*nu_range)
    period = 2 * math.pi * math.sqrt(a ** 3 / mu)
    h_vec = np.array([0., 0., math.sqrt(mu * a * (1 - e ** 2))])
    return OrbitalElements(
        semi_major_axis=a,
        eccentricity=e,
        inclination=i,
        longitude_of_ascending_node=Omega,
        argument_of_periapsis=omega,
        true_anomaly=nu,
        orbital_period=period,
        specific_angular_momentum=h_vec[:2]
    )

def orbit_cluster_centroid(elements_list: List[OrbitalElements]) -> OrbitalElements:
    a = np.mean([el.semi_major_axis for el in elements_list])
    e = np.mean([el.eccentricity for el in elements_list])
    i = np.mean([el.inclination for el in elements_list])
    Omega = np.mean([el.longitude_of_ascending_node for el in elements_list])
    omega = np.mean([el.argument_of_periapsis for el in elements_list])
    nu = np.mean([el.true_anomaly for el in elements_list])
    period = np.mean([el.orbital_period for el in elements_list])
    sam = np.mean([el.specific_angular_momentum for el in elements_list], axis=0)
    return OrbitalElements(
        semi_major_axis=a, eccentricity=e, inclination=i,
        longitude_of_ascending_node=Omega, argument_of_periapsis=omega,
        true_anomaly=nu, orbital_period=period, specific_angular_momentum=sam
    )

def find_all_resonant_pairs(bodies: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig, tol: float = 1e-4) -> List[Tuple[int, int, float, float]]:
    elements_list = [calculate_orbital_elements(b, central_body, gravity_config) for b in bodies]
    pairs = []
    for i in range(len(elements_list)):
        for j in range(i+1, len(elements_list)):
            T1 = elements_list[i].orbital_period
            T2 = elements_list[j].orbital_period
            if T1 == 0 or T2 == 0:
                continue
            ratio = T1 / T2
            round_ratio = round(ratio)
            deviation = abs(ratio - round_ratio)
            if deviation < tol:
                pairs.append((i, j, ratio, deviation))
    return pairs

def orbit_precession(elements: OrbitalElements, mass_perturber: float, central_mass: float, a_perturber: float) -> float:
    a = elements.semi_major_axis
    e = elements.eccentricity
    n = math.sqrt(G * central_mass / a ** 3)
    alpha = a / a_perturber
    b = laplace_coefficient(1.5, alpha)
    return n * (mass_perturber / central_mass) * alpha * b

def set_orbit_elements_to_body(body: BodyProperties, central_body: BodyProperties, elements: OrbitalElements):
    position, velocity = calculate_orbit_state_from_elements(elements, central_body)
    body.position = position
    body.velocity = velocity

def evolve_body_with_nbody(bodies: List[BodyProperties], central_body: BodyProperties, dt: float, gravity_integrator: GravityIntegrator, steps: int):
    for _ in range(steps):
        gravity_integrator.integrate_step(bodies + [central_body], dt)

def minimum_energy_transfer_time(a1: float, a2: float, mu: float) -> float:
    return math.pi * math.sqrt(((a1 + a2) / 2) ** 3 / mu)

def serialize_orbital_elements(elements: OrbitalElements) -> dict:
    return {
        "semi_major_axis": elements.semi_major_axis,
        "eccentricity": elements.eccentricity,
        "inclination": elements.inclination,
        "longitude_of_ascending_node": elements.longitude_of_ascending_node,
        "argument_of_periapsis": elements.argument_of_periapsis,
        "true_anomaly": elements.true_anomaly,
        "orbital_period": elements.orbital_period,
        "specific_angular_momentum": elements.specific_angular_momentum.tolist() if hasattr(elements.specific_angular_momentum, 'tolist') else elements.specific_angular_momentum
    }

def orbit_batch_prop_epochs(bodies: List[BodyProperties], central_body: BodyProperties, dt: float, epochs: int, gravity_config: GravityConfig) -> List[List[np.ndarray]]:
    all_positions = []
    for body in bodies:
        pos_series = []
        for epoch in range(epochs):
            propagate_orbit_fixed_step(body, central_body, dt, gravity_config)
            pos_series.append(body.position.copy())
        all_positions.append(pos_series)
    return all_positions

def orbit_match_score(elements1: OrbitalElements, elements2: OrbitalElements, weights: Optional[dict] = None) -> float:
    if weights is None:
        weights = {
            "a": 1.0, "e": 1.0, "i": 1.0, "O": 1.0, "o": 1.0, "nu": 1.0
        }
    score = (weights['a'] * abs(elements1.semi_major_axis - elements2.semi_major_axis) +
             weights['e'] * abs(elements1.eccentricity - elements2.eccentricity) +
             weights['i'] * abs(elements1.inclination - elements2.inclination) +
             weights['O'] * abs(elements1.longitude_of_ascending_node - elements2.longitude_of_ascending_node) +
             weights['o'] * abs(elements1.argument_of_periapsis - elements2.argument_of_periapsis) +
             weights['nu'] * abs(elements1.true_anomaly - elements2.true_anomaly))
    return score

def migration_due_to_drag(body: BodyProperties, central_body: BodyProperties, drag_coeff: float, dt: float, gravity_config: GravityConfig) -> None:
    velocity = body.velocity - central_body.velocity
    damping = 1 - drag_coeff * dt
    body.velocity = central_body.velocity + velocity * damping
    propagate_orbit_fixed_step(body, central_body, dt, gravity_config)

def is_coplanar(elements1: OrbitalElements, elements2: OrbitalElements, tol: float = 1e-5) -> bool:
    return abs(elements1.inclination - elements2.inclination) < tol

def time_to_periapsis_from_true_anomaly(a: float, e: float, true_anomaly: float, mu: float) -> float:
    E = 2 * math.atan(math.sqrt((1 - e) / (1 + e)) * math.tan(true_anomaly / 2))
    M = E - e * math.sin(E)
    n = math.sqrt(mu / a ** 3)
    return M / n

def batch_orbit_parameters(bodies: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig) -> List[dict]:
    return [serialize_orbital_elements(calculate_orbital_elements(b, central_body, gravity_config)) for b in bodies]

if __name__ == "__main__":
    print("Orbits module internal test - Basic simulation")

    # Assume gravity.py is correctly imported with BodyProperties, GravityConfig, G
    sun = BodyProperties(
        body_type=BodyType.STAR,
        mass=1.989e30,
        radius=6.96e8,
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.0, 0.0])
    )
    earth = BodyProperties(
        body_type=BodyType.PLANET,
        mass=5.972e24,
        radius=6.371e6,
        position=np.array([1.496e11, 0.0]),
        velocity=np.array([0.0, 29783.0])
    )
    config = GravityConfig()
    elements = calculate_orbital_elements(earth, sun, config)
    print(f"Earth semi-major axis: {elements.semi_major_axis:.3e} m, ecc: {elements.eccentricity:.6f}, period: {elements.orbital_period:.3f} s")

    history = orbit_state_series(earth, sun, config, duration=3.154e7, dt=86400)
    print(f"Simulated {len(history['positions'])} days of orbit.")

    peri, apo = orbit_apocenter_pericenter(elements)
    print(f"Earth periapsis: {peri}, apoapsis: {apo}")

    pos, vel = calculate_orbit_state_from_elements(elements, sun)
    print("Earth orbit state from elements")
    print("  Position (m):", pos)
    print("  Velocity (m/s):", vel)

    transfer_delta_v = calculate_orbit_transfer_delta_v(earth, 2.279e11, sun)  # Example: transfer to Mars's orbital radius
    print(f"Hohmann transfer delta-v from Earth's to Mars's orbit: {transfer_delta_v:.2f} m/s")

    elements_list = [calculate_orbital_elements(earth, sun, config)]
    for i in range(4):
        new_elem = random_orbital_elements(
            mu=G*sun.mass, semi_major_axis_range=(1.2e11, 1.8e11),
            eccentricity_range=(0, 0.1), inclination_range=(0, 0.05),
            omega_range=(0, 2*math.pi), Omega_range=(0, 2*math.pi), nu_range=(0, 2*math.pi)
        )
        elements_list.append(new_elem)
    cluster_centroid = orbit_cluster_centroid(elements_list)
    print("Orbital elements cluster centroid:")
    print(serialize_orbital_elements(cluster_centroid))

    print("Orbits module tests complete.")
