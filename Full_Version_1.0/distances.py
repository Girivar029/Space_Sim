import numpy as np
import math
from typing import Tuple, List, Optional
from gravity import BodyProperties, GravityConfig, G
from orbits import calculate_orbital_elements, orbit_radius_from_the_true_anomaly, solve_kepler

def euclidean_distance(body1: BodyProperties, body2: BodyProperties) -> float:
    return np.linalg.norm(body1.position - body2.position)

def polar_to_cartesian(r: float, theta: float) -> np.ndarray:
    return np.array([r * np.cos(theta), r * np.sin(theta)])

def closest_approach_distance(body1: BodyProperties, body2: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig, num_samples: int = 360) -> float:
    elements1 = calculate_orbital_elements(body1, central_body, gravity_config)
    elements2 = calculate_orbital_elements(body2, central_body,gravity_config)
    min_distance = float('inf')
    for nu1 in np.linspace(0 , 2 * np.pi, num_samples):
        r1 = orbit_radius_from_the_true_anomaly(elements1.semi_major_axis, elements1.eccentricity, nu1)
        pos1 = polar_to_cartesian(r1, nu1) + central_body.position
        for nu2 in np.linspace(0, 2 *np.pi, num_samples):
            r2 = orbit_radius_from_the_true_anomaly(elements2.semi_major_axis, elements2.eccentricity, nu2)
            pos2 = polar_to_cartesian(r2, nu2) + central_body.position
            dist = np.linalg.norm(pos1 - pos2)
            if dist < min_distance:
                min_distance = dist

    return min_distance

def is_collision_likely(body1: BodyProperties, body2: BodyProperties, distance_threshold: Optional[float] = None) -> bool:
    if distance_threshold is None:
        distance_threshold = body1.radius + body2.radius
    actual_distance = euclidean_distance(body1, body2)
    return actual_distance <= distance_threshold

def propagate_position(body: BodyProperties, time: float, central_body: BodyProperties, gravity_config: GravityConfig) -> np.ndarray:
    elements = calculate_orbital_elements(body,central_body, gravity_config)
    mu = G * (body.mass + central_body.mass)
    n = np.sqrt(mu / elements.semi_major_axis ** 3)
    M = n * time
    E = solve_kepler(M, elements.eccentricity)
    true_anomaly = 2 * np.arctan2(np.sqrt(1 + elements.eccentricity) * np.sin(E / 2), np.sqrt(1 - elements.eccentricity) * np.cos(E / 2))
    r = orbit_radius_from_the_true_anomaly(elements.semi_major_axis, elements.eccentricity, true_anomaly)
    pos = polar_to_cartesian(r, true_anomaly)
    return central_body.position

def closest_approach_time(body1: BodyProperties, body2: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig, max_time: float, dt: float = 1e4) -> Optional[float]:
    t = 0.0
    min_dist = float('inf')
    min_t = None
    while t <= max_time:
        pos1 = propagate_position(body1, t, central_body, gravity_config)
        pos2 = propagate_position(body2, t, central_body, gravity_config)
        dist = np.linalg.norm(pos1 - pos2)
        if dist < min_dist:
            min_dist = dist
            min_t = t
        t += dt
    return min_t

def sitances_matrix(bodies: List[BodyProperties]) -> np.ndarray:
    n = len(bodies)
    matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i + 1,n):
            dist = euclidean_distance(bodies[i], bodies[j])
            matrix[i,j] = dist
            matrix[j,i] = dist
    return matrix

def find_close_pairs(bodies: List[BodyProperties],threshold:float) -> List[Tuple[int, int]]:
    pairs = []
    n = len(bodies)
    for i in range(n):
        for j in range(i + 1,n):
            if euclidean_distance(bodies[i],bodies[j]) <= threshold:
                pairs.append((i,j))
    return pairs

def find_colliding_pairs(bodies: List[BodyProperties]) -> List[Tuple[int,int]]:
    pairs = []
    n = len(bodies)
    for i in range(n):
        for j in range(i+1,n):
            r_sum = bodies[i].radius + bodies[j].radius
            if euclidean_distance(bodies[i],bodies[j]) <= r_sum:
                pairs.append((i,j))
    return pairs

def radial_distance_variation(body: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig, num_samples: int = 360) -> Tuple[float,float]:
    elements = calculate_orbital_elements(body, central_body, gravity_config)
    min_r = float('inf')
    max_r = 0.0
    for nu in np.linspace(0,2 * math.pi, num_samples):
        r = orbit_radius_from_the_true_anomaly(elements.semi_major_axis, elements.eccentricity, nu)
        if r < min_r:
            min_r = r
        if r > max_r:
            max_r = r
    return min_r, max_r

def average_interbody_distance(bodies: List[BodyProperties]) -> float:
    distances = []
    n = len(bodies)
    for i in range(n):
        for j in range(i + 1,n):
            distances.append(euclidean_distance(bodies[i], bodies[j]))
    if len(distances) == 0:
        return 0.0
    return sum(distances) / len(distances)

def minimum_distance_to_others(body_index: int, bodies: List[BodyProperties]) -> float:
    dists = []
    body = bodies[body_index]
    for i, other in enumerate(bodies):
        if i != body_index:
            dists.append(euclidean_distance(body, other))
    return min(dists) if dists else 0.0

def farthest_distance_to_others(body_index: int, bodies: List[BodyProperties]) -> float:
    dists = []
    body = bodies[body_index]
    for i, other in enumerate(bodies):
        if i != body_index:
            dists.append(euclidean_distance(body, other))
    return max(dists) if dists else 0.0

def detect_clump(bodies: List[BodyProperties], threshold: float) -> List[List[int]]:
    clumps = []
    visited = set()
    n = len(bodies)

    def dfs(node, group):
        visited.add(node)
        group.append(node)
        for neighbor in range(n):
            if neighbor not in visited:
                if euclidean_distance(bodies[node], bodies[neighbor]) <= threshold:
                    dfs(neighbor,group)

    for i in range(n):
        if i not in visited:
            group = []
            dfs(i, group)
            if len(group) > 1:
                clumps.append(group)

    return clumps

def calculate_center_of_mass(bodies: List[BodyProperties]) -> np.ndarray:
    total_mass = sum(b.mass for b in bodies)
    if total_mass == 0:
        return np.zeros(2)
    
    weighted_positions = sum(b.mass * b.position for b in bodies)
    return weighted_positions / total_mass

def distances_report(bodies: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig):
    report= {}
    n = len(bodies)
    for i in range(n):
        body = bodies[i]
        min_dist = minimum_distance_to_others(i, bodies)
        max_dist = farthest_distance_to_others(i, bodies)
        periapsis, apoapsis = radial_distance_variation(body, central_body, gravity_config)
        report[body] = {
            "min_distance": min_dist,
            "max_distance": max_dist,
            "radial_min": periapsis,
            "radial_max": apoapsis
        }
    return report
    
def find_near_contact_pairs(bodies: List[BodyProperties], buffer: float = 1e7) -> List[Tuple[int,int]]:
    pairs = []
    n = len(bodies)
    for i in range(n):
        for j in range(i + 1,n):
            threshold = bodies[i].radius,bodies[j].radius + buffer
            if euclidean_distance(bodies[i], bodies[j]) <= threshold:
                pairs.append((i,j))
    return pairs

def estimate_collision_likelihood(body1: BodyProperties, body2: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig, time_window: float, dt: float = 1e4) -> bool:
    t = 0.0
    while t < time_window:
        pos1 = propagate_position(body1, t, central_body, gravity_config)
        pos2 = propagate_position(body2, t, central_body, gravity_config)
        dist = np.linalg.norm(pos1 - pos2)
        threshold = body1.radius + body2.radius
        if dist <= threshold:
            return True
        t += dt
    return False

def projected_distance(body1_pos: np.ndarray, body2_pos: np.ndarray, observer_direction: np.ndarray = np.ndarray([0,0,1])) -> float:
    relative_vector = np.append(body1_pos, 0) - np.append(body2_pos,0)
    projected_vector = relative_vector - np.dot(relative_vector, observer_direction) * observer_direction
    return np.linalg.norm(projected_vector)

def sorted_all_distances(bodies: List[BodyProperties]) -> List[Tuple[float,int,int]]:
    distances = []
    n = len(bodies)
    for i in range(n):
        for j in range(i+1,n):
            dist = euclidean_distance(bodies[i], bodies[j])
            distances.append((dist,i,j))
    distances.sort(key = lambda x: x[0])
    return distances

def find_minimum_distance_pair(bodies: List[BodyProperties]) -> Tuple[int, int, float]:
    min_dist = float('inf')
    min_pair = (-1,-1)
    n = len(bodies)
    for i in range(n):
        for j in range(i+1,n):
            dist = euclidean_distance(bodies[i], bodies[j])

            if dist < min_dist:
                min_dist = dist
                min_pair = (i,j)
    return min_pair[0], min_pair[1], min_dist