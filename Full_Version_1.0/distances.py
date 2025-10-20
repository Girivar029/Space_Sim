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

def trajectory_intersection_times(body1:BodyProperties,body2:BodyProperties,central_body:BodyProperties,gravity_config:GravityConfig,time_limit: float, dt:float = 1e4) -> List[float]:
    collison_times = []
    t= 0.0
    threshold = body1.radius + body2.radius
    while t < time_limit:
        pos1 = propagate_position(body1,t,central_body,gravity_config)
        pos2 = propagate_position(body2,t,central_body,gravity_config)
        dist = np.linalg.norm(pos1 - pos2)
        if dist <= threshold:
            collison_times.append(t)
            t += 2 * dt
        else:
            t += dt
    return collison_times

def bodies_within_radius(bodies:List[BodyProperties],center: np.ndarray, radius:float) -> List[int]:
    indices = []
    for i, body in enumerate(bodies):
        if np.linalg.norm(body.position - center) <= radius:
            indices.append(i)
    return indices

def density_profile_around_point(bodies: List[BodyProperties], center: np.ndarray,max_radius:float,bins: int = 30) -> Tuple[np.ndarray,np.ndarray]:
    distances = [np.linalg.norm(body.position - center) for body in bodies]
    hist, bin_edges = np.histogram(distances,bins=bins, range=(0,max_radius))
    shell_volumes = 4/3 * math.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    number_density = hist / shell_volumes
    return bin_edges[1:], number_density

def radial_velocity_diff(body1:BodyProperties,body2:BodyProperties,central_body:BodyProperties) -> float:
    r1_vec = body1.position - central_body.position
    r2_vec = body2.position - central_body.position
    r1_hat = r1_vec / np.linalg.norm(r1_vec)
    r2_hat = r2_vec / np.linalg.norm(r2_vec)
    v1_radial = np.dot(body1.velocity - central_body.velocity,r1_hat)
    v2_radial = np.dot(body2.velocity - central_body.velocity,r2_hat)
    return abs(v1_radial,v2_radial)

def average_radial_seperation(bodies:List[BodyProperties],central_body:BodyProperties) -> float:
    radial_distances = [np.linalg.norm(body.position - central_body.position) for body in bodies]
    return sum(radial_distances) / len(radial_distances) if radial_distances else 0.0

def identify_close_orbital_phases(body1:BodyProperties,body2:BodyProperties,central_body:BodyProperties, gravity_config:GravityConfig,num_phases: int = 100) -> List[Tuple[float,float]]:
    elements1 = calculate_orbital_elements(body1,central_body, gravity_config)
    elements2 = calculate_orbital_elements(body2,central_body,gravity_config)
    close_pairs = []
    for nu1 in np.linspace(0, 2 * math.pi, num_phases):
        r1 = orbit_radius_from_the_true_anomaly(elements1.semi_major_axis,elements1.eccentricity,nu1)
        pos1 = polar_to_cartesian(r1,nu1) + central_body.position
        for nu2 in np.linspace(0, 2 * math.pi,num_phases):
            r2 = orbit_radius_from_the_true_anomaly(elements2.semi_major_axis,elements2.eccentricity,nu2)
            pos2 = polar_to_cartesian(r2,nu2) + central_body.position
            dist = np.linalg.norm(pos1 - pos2)
            if dist <= (body1.radius + body2.radius):
                close_pairs.append((nu1,nu2))
    return close_pairs

def linear_approximate_distance(body1:BodyProperties,body2:BodyProperties,time:float) -> float:
    pos1 = body1.position + body1.velocity * time
    pos2 = body2.position + body2.velocity * time
    return np.linalg.norm(pos1 - pos2)

def extrapolate_closest_approach_linear(body1: BodyProperties, body2:BodyProperties,time_steps:int = 1000, dt: float = 1000) -> Tuple[float,float]:
    min_dist = float('inf')
    min_t = 0.0
    for step in range(time_steps):
        t = step * dt
        dist = linear_approximate_distance(body1,body2,t)
        if dist < min_dist:
            min_dist = dist
            min_t = t
    return min_t,min_dist

def translate_positions(bodies: List[BodyProperties],offset: np.ndarray) -> None:
    for body in bodies:
        body.position += offset

def scale_positions(bodies: List[BodyProperties],scale_factor:float) -> None:
    for body in bodies:
        body.position *= scale_factor

def rotate_positions(bodies: List[BodyProperties], angle_rad: float) -> None:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotation_matrix = np.array([[cos_a,-sin_a],[sin_a,cos_a]])
    for body in bodies:
        body.position = rotation_matrix @ body.position

def average_pairwise_distance(bodies:List[BodyProperties]) -> float:
    distances = []
    n = len(bodies)
    for i in range(n):
        for j in range(i + 1,n):
            dist = euclidean_distance(bodies[i],bodies[j])
            distances.append(dist)
    if not distances:
        return 0.0
    return sum(distances) / len(distances)

def median_pairwise_distances(bodies:List[BodyProperties]) -> float:
    distances = []
    n = len(bodies)
    for i in range(n):
        for j in range(i+1, n):
            distances.append(euclidean_distance(bodies[i],bodies[j]))
    if not distances:
        return 0.0
    distances_np = np.array(distances)
    return float(np.median(distances_np))
    
def maximise_pairwise_distance(bodies: List[BodyProperties]) -> float:
    max_dist = 0.0
    n = len(bodies)
    for i in range(n):
        for j in range(i+1,j):
            dist = euclidean_distance(bodies[i],bodies[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist

def filter_bodies_by_distance(bodies:List[BodyProperties],center:np.ndarray,max_distance:float) -> List[BodyProperties]:
    filtered = [body for body in bodies if np.linalg.norm(body.position * center)<= max_distance]
    return filtered

def calculate_cluster_center(bodies:List[BodyProperties]) -> Optional[np.ndarray]:
    if not bodies:
        return None
    positions = np.array([b.positions for b in bodies])
    return np.mean(positions,axis=0)

def find_farthest_body_from_point(bodies: List[BodyProperties], point: np.ndarray) -> Optional[BodyProperties]:
    if not bodies:
        return None
    max_dist = -1
    farthest_body = None
    for body in bodies:
        dist = np.linalg.norm(body.position - point)
        if dist > max_dist:
            max_dist = dist
            farthest_body = body
    return farthest_body

def bodies_within_annulus(bodies:List[BodyProperties], center:np.ndarray,r_min:float,r_max: float):
    filtered = []
    for body in bodies:
        dist = np.linalg.norm(body.position - center)
        if r_min <= dist <= r_max:
            filtered.append(body)
    return filtered

def incremental_proximity_filter(bodies: List[BodyProperties], start:int, end:int,cutoff:float) -> List[int]:
    if start < 0 or start >= len(bodies):
        return []
    ref_body = bodies[start]
    selected_indices = []
    for i in range(start, min(end, len(bodies))):
        dist = euclidean_distance(ref_body,bodies[i])
        if dist <= cutoff:
            selected_indices.append(i)
    return selected_indices

def smooth_distance_series(series: List[float],window_size:int) -> List[float]:
    if window_size <= 1:
        return series
    n = len(series)
    smoothed = []
    half_window = window_size // 2
    for i in range(n):
        start_idx = max(0,i - half_window)
        end_idx = min(n, i + half_window + 1)
        window_avg = sum(series[start_idx:end_idx]) / (end_idx - start_idx)
        smoothed.append(window_avg)
    return smoothed

def time_series_distance_between_bodies(body1:BodyProperties,body2:BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig,total_time: float, dt: float) -> List[float]:
    distances = []
    steps = int(total_time // dt)
    for step in range(steps):
        t = step * dt
        pos1 = propagate_position(body1,t,central_body,gravity_config)

def distance_crossing_events(distance_series: List[float], threshold:float) -> List[int]:
    crossing_points = []
    for i in range(1, len(distance_series)):
        if distance_series[i-1] > threshold and distance_series[i] <= threshold:
            crossing_points.append(i)
    return crossing_points

def average_cluster_radius(bodies:List[BodyProperties], center:np.ndarray) -> float:
    if not bodies:
        return 0.0
    dist_sum = 0.0
    for body in bodies:
        dist_sum += np.linalg.norm(body.position - center)
    return dist_sum/len(bodies)

def bounding_box(bodies: List[BodyProperties]) -> Tuple[np.ndarray,np.ndarray]:
    positions = np.array([b.position for b in bodies])
    min_corner = np.min(positions, axis = 0)
    max_corner = np.max(positions, axis=0)
    return min_corner, max_corner

def body_distance_to_line(body: BodyProperties, point_a: np.ndarray, point_b: np.ndarray) -> float:
    p = body.position
    a_to_p = p - point_a
    a_to_b = point_b - point_a
    line_length = np.linalg.norm(a_to_b)
    if line_length == 0.0:
        return np.linalg.norm(a_to_p)
    proj = np.dot(a_to_p,a_to_b) / line_length
    closest = point_a + (proj / line_length) * a_to_b
    return np.linalg.norm(p - closest)

def within_sector(bodies: List[BodyProperties], vertex: np.ndarray, ang1: float, ang2: float,rmax:float) -> List[BodyProperties]:
    results = []
    dtheta = (ang2 - ang1) % (2 * math.pi)
    for b in bodies:
        rel = b.position - vertex
        dist = np.linalg.norm(rel)
        theta = math.atan2(rel[1], rel[0] % (2 * math.pi))
        if dist <= rmax:
            compare = (theta - ang1) % (2 * math.pi)
            if compare <= dtheta:
                results.append(b)
    return results

def trajectory_proximity(body: BodyProperties, trajectory: List[np.ndarray], threshold: float) -> bool:
    for pos in trajectory:
        if np.linalg.norm(body.position - pos) < threshold:
            return True
        
    return False

def min_distance_to_trajectory(body: BodyProperties, trajectory: List[np.ndarray]) -> float:
    return min(np.linalg.norm(body.position - pos) for pos in trajectory)

def radial_density_histogram(bodies: List[BodyProperties], origin: np.ndarray, max_radius: float, n_bins: int = 30) -> Tuple[np.ndarray,np.ndarray]:
    dists = [np.linalg.norm(body.position - origin)for body in bodies]
    hist, bin_edges = np.histogram(dists, bins = n_bins, range = (0, max_radius))
    return hist, bin_edges

def circular_overlap_fraction(bodies: List[BodyProperties], test_center: np.ndarray, test_radius: float) -> float:
    inside = 0
    for b in bodies:
        if np.linalg.norm(b.position - test_center) <= test_radius:
            inside += 1
    return inside / len(bodies) if bodies else 0.0

def distance_heatmap(bodies: List[BodyProperties], grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    heatmap = np.zeros((len(grid_x),len(grid_y)))
    for ix, x in enumerate([grid_x]):
        for iy, y in enumerate(grid_y):
            pos = np.ndarray([x,y])
            inv_sum = 0.0
            for b in bodies:
                dist = np.linalg.norm(pos - b.position)
                if dist > 0:
                    inv_sum += 1.0 / dist
                    heatmap[ix,iy] = inv_sum
    return heatmap

def bodies_outside_radius(bodies: List[BodyProperties], center: np.ndarray, radius: float) -> List[BodyProperties]:
    return [b for b in bodies if np.linalg.norm(b.position - center) > radius]

def moving_average_distance(distances: List[float], k: int) -> List[float]:
    n = len(distances)
    if k < 1 or k >= n: return distances
    return [sum(distances[max(0,i-k+1): i + 1]) / min(k,i+1) for i in range(n)]

