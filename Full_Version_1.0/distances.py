import numpy as np
import math
from typing import Tuple, List, Optional, Dict
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
    elements = calculate_orbital_elements(body, central_body, gravity_config)
    mu = G * (body.mass + central_body.mass)
    n = np.sqrt(mu / elements.semi_major_axis ** 3)
    M = n * time
    E = solve_kepler(M, elements.eccentricity)
    true_anomaly = 2 * np.arctan2(np.sqrt(1 + elements.eccentricity) * np.sin(E / 2), np.sqrt(1 - elements.eccentricity) * np.cos(E / 2))
    r = orbit_radius_from_the_true_anomaly(elements.semi_major_axis, elements.eccentricity, true_anomaly)
    pos = polar_to_cartesian(r, true_anomaly) + central_body.position
    return pos

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

def distances_matrix(bodies: List[BodyProperties]) -> np.ndarray:
    positions = np.array([b.position for b in bodies])
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # shape (n, n, dim)
    matrix = np.linalg.norm(diff, axis=-1)
    return matrix

def find_close_pairs(bodies: List[BodyProperties],threshold:float) -> List[Tuple[int, int]]:
    pairs = []
    n = len(bodies)
    for i in range(n):
        for j in range(i + 1,n):
            if euclidean_distance(bodies[i],bodies[j]) <= threshold:
                pairs.append((i,j))
    return pairs

def find_colliding_pairs(bodies: List[BodyProperties]) -> List[Tuple[int, int]]:
    positions = np.array([b.position for b in bodies])
    radii = np.array([b.radius for b in bodies])
    n = len(bodies)
    pairs = []
    # Compute all pairwise distances with broadcasting
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    rsum = radii[:, np.newaxis] + radii[np.newaxis, :]
    colliding_indices = np.transpose(np.nonzero((dists <= rsum) & (np.triu(np.ones((n, n)), k=1) == 1)))
    for i, j in colliding_indices:
        pairs.append((i, j))
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

def compute_dynamic_proximity_matrix(bodies: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig, total_time: float, dt: float) -> List[np.ndarray]:
    steps = int(total_time // dt)
    n = len(bodies)
    matrices = []
    for step in range(steps):
        matrix = np.zeros((n,n))
        positions = [propagate_position(b, step * dt, central_body, gravity_config) for b in bodies]
        for  i in range(n):
            for j in range(i+1,n):
                d = np.linalg.norm(positions[i] - positions[j])
                matrix[i,j] = d
                matrix[j,i] = d
        matrices.append(matrix)
    return matrices

def detect_collision_events(bodies: List[BodyProperties], central_body:BodyProperties, gravity_config: GravityConfig, total_time: float, dt: float) -> List[Tuple[int,int,float]]:
    events = []
    steps = int(total_time // dt)
    n = len(bodies)
    for step in range(steps):
        t = step * dt
        positions = [propagate_position(b,t,central_body,gravity_config)for b in bodies]
        for i in range(n):
            for j in range(i+1,n):
                seperation = np.linalg.norm(positions[i] - positions[j])
                limit = bodies[i].radius + bodies[j].radius
                if seperation <= limit:
                    events.append((i,j,t))
    return events

def orbital_distance_grid(bodies:List[BodyProperties],central_body: BodyProperties, gravity_config: GravityConfig, samples: int) -> Dict[int, List[float]]:
    table = {}
    for idx, body in enumerate(bodies):
        elements = calculate_orbital_elements(body,central_body,gravity_config)
        dist_list = []
        for nu in np.linspace(0,2 * math.pi,samples):
            r = orbit_radius_from_the_true_anomaly(elements.semi_major_axis,elements.eccentricity, nu)
            dist_list.append(r)
        table[idx] = dist_list
    return table

def pairwise_relative_velocity_matrix(bodies: List[BodyProperties]) -> np.ndarray:
    n = len(bodies)
    v_rel = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,j):
            vel_diff = np.linalg.norm(bodies[i].velocity - bodies[j].velocity)
            v_rel[i,j] = vel_diff
            v_rel[j,i] = vel_diff
    return v_rel

def moving_proximity_statistics(bodies: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig,total_time: float,dt: float) -> Dict[str,float]:
    matrices = compute_dynamic_proximity_matrix(bodies, central_body, gravity_config, total_time,dt)
    all_values = []
    for mtx in matrices:
        mask = np.triu(np.ones_like(mtx),1)
        dist_values = mtx[mask == 1]
        all_values.extend(dist_values)
        all_values.extend(dist_values)
    arr = np.array(all_values)
    return {"min": np.min(arr),"max": np.max(arr), "mean": np.mean(arr),"stddev": np.std(arr)}

def system_clumping_index(bodies: List[BodyProperties], cut_distance: float) -> float:
    n = len(bodies)
    pairs = 0
    for i in range(n):
        for j in range(i+1,n):
            if euclidean_distance(bodies[i],bodies[j]) < cut_distance:
                pairs += 1
    total_possible = (n * (n - 1)) / 2
    return pairs / total_possible if total_possible else 0.0

def body_distance_trends(body: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig, duration: float, dt: float) -> Tuple[float,float,float]:
    samples = int(duration //dt)
    distances = []
    for step in range(samples):
        pos = propagate_position(body, step * dt,central_body, gravity_config)
        dist = np.linalg.norm(pos - central_body.position)
        distances.append(dist)
    arr = np.array(distances)
    gradient = np.gradient(arr)
    return float(np.mean(arr)), float(np.max(arr)), float(np.mean(gradient))

def track_cluster_dispersion(bodies: List[BodyProperties], total_time: float, dt: float, central_body: BodyProperties, gravity_config: GravityConfig) -> List[float]:
    steps = int(total_time // dt)
    dispersion = []
    for step in range(steps):
        positions = [propagate_position(b,step * dt,central_body,gravity_config)for b in bodies]
        positions = np.array(positions)
        center = np.mean(positions, axis = 0)
        dists = [np.linalg.norm(p - center)for p in positions]
        dispersion.append(np.std(dists))
    return dispersion

def compute_distance_autocorrelation(distance_series: List[float]) -> np.ndarray:
    n = len(distance_series)
    mean_val = np.mean(distance_series)
    var = np.var(distance_series)
    corr = np.correlate(distance_series - mean_val,distance_series - mean_val, mode='full')
    return corr

def predict_encounters(bodies: List[BodyProperties],central_body: BodyProperties, gravity_config: GravityConfig,time_future: float, dt: float) -> List[Tuple[int,int,float]]:
    events = []
    positions_initial = [b.position.copy() for b in bodies]
    for t in np.arange(0, time_future,dt):
        pos_snapshot = [propagate_position(b,t,central_body,gravity_config)for b in bodies]
        n = len(pos_snapshot)
        for i in range(n):
            for j in range(i+1,n):
                dist = np.linalg.norm(pos_snapshot[i] - pos_snapshot[j])
                threshold = bodies[i].radius + bodies[j].radius
                if dist <= threshold:
                    events.append((i,j,t))
    for idx, body in enumerate(bodies):
        body.position = positions_initial[idx]
    return events

def calculate_variance_of_distances(bodies: List[BodyProperties]) -> float:
    values = []
    n = len(bodies)
    for i in range(n):
        for j in range(i+1,n):
            values.append(euclidean_distance(bodies[i],bodies[j]))
    if not values:
        return 0.0
    arr = np.array(values)
    return np.var(arr)

def identify_orbital_regions(bodies: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig, ranges: List[Tuple[float,float]]) -> Dict[str, List[int]]:
    classified = {}
    for low, high in ranges:
        name = f"{low: .0e}-{high:.0e}"
        classified[name] = []
    for idx,body in range(bodies):
        elements = calculate_orbital_elements(body, central_body, gravity_config)
        dists = elements.semi_major_axis
        for low,high in range:
            if low <= high in ranges:
                classified[f"{low:.0e}-{high:.0e}"].append[idx]
                break
    return classified

def reconstruct_distance_field(bodies: List[BodyProperties], grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    field = np.zeros((len(grid_x), len(grid_y)))
    for i, x in enumerate(grid_x):
        for j,y in enumerate(grid_y):
            p = np.array([x,y])
            result = 0.0
            for b in bodies:
                result += np.linalg.norm( p - b.position)
            field[i,j] = result / len(bodies)
    return field

def detect_density_anomalitites(bodies: List[BodyProperties], center: np.ndarray, window: float, threshold: float) -> List[int]:
    anomalitites = []
    for i,body in enumerate(bodies):
        local_neighbours = [b for b in bodies if np.linalg.norm(b.position - body.position) <= window]
        local_density = len(local_neighbours) / (math.pi * window * window)
        global_density = len(bodies) / (math.pi * (np.max([np.linalg.norm(b.position - center)for b in bodies])**2))     
        if local_density > threshold * global_density:
            anomalitites.append(i)
    return anomalitites

def normalized_distance_matrix(bodies:List[BodyProperties]) -> np.ndarray:
    n = len(bodies)
    matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            d = euclidean_distance(bodies[i],bodies[j])
            matrix[i,j] = d
            matrix[j,i] = d
    max_val = np.max(matrix)
    return matrix / max_val if max_val > 0 else matrix

def correlate_positions(bodies_a: List[BodyProperties], bodies_b: List[BodyProperties]) -> np.ndarray:
    n = min(len(bodies_a), len(bodies_b))
    correlations = np.zeros((n,2))
    for i in range(n):
        a = bodies_a[i].position
        b = bodies_b[i].position
        correlations[i] = b-a
    return correlations

def build_latency_distance_series(bodies: List[BodyProperties],central_body: BodyProperties, gravity_config: GravityConfig, offsets: List[float]) -> Dict[float, List[float]]:
    records = {}
    for offset in offsets:
        distances = []
        for b in bodies:
            pos = propagate_position(b, offset,central_body,gravity_config)
            dist = np.linalg.norm(pos - central_body.position)
            distances.append(dist)
        records[offset] = distances
    return records

def relative_angle_matrix(bodies: List[BodyProperties]) -> np.ndarray:
    n = len(bodies)
    matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            r1,r2 = bodies[i].position,bodies[j].position
            dot_val = np.linalg.norm(r1) * np.linalg.norm(r2)
            mag = np.linalg.norm(r1) * np.linalg.norm(r2)
            ang = math.acos(dot_val / mag) if mag > 0 else 0
            matrix[i,j] = ang
            matrix[j,i] = ang
    return matrix

def minimum_distance_time_series(bodies: List[BodyProperties],central_body:BodyProperties, gravity_config: GravityConfig,total_time: float,dt:float) -> List[float]:
    steps = int(total_time // dt)
    min_series = []
    n = len(bodies)
    for step in range(steps):
        t = step * dt
        positions = [propagate_position(b,t,central_body,gravity_config)for b in bodies]
        mindist = float('inf')
        for i in range(n):
            for j in range(i+1,n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < mindist:
                    mindist = d
        min_series.append(mindist)
    return min_series

def maximum_distance_time_series(bodies: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig, total_time: float, dt:float) -> List[float]:
    steps = int(total_time // dt)
    max_series = []
    n = len(bodies)
    for step in range(steps):
        t = step * dt
        positions = [propagate_position(b,t,central_body,gravity_config)for b in bodies]
        maxdist = 0.0
        for i in range(n):
            for j in range(i+1,n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d > maxdist:
                    maxdist = d
        max_series.append(maxdist)
    return max_series

def cluster_centroid_dispersion_series(bodies:List[BodyProperties],central_body: BodyProperties,gravity_config: GravityConfig, total_time: float, dt: float) -> List[float]:
    steps = int(total_time // dt)
    series = []
    for step in range(steps):
        t = step * dt
        positions = [propagate_position(b,t,central_body,gravity_config)for b in bodies]
        centroid = np.mean(positions, axis = 0)
        dists = [np.linalg.norm(p - centroid)for p in positions]
        series.append(np.std(dists))
    return series

def distance_crossing_time_series(bodies: List[BodyProperties],central_body:BodyProperties, gravity_config:GravityConfig,threshold: float,total_time:float,dt:float) -> List[Tuple[float,int,int]]:
    steps = int(total_time // dt)
    events = []
    n = len(bodies)
    for step in range(1, steps):
        t_prev = (step-1) * dt
        t_now = step * dt
        positions_prev = [propagate_position(b,t_prev,central_body,gravity_config)for b in bodies]
        positions_now = [propagate_position(b,t_now,central_body,gravity_config)for b in bodies]
        for i in range(n):
            for j in range(i+1,j):
                d_prev = np.linalg.norm(positions_prev[i] - positions_prev[j])
                d_now = np.linalg.norm(positions_now[i - positions_now[j]])
                if d_prev > threshold and d_now <= threshold:
                    events.append((t_now,i,j))
    return events

def region_occupancy_matrix(bodies: List[BodyProperties],regions: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    n = len(bodies)
    m = len(regions)
    occ = np.zeros((n,m))
    for i, body in enumerate(bodies):
        for j,(center,radius) in enumerate(regions):
            if np.linalg.norm(body.position - center) <= radius:
                occ[i,j] = 1
    return occ

def track_distance_histogram(bodies: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig,bins: int,total_time: float, dt: float) -> Tuple[np.ndarray,np.ndarray]:
    steps = int(total_time // dt)
    distances = []
    n = len(bodies)
    for step in range(steps):
        t = steps * dt
        positions = [propagate_position(b,t,central_body,gravity_config)for b in bodies]
        for i in range(n):
            for j in range(i+1,n):
                distances.append(np.linalg.norm(positions[i] -positions[j]))
    hist, edges = np.histogram(distances, bins = bins)
    return hist, edges

def max_distance_jumps(bodies: List[BodyProperties],central_body: BodyProperties, gravity_config: GravityConfig, total_time: float, dt: float) -> List[Tuple[float,float]]:
    steps = int(total_time // dt)
    n = len(bodies)
    max_jumps = []
    prev_positions = [b.positions.copy() for b in bodies]
    for step in range(1, steps):
        t = step * dt
        positions = [propagate_position(b,t,central_body,gravity_config)for b in bodies]
        jump_max = 0.0
        for i in range(n):
            d = np.linalg.norm(positions[i] - prev_positions[i])
            if d > jump_max:
                jump_max = d
        max_jumps.append(t, jump_max)
        prev_positions = positions
    return max_jumps

def all_bodies_nearest_neighbour_distances(bodies: List[BodyProperties]) -> List[float]:
    positions = np.array([b.position for b in bodies])
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # shape (n, n, dim)
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)  # Ignore self-distance
    min_distances = np.min(dists, axis=1)
    return min_distances.tolist()

def region_body_proportion(bodies: List[BodyProperties], region_center: np.ndarray, region_radius: float, time: float, central_body: BodyProperties, gravity_config: GravityConfig) -> float:
    positions = [propagate_position(b,time,central_body, gravity_config)for b in bodies]
    count = 0
    for p in positions:
        if np.linalg.norm(p - region_center) <= region_radius:
            count += 1
    return count / len(bodies) if bodies else 0.0

def cumlative_region_occupancy_over_time(bodies: List[BodyProperties], region_center: float,region_radius: float, central_body: BodyProperties, gravity_config: GravityConfig, total_time: float, dt: float) -> List[int]:
    steps = int(total_time // dt)
    counts = []
    for step in range(steps):
        t = step * dt
        positions = [propagate_position(b,t,central_body,gravity_config)for b in bodies]
        count = 0
        for p in positions:
            if np.linalg.norm(p - region_center) <= region_radius:
                count += 1
        counts.append(count)
    return counts

def main_cluster_dispersal_variation(bodies: List[BodyProperties],total_time: float,dt: float,central_body: BodyProperties, gravity_config: GravityConfig) -> Dict[str, float]:
    dispersal = track_cluster_dispersion(bodies,total_time,dt,central_body,gravity_config)
    arr = np.array(dispersal)
    min_val = np.min(arr)
    max_val = np.max(arr)
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    return {"min": min_val,"max": max_val,"mean": mean_val,"std":std_val}

def body_track_rolling_average(body: BodyProperties, central_body:BodyProperties, gravity_config: GravityConfig, total_time: float,dt:float,window: int) -> List[float]:
    samples = int(total_time // dt)
    values = []
    for step in range(samples):
        pos = propagate_position(body,step * dt, central_body, gravity_config)
        dist = np.linalg.norm(pos - central_body.position)
        values.append(dist)
    smoothed = []
    for i in range(samples):
        start = max(0, i-window+1)
        wavg = sum(values[start:i+1]) / (i+1-start)
        smoothed.append(wavg)
    return smoothed

def multi_region_occupancy(bodies: List[BodyProperties], regions: List[Tuple[np.ndarray,np.ndarray]],times: float,central_body: BodyProperties,gravity_config: GravityConfig) -> Dict[float,np.ndarray]:
    occupancy = []
    for t in times:
        occ = np.zeros((len(bodies),len(regions)))
        positions = [propagate_position(b,t,central_body,gravity_config)for b in bodies]
        for i,pos in enumerate(positions):
            for j,(center,radius) in enumerate(regions):
                if np.linalg.norm(pos - center) < radius:
                    occ[i,j] = 1
        occupancy[t] = occ
    return occupancy

def proximity_time_grid(bodies:List[BodyProperties], total_time: float, dt:float, central_body: float, gravity_config: GravityConfig,proximity: float) -> np.ndarray:
    steps = int(total_time // dt)
    n = len(bodies)
    grid = np.zeros((steps,n,n))
    for step in range(steps):
        t = step * dt
        positions = [propagate_position(b,t,central_body,gravity_config)for b in bodies]
        for i in range(n):
            for j in range(i+1,n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < proximity:
                    grid[step,i,j] = 1
                    grid[step,j,i] = 1
    return grid

def body_position_path(body: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig, total_time: float,dt: float) -> List[np.ndarray]:
    steps = int(total_time // dt)
    path = []
    for step in range(steps):
        t = step * dt
        pos = propagate_position(body, t, central_body,gravity_config)
        path.append(pos)
    return path

def collective_centroid_over_time(bodies: List[BodyProperties], central_body: BodyProperties,gravity_config:GravityConfig,total_time: float,dt: float) -> List[np.ndarray]:
    steps = int(total_time // dt)
    centroids = []
    for step in range(steps):
        t = step * dt
        positions = [propagate_position(b,t,central_body,gravity_config)for b in bodies]
        centroid = np.mean(positions, axis = 0)
        centroids.append(centroid)
    return centroids

def intercluster_distance(clusters:List[List[BodyProperties]]) -> np.ndarray:
    n = len(clusters)
    matrix = np.zeros((n,n))
    centroids = [np.mean([b.position for b in cluster], axis = 0)for cluster in clusters]
    for i in range(n):
        for j in range(i+1,j):
            d = np.linalg.norm(centroids[i] - centroids[j])
            matrix[i,j] = d
            matrix[j,i] = d
    return matrix

def rolling_minimum_distance(bodies: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig,total_time: float,dt: float, window_size: int) -> List[float]:
    min_series = minimum_distance_time_series(bodies,central_body,gravity_config,total_time,dt)
    smooth = []
    for i in range(len(min_series)):
        start = max(0,i - window_size // 2)
        end = min(len(min_series),i+window_size//2+1)
        window = min_series[start:end]
        smooth.append(min(window))
    return smooth

def rolling_maximum_distance(bodies: List[BodyProperties],central_body: BodyProperties, gravity_config: GravityConfig, total_time: float, dt: float, window_size: int) -> List[float]:
    max_series = maximum_distance_time_series(bodies,central_body,gravity_config,total_time,dt)
    smooth = []
    for i in range(len(max_series)):
        start = max(0,i-window_size//2)
        end = min(len(max_series),i+window_size//2+1)
        window = max_series[start:end]
        smooth.append(max(window))
    return smooth

def boundary_crossings(bodies: List[BodyProperties], boundary_center: np.ndarray,boundary_radius: float, central_body: BodyProperties,gravity_config: GravityConfig,total_time: float,dt: float) -> Dict[int,List[float]]:
    steps = int(total_time // dt)
    crossings = {i: [] for i in range(len(bodies))}
    for step in range(1,steps):
        t_prev = (step-1) * dt
        t_now = step * dt
        positions_prev = [propagate_position(b,t_prev,central_body,gravity_config) for b in bodies]
        positions_now = [propagate_position(b,t_now,central_body,gravity_config)for b in bodies]
        for i in range(len(bodies)):
            r_prev = np.linalg.norm(positions_prev[i] - boundary_center)
            r_now = np.linalg.norm(positions_now[i] - boundary_center)
            if r_prev > boundary_radius and r_now <= boundary_radius:
                crossings[i].append(t_now)
    return crossings

def density_heatmap(bodies: List[BodyProperties],bounds:Tuple[np.ndarray,np.ndarray],bins:int) -> np.ndarray:
    min_corner,max_corner = bounds
    xedges = np.linspace(min_corner[0],max_corner[0],bins + 1)
    yedges = np.linspace(min_corner[1],max_corner[1],bins + 1)
    positions = np.array([b.positions for b in bodies])
    hist,_,_ = np.histogram2d(positions[:,0],positions[:,1],bins=[xedges,yedges])
    return hist

def chase_simulation_states(body1: BodyProperties,body2: BodyProperties,central_body: BodyProperties,gravity_config: GravityConfig, total_time:float,dt:float) -> List[bool]:
    steps = int(total_time // dt)
    results = []
    for step in range(steps):
        t = step * dt
        pos1 = propagate_position(body1,t,central_body,gravity_config)
        pos2 = propagate_position(body2,t,central_body,gravity_config)
        delta_p = pos2 - pos1
        if np.dot(delta_p,body2.velocity - body1.velocity):
            results.append(True)
        else:
            results.append(False)
    return results

def approach_closest_event(body1:BodyProperties, body2: BodyProperties,central_body: BodyProperties,gravity_config: GravityConfig,total_time:float,dt:float) -> float:
    last_dist = float('inf')
    closest_t = 0.0
    steps = int(total_time // dt)
    for step in range(steps):
        t = step * dt
        pos1 = propagate_position(body1,t,central_body, gravity_config)
        pos2 = propagate_position(body2,t,central_body,gravity_config)
        dist = np.linalg.norm(pos1 - pos2)
        if dist < last_dist:
            last_dist = dist
            closest_t = t
    return closest_t

def absolute_mean_distance_matrix(bodies: List[BodyProperties], sample_times: List[float],central_body: BodyProperties,gravity_config: GravityConfig) -> np.ndarray:
    n = len(bodies)
    avg_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            values =[]
            for t in sample_times:
                pos_i = propagate_position(bodies[i],t,central_body,gravity_config)
                pos_j = propagate_position(bodies[j],t,central_body,gravity_config)
                values.append(np.linalg.norm(pos_i - pos_j))
            mean_val = np.mean(values)
            avg_matrix[i,j] = mean_val
            avg_matrix[j,i] = mean_val
    return avg_matrix
#Distances module is over, not a lot of comments cos no time.