#This module takes care of all the physics which will be done for the simulation in real-time.
#Also the most important part of the program turning this module into a highly reusable physics engine for low-end computers to run.
import numpy as np
from typing import List,Dict,Tuple,Optional
from dataclasses import dataclass, field
from enum import Enum

from gravity import (
    BodyProperties, GravityConfig, calculate_distance_vector, newtonian_gravity_force, post_newtonian_correction,
    calculate_gravitational_potential,calculate_system_energy, calculate_center_of_mass,calculate_angular_momrntum,
    GravityIntegrator,G,AU)

from orbits import (
    calculate_orbital_elements,OrbitalElements,propagate_orbit,calculate_orbital_velocity,
    solve_kepler,calculate_orbit_state_from_elements, orbit_radius_from_the_true_anomaly)

from distances import (
    euclidean_distance, find_colliding_pairs, all_bodies_nearest_neighbour_distances,distances_matrix,
    find_close_pairs
)

from collisions import (
    detect_physical_collisions, process_advanced_collisions, collision_outcome, system_mass_summary,
    count_body_types, system_record_state,advanced_collision_outcome,track_body_collision_history,
    merge_debris_into_larger_fragments,detect_secondary_moon_formation,calculate_total_kinetic_energy,calculate_total_potential_energy,
    classify_collision_type
)

class SimulateMode(Enum):
    """Execution modes, like normal speed, fast forward, slow time and recording"""
    REAL_TIME = "real_time"
    FAST_FORWARD = "fast_forward"
    STEP_BY_STEP = "step_by_step"
    RECORDING = "recording"

#Locking in today as I skipped school today to work for a long time!!!

@dataclass
class PhysicsState:
    """All state related functions of the sim"""
    time: float
    bodies: List[BodyProperties]
    total_energy: float
    total_momentum: np.ndarray
    angular_momentum: np.ndarray
    center_of_mass: np.ndarray
    collision_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "time":self.time,
            "body_count":len(self.bodies),
            "total_energy": self.total_energy,
            "collision_count": self.collision_count
        }
    
class CentralPhysicsEngine:
    """Manages gravity, orbital mechanics and updating body, in sense all physics calculations"""

    def __init__(self, gravity_config:GravityConfig,dt:float = 1.0):
        self.gravity_config = gravity_config
        self.dt = dt
        self.time = 0.0
        self.bodies: List[BodyProperties] = []
        self.integrator = GravityIntegrator(gravity_config)

    def add_body(self, body: BodyProperties):
        self.bodies.append(body)

    def add_bodies(self, bodies: List[BodyProperties]):
        self.bodies.extend(bodies)

    def remove_body(self, index:int):
        if 0 <= index < len(self.bodies):
            self.bodies.pop(index)

    def calculate_gravitational_forces(self) -> np.ndarray:
        n = len(self.bodies)
        forces = np.zeros((n,2))

        if n < 2:
            return forces
        
        positions = np.array([b.position for b in self.bodies])
        masses = np.array([b.mass for b in self.bodies])

        for i in range(n):
            for j in range(i+1,n):
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                if r == 0:
                    continue

                force_mag = G * masses[i] * masses[j] / (r**2)
                force_vec = force_mag * r_vec / r
                forces[i] += force_vec
                forces[j] -= force_vec

        return forces
    
    def update_velocities(self, forces: np.ndarray):
        for i, body in enumerate(self.bodies):
            acceleration = forces[i] / body.mass
            body.velocity += acceleration * self.dt
            
    def update_position(self):
        for body in self.bodies:
            body.position += body.velocity * self.dt

    def step(self):
        forces = self.calculate_gravitational_forces()
        self.update_velocities(forces)
        self.update_position()
        self.time += self.dt

    def get_state(self) -> PhysicsState:
        total_energy = calculate_system_energy(self.bodies)
        com, com_vel = calculate_center_of_mass(self.bodies)

        total_angular_momentum = np.zeros(3)
        for body in self.bodies:
            r = np.append(body.position,0.0)
            v = np.append(body.velocity,0.0)
            L = body.mass * np.cross(r,v)
            total_angular_momentum += L


        return PhysicsState(
            time=self.time,
            bodies=self.bodies[:],
            total_energy=total_energy,
            total_momentum=com_vel * sum(b.mass for b in self.bodies),
            angular_momentum=total_angular_momentum,
            center_of_mass=com
        )
    
class OrbitalMechanicsManager:
    """Manages orbital calculations just as the name suggests"""

    def __init__(self,central_body: BodyProperties, gravity_config: GravityConfig):
        self.central_body = central_body
        self.gravity_config = gravity_config
        self.orbital_elements_cache: Dict[int, OrbitalElements] = {}

    def compute_orbital_elements(self, body: BodyProperties, body_id: int) -> OrbitalElements:
        elements = calculate_orbital_elements(body,self.central_body,self.gravity_config)
        self.orbital_elements_cache[body_id] = elements

        return elements
    
    def compute_all_orbital_elements(self,bodies: List[BodyProperties]) -> List[OrbitalElements]:
        elements_list = []
        for i, body in enumerate(bodies):
            elements = self.compute_orbital_elements(body,i)
            elements_list.append(elements)

        return elements_list
    
    def propogate_orbit_by_time(self, body: BodyProperties, time: float) ->BodyProperties:
        elements = calculate_orbital_elements(body,self.central_body,self.gravity_config)
        
        mean_motion = np.sqrt(G * self.central_body.mass / elements.semi_major_axis ** 3)

        mean_anomaly = elements.mean_anomaly + mean_motion * time

        eccentric_anomaly = solve_kepler(mean_anomaly,elements.eccentricity)

        true_anamoly = 2 * np.arctan2(np.sqrt(1 + elements.eccentricity) * np.sin(eccentric_anomaly / 2),
                                      np.sqrt(1 - elements.eccentricity) * np.cos(eccentric_anomaly / 2)
                                      )
        
        new_state = calculate_orbit_state_from_elements(elements, true_anamoly)

        body.position = new_state[0]
        body.velocity = new_state[1]

        return body
    
    def get_orbital_period(self, body: BodyProperties) -> float:
        elements = calculate_orbital_elements(body, self.central_body, self.gravity_config)
        return 2 * np.pi * np.sqrt(elements.semi_major_axis ** 3 / (G * self.central_body.mass))
    
    def compute_orbit_at_phase(self, elements:OrbitalElements, phase: float) -> Tuple[np.ndarray,np.ndarray]:
        return calculate_orbit_state_from_elements(elements,phase)
    
    def find_resonances(self, bodies: List[BodyProperties], tolerance: float = 0.05) -> List[Tuple[int,int,float]]:
        resonances = []
        periods = []

        for body in bodies:
            period = self.get_orbital_period(body)
            periods.append(period)

        n = len(periods)
        for i in range(n):
            for j in range(i+1,j):
                if periods[i] == 0 or periods[j] == 0:
                    continue

                ratio = periods[i] / periods[j]

                for p in range(1,10):
                    for q in range(1,10):
                        if abs(p / q - ratio) < tolerance:
                            resonances.append((i,j,p/q))
                            break

        return resonances
    
    def clear_cache(self):
        self.orbital_elements_cache.clear()

class DistanceProximityManager:
    """Calculated Distances."""

    def __init__(self):
        self.proximity_threshold = 1e10
        self.collision_pairs_history: List[List[Tuple[int,int]]] = []
        self.close_encounter_log: List[Dict] = []

    def compute_distance_matrix(self, bodies: List[BodyProperties]) -> np.ndarray:
        positions = np.array([b.position for b in bodies])
        diff = positions[:, np.newaxis,:] - positions[np.newaxis,:,:]
        matrix = np.linalg.norm(diff, axis = -1)
        return matrix
    
    def find_all_close_pairs(self, bodies: List[BodyProperties], threshold: float) -> List[Tuple[int,int,float]]:
        n = len(bodies)
        pairs = []
        dist_matrix = self.compute_distance_matrix(bodies)

        for i in range(n):
            for j in range(i+1,n):
                if dist_matrix[i,j] < threshold:
                    pairs.append((i,j,dist_matrix[i,j]))

        return pairs
    
    def detect_collision_candidates(self, bodies:List[BodyProperties]) -> List[Tuple[int,int]]:
        collision_pairs = find_colliding_pairs(bodies)

        self.collision_pairs_history.append(collision_pairs)

        return collision_pairs
    
    def track_close_approach(self, bodies: List[BodyProperties], time: float):
        close_pairs = self.find_all_close_pairs(bodies,self.proximity_threshold)

        for i,j, dist in close_pairs:
            encounter = {
                "time": time,
                "body1_index": i,
                "body2_index":j,
                "distance": dist,
                "relative_velocity": np.linalg.norm(bodies[i].velocity - bodies[j].velocity)
            }

            self.close_encounter_log.append(encounter)

    def compute_nearest_neigbours(self, bodies: List[BodyProperties],reference_point: np.ndarray,radius: float) -> float:
        count = 0
        for body in bodies:
            if np.linalg.norm(body.position - reference_point) <= radius:
                count += 1

        volume = (4/3) * np.pi * radius ** 3

        return count / volume if volume > 0 else 0
    
    def identify_clusters(self,bodies:List[BodyProperties], cluster_radius: float) -> List[List[int]]:
        n = len(bodies)
        visited = set()
        clusters = []

        dist_matrix = self.compute_distance_matrix(bodies)

        for i in range(n):
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)

            for j in range(n):
                if j not in visited and dist_matrix[i,j] < cluster_radius:
                    cluster.append(j)
                    visited.add(j)

            clusters.append(cluster)

        return clusters
    
    def get_encounter_statistics(self) -> Dict:
        if not self.close_encounter_log:
            return {"total_encounters": 0}
        
        distances = [e["distance"]for e in self.close_encounter_log]
        velocities = [e["relative_velocity"] for e in self.close_encounter_log]

        return {
            "total_encounters":len(self.close_encounter_log),
            "min_distance": min(distances),
            "max_distance": max(distances),
            "mean_distance": np.mean(distances),
            "mean_velocity": np.mean(velocities)
        }
    
    def clear_history(self):
        self.collision_pairs_history.clear()
        self.close_encounter_log.clear()

class CollisionHandler:

    def __init__(self, gravity_config: GravityConfig):
        self.gravity_config = gravity_config
        self.collision_history: Dict[int,List[Tuple[int,float]]] = {}
        self.debris_generated: List[BodyProperties] = []
        self.moons_created: List[BodyProperties] = []
        self.merged_bodies: List[BodyProperties] = []

    def detect_collisions(self, bodies: List[BodyProperties]) -> List[Tuple[int,int]]:
        return detect_physical_collisions(bodies)
    
    def process_collision_pair(self, body1: BodyProperties, body2: BodyProperties, body1_idx:int,body2_idx:int,time:float) -> Dict:
        impact_velocity = np.linalg.norm(body1.velocity - body2.velocity)
        outcome = advanced_collision_outcome(body1,body2,self.gravity_config)

        track_body_collision_history(self.collision_history,body1_idx,body2_idx, time)
        track_body_collision_history(self.collision_history,body2_idx,body1_idx,time)

        return outcome
    
    def handled_all_collisions(self, bodies: List[BodyProperties],time:float) -> List[BodyProperties]:
        collision_pairs = self.detect_collisions(bodies)

        if not collision_pairs:
            return bodies
        
        new_bodies = bodies[:]
        handled = set()

        for i,j in collision_pairs:
            if i in handled or i in handled:
                continue

            b1,b2 = bodies[i], bodies[j]
            outcome = self.process_collision_pair(b1,b2,i,j,time)

            result_bodies = []
            for key, val in outcome.items():
                if isinstance(val, BodyProperties):
                    result_bodies.append(val)
                    if key == "main" or "main_" in key:
                        self.merged_bodies.append(val)
                    elif key == "moon":
                        self.moons_created.append(val)
                elif isinstance(val,List):
                    for item in val:
                        if isinstance(item,BodyProperties):
                            result_bodies.append(item)
                            if getattr(item,'body_type',None) == 'debris':
                                self.debris_generated.append(item)
            new_bodies[i] = None
            new_bodies[j] = None
            new_bodies.extend(result_bodies)

            handled.add(i)
            handled.add(j)

        return [b for b in new_bodies if b is not None]

    def simulate_debris_evolution(self,debris_list: List[BodyProperties],central_body: BodyProperties,escape_velocity:float) -> Tuple[List[BodyProperties],List[BodyProperties]]:
        escaping = []
        remaining = []

        for debris in debris_list:
            v_rel = np.linalg.norm(debris.velocity - central_body.velocity)

            if v_rel > escape_velocity:
                escaping.append(debris)
            else:
                remaining.append(debris)
        
        return remaining, escaping
    
    def merge_nearby_debris(self, debris_list:List[BodyProperties],merge_distance:float) -> List[BodyProperties]:
        return merge_debris_into_larger_fragments(debris_list,merge_distance)
    
    def detect_secondary_moons(self, debris_list: List[BodyProperties],central_body: BodyProperties) -> List[BodyProperties]:
        return  detect_secondary_moon_formation(debris_list,central_body,self.gravity_config)
    
    def get_collision_statistics(self) -> Dict:
        return {
            "total_collisions": sum(len(events) for events in self.collision_history.values()),
            "debris_count": len(self.debris_generated),
            "moons_created":len(self.moons_created),
            "mergers": len(self.merged_bodies),
            "total_debris_mass": sum(d.mass for d in self.debris_generated)
        }
    
    def clear_collision_data(self):
        self.collision_history.clear()
        self.debris_generated.clear()
        self.moons_created.clear()
        self.merged_bodies.clear()


class EnergyMomentumMonitor:

    def __init__(self):
        self.energy_history: List[Tuple[float,float]] = []
        self.momentum_history: List[Tuple[float,np.ndarray]] = []
        self.angular_momentum_history: List[Tuple[float,np.ndarray]] = []

    def record_energy(self, time: float, bodies: List[BodyProperties],central_body: BodyProperties, gravity_config:GravityConfig):
        ke = calculate_total_kinetic_energy(bodies, central_body)
        pe = calculate_total_potential_energy(bodies,central_body)
        total_energy = ke + pe

        self.energy_history.append((time,total_energy))

    def record_momentum(self,time: float, bodies: List[BodyProperties]):
        total_momentum = np.zeros(2)

        for body in bodies:
            total_momentum += body.mass * body.velocity

        self.momentum_history.append((time,total_momentum))

    def record_angular_momentum(self, time: float, bodies: List[BodyProperties]):

        total_L = np.zeros(3)

        for body in bodies:
            r = np.append(body.position,0.0)
            v = np.append(body.velocity, 0.0)
            L = body.mass * np.cross(r,v)
            total_L += L

        self.angular_momentum_history.append((time,total_L))

    def compute_energy_conservation_error(self) -> float:
        if len(self.energy_history) < 2:
            return 0.0
        
        initial_energy = self.energy_history[0][1]
        final_energy = self.energy_history[-1][1]

        if initial_energy == 0:
            return 0.0
        
        return abs((final_energy - initial_energy) / initial_energy)
    
    def compute_momentum_conservation_error(self) -> float:
        if len(self.momentum_history) < 2:
            return 0.0
        
        initial_momentum = np.linalg.norm(self.momentum_history[0][1])
        final_momentum = np.linalg.norm(self.momentum_history[-1][1])

        if initial_momentum == 0:
            return 0.0
        
        return abs((final_momentum - initial_momentum) / initial_momentum)
    
    def get_energy_report(self) -> Dict:
        if not self.energy_history:
            return {}
        
        energies = [e[1] for e in self.energy_history]

        return {
            "initial_energy": energies[0],
            "final_energy": energies[-1],
            "mean_energy": np.mean(energies),
            "energy_drift": energies[-1] - energies[0],
            "conservation_error": self.compute_energy_conservation_error()
        }
    
    def clear_history(self):
        self.energy_history.clear()
        self.momentum_history.clear()
        self.angular_momentum_history.clear()

class EventLogger:

    def __init__(self):
        self.events: List[Dict] = []
        self.collision_events: List[Dict] = []
        self.orbital_events: List[Dict] = []

    def log_event(self, event_type:str, time: float, data: Dict):
        event = {
            "type": event_type,
            "time": time,
            "data": data
        }

        self.events.append(event)

    def log_collision(self, time: float, body1_idx:int, body2_idx: int, collision_type: str, outcome: Dict):

        event = {
            "time": time,
            "body1": body1_idx,
            "body2": body2_idx,
            "type": collision_type,
            "outcome_keys": list(outcome.keys())
            }
        
        self.collision_events.append(event)

    def log_orbital_event(self, time: float, body_idx: int, event_type: str, data: Dict):

        event = {
            "time": time,
            "body": body_idx,
            "event": event_type,
            "data": data
        }

        self.orbital_events.append(event)

    def get_events_in_range(self, start_time: float, end_time: float) -> List[Dict]:
        return [e for e in self.events if start_time >= e["time"] <= end_time]
    
    def get_collision_count(self) -> int:
        return len(self.collision_events)
    
    def export_events(self) -> Dict:
        return {
            "all_events": self.events,
            "collisions": self.collision_events,
            "orbital_events": self.orbital_events
        }
    
    def clear_logs(self):
        self.events.clear()
        self.collision_events.clear()
        self.orbital_events.clear()


class SimulationOrchestrator:
    
    def __init__(self, gravity_config: GravityConfig, dt: float = 1.0):
        self.physics_engine = CentralPhysicsEngine(gravity_config, dt)
        self.orbital_manager = None
        self.distance_manager = DistanceProximityManager()
        self.collision_handler = CollisionHandler(gravity_config)
        self.energy_monitor = EnergyMomentumMonitor()
        self.event_logger = EventLogger()
        
        self.simulation_mode = SimulateMode.REAL_TIME
        self.is_running = False
        self.current_step = 0
        self.max_steps = 0
        
    def initialize(self, bodies: List[BodyProperties], central_body: Optional[BodyProperties] = None):
        self.physics_engine.add_bodies(bodies)
        
        if central_body:
            self.orbital_manager = OrbitalMechanicsManager(central_body, self.physics_engine.gravity_config)
        
        self.is_running = True
        self.current_step = 0
    
    def step(self):
        if not self.is_running:
            return
        
        current_time = self.physics_engine.time
        bodies = self.physics_engine.bodies
        
        self.distance_manager.track_close_approach(bodies, current_time)
        
        collision_pairs = self.collision_handler.detect_collisions(bodies)
        if collision_pairs:
            bodies = self.collision_handler.handle_all_collisions(bodies, current_time)
            self.physics_engine.bodies = bodies
            
            for i, j in collision_pairs:
                collision_type = classify_collision_type(bodies[i], bodies[j]) if i < len(bodies) and j < len(bodies) else "unknown"
                self.event_logger.log_collision(current_time, i, j, collision_type, {})
        
        self.physics_engine.step()
        
        if self.orbital_manager and self.orbital_manager.central_body:
            self.energy_monitor.record_energy(current_time, bodies, self.orbital_manager.central_body, self.physics_engine.gravity_config)
        
        self.energy_monitor.record_momentum(current_time, bodies)
        self.energy_monitor.record_angular_momentum(current_time, bodies)
        
        self.current_step += 1
    
    def run(self, num_steps: int):
        self.max_steps = num_steps
        
        for _ in range(num_steps):
            self.step()
            
            if not self.is_running:
                break
    
    def run_until_time(self, target_time: float):
        while self.physics_engine.time < target_time and self.is_running:
            self.step()
    
    def pause(self):
        self.is_running = False
    
    def resume(self):
        self.is_running = True
    
    def reset(self):
        self.physics_engine = CentralPhysicsEngine(self.physics_engine.gravity_config, self.physics_engine.dt)
        self.distance_manager.clear_history()
        self.collision_handler.clear_collision_data()
        self.energy_monitor.clear_history()
        self.event_logger.clear_logs()
        
        if self.orbital_manager:
            self.orbital_manager.clear_cache()
        
        self.current_step = 0
        self.is_running = False
    
    def get_current_state(self) -> PhysicsState:
        return self.physics_engine.get_state()
    
    def get_bodies(self) -> List[BodyProperties]:
        return self.physics_engine.bodies
    
    def get_simulation_statistics(self) -> Dict:
        stats = {
            "current_time": self.physics_engine.time,
            "current_step": self.current_step,
            "body_count": len(self.physics_engine.bodies),
            "collision_stats": self.collision_handler.get_collision_statistics(),
            "encounter_stats": self.distance_manager.get_encounter_statistics(),
            "energy_report": self.energy_monitor.get_energy_report(),
            "total_events": len(self.event_logger.events)
        }
        return stats
    
    def set_time_step(self, dt: float):
        self.physics_engine.dt = dt
    
    def set_simulation_mode(self, mode: SimulateMode):
        self.simulation_mode = mode


class HistoryTracker:
    
    def __init__(self, snapshot_interval: int = 10):
        self.snapshot_interval = snapshot_interval
        self.state_snapshots: List[PhysicsState] = []
        self.body_trajectories: Dict[int, List[Tuple[float, np.ndarray]]] = {}
        self.velocity_history: Dict[int, List[Tuple[float, np.ndarray]]] = {}
        self.mass_history: Dict[int, List[Tuple[float, float]]] = {}
        
    def record_snapshot(self, state: PhysicsState):
        self.state_snapshots.append(state)
    
    def record_body_state(self, body_id: int, time: float, position: np.ndarray, velocity: np.ndarray, mass: float):
        if body_id not in self.body_trajectories:
            self.body_trajectories[body_id] = []
            self.velocity_history[body_id] = []
            self.mass_history[body_id] = []
        
        self.body_trajectories[body_id].append((time, position.copy()))
        self.velocity_history[body_id].append((time, velocity.copy()))
        self.mass_history[body_id].append((time, mass))
    
    def get_trajectory(self, body_id: int) -> List[Tuple[float, np.ndarray]]:
        return self.body_trajectories.get(body_id, [])
    
    def get_position_at_time(self, body_id: int, time: float) -> Optional[np.ndarray]:
        trajectory = self.body_trajectories.get(body_id, [])
        
        for t, pos in trajectory:
            if abs(t - time) < 1e-6:
                return pos
        
        return None
    
    def get_velocity_evolution(self, body_id: int) -> List[float]:
        if body_id not in self.velocity_history:
            return []
        
        return [np.linalg.norm(v) for t, v in self.velocity_history[body_id]]
    
    def get_mass_evolution(self, body_id: int) -> List[Tuple[float, float]]:
        return self.mass_history.get(body_id, [])
    
    def compute_trajectory_length(self, body_id: int) -> float:
        trajectory = self.body_trajectories.get(body_id, [])
        
        if len(trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(trajectory)):
            pos1 = trajectory[i - 1][1]
            pos2 = trajectory[i][1]
            total_length += np.linalg.norm(pos2 - pos1)
        
        return total_length
    
    def export_trajectory_data(self, body_id: int) -> Dict:
        return {
            "positions": self.body_trajectories.get(body_id, []),
            "velocities": self.velocity_history.get(body_id, []),
            "masses": self.mass_history.get(body_id, [])
        }
    
    def clear_history(self):
        self.state_snapshots.clear()
        self.body_trajectories.clear()
        self.velocity_history.clear()
        self.mass_history.clear()


class OutputManager:
    
    def __init__(self, output_directory: str = "./output"):
        self.output_directory = output_directory
        self.frame_count = 0
        
    def export_state_to_dict(self, state: PhysicsState) -> Dict:
        bodies_data = []
        for body in state.bodies:
            bodies_data.append({
                "position": body.position.tolist(),
                "velocity": body.velocity.tolist(),
                "mass": body.mass,
                "radius": body.radius,
                "body_type": getattr(body, 'body_type', None)
            })
        
        return {
            "time": state.time,
            "total_energy": state.total_energy,
            "total_momentum": state.total_momentum.tolist(),
            "angular_momentum": state.angular_momentum.tolist(),
            "center_of_mass": state.center_of_mass.tolist(),
            "collision_count": state.collision_count,
            "bodies": bodies_data
        }
    
    def export_simulation_summary(self, orchestrator: SimulationOrchestrator) -> Dict:
        stats = orchestrator.get_simulation_statistics()
        state = orchestrator.get_current_state()
        
        return {
            "simulation_info": {
                "total_time": state.time,
                "total_steps": orchestrator.current_step,
                "time_step": orchestrator.physics_engine.dt,
                "mode": orchestrator.simulation_mode.value
            },
            "statistics": stats,
            "final_state": self.export_state_to_dict(state)
        }
    
    def generate_visualization_data(self, bodies: List[BodyProperties]) -> Dict:
        positions = [b.position.tolist() for b in bodies]
        radii = [b.radius for b in bodies]
        masses = [b.mass for b in bodies]
        types = [getattr(b, 'body_type', 'unknown') for b in bodies]
        
        return {
            "positions": positions,
            "radii": radii,
            "masses": masses,
            "types": types,
            "count": len(bodies)
        }
    
    def create_snapshot_frame(self, state: PhysicsState) -> Dict:
        frame = {
            "frame_id": self.frame_count,
            "time": state.time,
            "data": self.export_state_to_dict(state)
        }
        self.frame_count += 1
        return frame
    
    def reset_frame_count(self):
        self.frame_count = 0


class ScalabilityManager:
    
    def __init__(self):
        self.chunk_size = 1000
        self.parallel_enabled = False
        self.optimization_level = 1
        
    def partition_bodies(self, bodies: List[BodyProperties], num_partitions: int) -> List[List[BodyProperties]]:
        partition_size = len(bodies) // num_partitions
        partitions = []
        
        for i in range(num_partitions):
            start_idx = i * partition_size
            end_idx = start_idx + partition_size if i < num_partitions - 1 else len(bodies)
            partitions.append(bodies[start_idx:end_idx])
        
        return partitions
    
    def enable_spatial_hashing(self, cell_size: float):
        self.spatial_cell_size = cell_size
        self.spatial_hash_enabled = True
    
    def compute_spatial_hash(self, bodies: List[BodyProperties]) -> Dict[Tuple[int, int], List[int]]:
        spatial_hash = {}
        
        for idx, body in enumerate(bodies):
            cell_x = int(body.position[0] / self.spatial_cell_size)
            cell_y = int(body.position[1] / self.spatial_cell_size)
            cell = (cell_x, cell_y)
            
            if cell not in spatial_hash:
                spatial_hash[cell] = []
            
            spatial_hash[cell].append(idx)
        
        return spatial_hash
    
    def optimize_collision_detection(self, bodies: List[BodyProperties]) -> List[Tuple[int, int]]:
        spatial_hash = self.compute_spatial_hash(bodies)
        candidate_pairs = []
        
        for cell, indices in spatial_hash.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    candidate_pairs.append((indices[i], indices[j]))
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor_cell = (cell[0] + dx, cell[1] + dy)
                    if neighbor_cell in spatial_hash:
                        for i in indices:
                            for j in spatial_hash[neighbor_cell]:
                                if i < j:
                                    candidate_pairs.append((i, j))
        
        return candidate_pairs
    
    def set_optimization_level(self, level: int):
        self.optimization_level = max(0, min(3, level))


class Physics:
    
    def __init__(self, gravity_config: GravityConfig, central_body: Optional[BodyProperties] = None, dt: float = 1.0):
        self.orchestrator = SimulationOrchestrator(gravity_config, dt)
        self.history_tracker = HistoryTracker()
        self.output_manager = OutputManager()
        self.scalability_manager = ScalabilityManager()
        self.central_body = central_body
        
        self.orchestrator.initialize([], central_body)
    
    def add_bodies(self, bodies: List[BodyProperties]):
        self.orchestrator.physics_engine.add_bodies(bodies)
    
    def load_bodies(self, bodies: List[BodyProperties]):
        self.orchestrator.reset()
        self.orchestrator.initialize(bodies, self.central_body)
    
    def advance_time(self, num_steps: int = 1):
        for _ in range(num_steps):
            self.orchestrator.step()
            
            state = self.orchestrator.get_current_state()
            time = state.time
            bodies = state.bodies
            
            for idx, body in enumerate(bodies):
                self.history_tracker.record_body_state(idx, time, body.position, body.velocity, body.mass)
            
            self.history_tracker.record_snapshot(state)
    
    def run_for_duration(self, duration: float, max_steps: Optional[int] = None):
        steps = max_steps if max_steps else int(duration / self.orchestrator.physics_engine.dt)
        self.orchestrator.run(steps)
        
        for state in self.history_tracker.state_snapshots:
            pass
    
    def change_simulation_mode(self, mode: SimulateMode):
        self.orchestrator.set_simulation_mode(mode)
    
    def set_time_step(self, dt: float):
        self.orchestrator.set_time_step(dt)
    
    def get_bodies(self) -> List[BodyProperties]:
        return self.orchestrator.get_bodies()
    
    def get_state(self) -> PhysicsState:
        return self.orchestrator.get_current_state()
    
    def get_statistics(self) -> Dict:
        return self.orchestrator.get_simulation_statistics()
    
    def output_simulation_summary(self) -> Dict:
        return self.output_manager.export_simulation_summary(self.orchestrator)
    
    def output_current_frame(self) -> Dict:
        state = self.orchestrator.get_current_state()
        return self.output_manager.create_snapshot_frame(state)
    
    def enable_spatial_hashing(self, cell_size: float):
        self.scalability_manager.enable_spatial_hashing(cell_size)
    
    def partition_bodies_for_parallelism(self, num_partitions: int) -> List[List[BodyProperties]]:
        bodies = self.get_bodies()
        return self.scalability_manager.partition_bodies(bodies, num_partitions)
    
    def optimize_collision_detection(self) -> List[Tuple[int, int]]:
        bodies = self.get_bodies()
        return self.scalability_manager.optimize_collision_detection(bodies)
    
    def clear_all_data(self):
        self.orchestrator.reset()
        self.history_tracker.clear_history()
        self.output_manager.reset_frame_count()
        self.scalability_manager.set_optimization_level(1)
    
    def summarize_energy_momentum(self) -> Dict:
        return {
            "energy_report": self.orchestrator.energy_monitor.get_energy_report(),
            "collision_stats": self.orchestrator.collision_handler.get_collision_statistics(),
            "close_encounters": self.orchestrator.distance_manager.get_encounter_statistics()
        }
    
    def track_event(self, event_type: str, time: float, data: Dict):
        self.orchestrator.event_logger.log_event(event_type, time, data)
    
    def export_events(self) -> Dict:
        return self.orchestrator.event_logger.export_events()
    


if __name__ == "__main__":
    print("===Physics Module Test ===\n")
    
    gravity_config = GravityConfig()
    
    sun = BodyProperties(
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.0, 0.0]),
        mass=1.989e30,
        radius=6.96e8,
        body_type="star"
    )
    
    earth = BodyProperties(
        position=np.array([1.496e11, 0.0]),
        velocity=np.array([0.0, 29780.0]),
        mass=5.972e24,
        radius=6.371e6,
        body_type="planet"
    )
    
    mars = BodyProperties(
        position=np.array([2.279e11, 0.0]),
        velocity=np.array([0.0, 24070.0]),
        mass=6.39e23,
        radius=3.389e6,
        body_type="planet"
    )
    
    asteroid = BodyProperties(
        position=np.array([3.5e11, 0.0]),
        velocity=np.array([0.0, 18000.0]),
        mass=1e20,
        radius=5e5,
        body_type="asteroid"
    )
    
    print("Initializing Physics System...")
    physics = Physics(gravity_config, central_body=sun, dt=86400.0)
    
    bodies = [earth, mars, asteroid]
    physics.add_bodies(bodies)
    
    print(f"Bodies loaded: {len(physics.get_bodies())}")
    print(f"Initial state: {physics.get_state().to_dict()}\n")
    
    print("Running simulation for 365 days (365 steps)...")
    physics.advance_time(num_steps=365)
    
    print("\n=== Simulation Results ===")
    stats = physics.get_statistics()
    
    print(f"\nFinal time: {stats['current_time'] / 86400:.2f} days")
    print(f"Total steps: {stats['current_step']}")
    print(f"Final body count: {stats['body_count']}")
    
    print("\n=== Energy Report ===")
    energy_report = stats.get('energy_report', {})
    if energy_report:
        print(f"Initial energy: {energy_report.get('initial_energy', 0):.3e} J")
        print(f"Final energy: {energy_report.get('final_energy', 0):.3e} J")
        print(f"Energy drift: {energy_report.get('energy_drift', 0):.3e} J")
        print(f"Conservation error: {energy_report.get('conservation_error', 0):.6f}")
    
    print("\n=== Collision Statistics ===")
    collision_stats = stats.get('collision_stats', {})
    print(f"Total collisions: {collision_stats.get('total_collisions', 0)}")
    print(f"Debris generated: {collision_stats.get('debris_count', 0)}")
    print(f"Moons created: {collision_stats.get('moons_created', 0)}")
    print(f"Mergers: {collision_stats.get('mergers', 0)}")
    
    print("\n=== Close Encounters ===")
    encounter_stats = stats.get('encounter_stats', {})
    print(f"Total encounters: {encounter_stats.get('total_encounters', 0)}")
    if encounter_stats.get('total_encounters', 0) > 0:
        print(f"Min distance: {encounter_stats.get('min_distance', 0):.3e} m")
        print(f"Mean distance: {encounter_stats.get('mean_distance', 0):.3e} m")
    
    print("\n=== Final Body Positions ===")
    final_bodies = physics.get_bodies()
    for i, body in enumerate(final_bodies):
        print(f"Body {i}: pos={body.position}, vel_mag={np.linalg.norm(body.velocity):.2f} m/s")
    
    print("\n=== Energy & Momentum Summary ===")
    summary = physics.summarize_energy_momentum()
    print(summary)
    
    print("\n=== Exporting Simulation Data ===")
    sim_summary = physics.output_simulation_summary()
    print(f"Simulation summary generated with {len(sim_summary)} keys")
    
    frame_data = physics.output_current_frame()
    print(f"Frame data: frame_id={frame_data['frame_id']}, time={frame_data['time']:.2f}s")
    
    print("\n=== Test with Collision Scenario ===")
    physics.clear_all_data()
    
    body1 = BodyProperties(
        position=np.array([0.0, 0.0]),
        velocity=np.array([1000.0, 0.0]),
        mass=1e22,
        radius=1e6,
        body_type="asteroid"
    )
    
    body2 = BodyProperties(
        position=np.array([5e6, 0.0]),
        velocity=np.array([-1000.0, 0.0]),
        mass=1e22,
        radius=1e6,
        body_type="asteroid"
    )
    
    physics.load_bodies([body1, body2])
    print(f"Collision test: {len(physics.get_bodies())} bodies loaded")
    
    print("Running collision simulation for 100 steps...")
    physics.advance_time(num_steps=100)
    
    collision_stats_final = physics.get_statistics()['collision_stats']
    print(f"\nCollision test results:")
    print(f"Collisions detected: {collision_stats_final.get('total_collisions', 0)}")
    print(f"Final body count: {physics.get_statistics()['body_count']}")
    print(f"Debris created: {collision_stats_final.get('debris_count', 0)}")
    
    print("\n=== Physics Module Test Complete ===")
