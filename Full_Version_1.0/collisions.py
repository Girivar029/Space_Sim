import numpy as np
from gravity import BodyProperties, GravityConfig
from orbits import calculate_orbital_elements, orbit_radius_from_the_true_anomaly
from distances import euclidean_distance,all_bodies_nearest_neighbour_distances,find_colliding_pairs
from typing import Tuple, List, Optional, Dict
import math

def detect_physical_collisions(bodies:List[BodyProperties]) -> list:
    events = []
    n = len(bodies)
    for i in range(n):
        for j in range(i+1,n):
            r_sum = bodies[i].radius + bodies[j].radius
            if euclidean_distance(bodies[i],bodies[j]) <= r_sum:
                events.append((i,j))
    return events

def velocity_after_elastic_collision(body1:BodyProperties,body2: BodyProperties) -> np.ndarray:
    m1,m2 = body1.mass, body2.mass
    v1,v2 = body1.velocity, body2.velocity
    new_vel = (v1 * (m1 - m2) + 2 * m2 * v2) / (m1 + m2)
    return new_vel

def velocity_after_inelastic_collision(body1: BodyProperties, body2: BodyProperties) -> np.ndarray:
    m1,m2 = body1.mass,body2.mass
    v1,v2 = body1.velocity, body2.velocity
    new_vel = (m1 * v1 + m2 * v2) / (m1 + m2)
    return new_vel

def merge_bodies(body1: BodyProperties, body2: BodyProperties) -> BodyProperties:
    total_mass = body1.mass + body2.mass
    centroid_pos = (body1.position * body1.mass + body2.position * body2.mass) / total_mass
    centroid_vel = velocity_after_inelastic_collision(body1,body2)
    new_radius = (body1.radius**3 + body2.radius**3)**(1.0/3.0)
    return BodyProperties(position=centroid_pos,velocity=centroid_vel,mass=total_mass,radius=new_radius,body_type=body1.body_type)

def collision_outcome(body1: BodyProperties,body2:BodyProperties, gravity_config: GravityConfig, impact_velocity: float) -> dict:
    mass_ratio = min(body1.mass,body2.mass) / max(body1.mass,body2.mass)
    kinetic_energy = 0.5 * body1.mass * impact_velocity**2
    threshold = 1e25 * mass_ratio
    debris_mass = 0.0
    moon_mass = 0.0
    final_body = None
    if kinetic_energy < threshold:
        final_body = merge_bodies(body1,body2)
    else:
        debris_mass = 0.4 * min(body1.mass,body2.mass)
        moon_mass = 0.05 * min(body1.mass,body2.mass)
        main_mass = body1.mass + body2.mass - debris_mass - moon_mass
        position = (body1.position + body2.position)/2
        velocity = (body1.velocity + body2.velocity)/2
        main_radius = (body1.radius**3 + body2.radius**3)**(1.0/3.0)
        final_body = BodyProperties(position=position,velocity=velocity,mass=main_mass,radius=main_radius,body_type=body1.body_type)
    return {
        "main": final_body,
        "debris_mass": debris_mass,
        "moon_mass": moon_mass,
    }

def create_debris(parent_body: BodyProperties,debris_mass: float,n_pieces:int = 10) -> list:
    debris_list = []
    for _ in range(n_pieces):
        frac_mass = debris_mass / n_pieces
        angle = np.random.uniform(0,2*np.pi)
        offset = np.array([np.cos(angle), np.sin(angle)]) * parent_body.radius * 1.5
        vel_mag = np.random.uniform(500,1500)
        velocity = parent_body.velocity + offset / np.linalg.norm(offset) * vel_mag
        radius = (frac_mass / parent_body.mass)** (1.0/3)*parent_body.radius
        debris = BodyProperties(position=parent_body.position+offset,velocity=velocity,mass = frac_mass,radius=radius,body_type=None)
        debris_list.append(debris)
    return debris_list

def create_moon(parent_body: BodyProperties,moon_mass: float) -> BodyProperties:
    angle = np.random.uniform(0,2*np.pi)
    offset = np.array([np.cos(angle),np.sin(angle)]) * parent_body.radius * 3.5
    velocity = parent_body.velocity + np.array([-offset[1], offset[0]]) / np.linalg.norm(offset) * 1000
    radius = (moon_mass / parent_body.mass) ** (1.0/3) * parent_body.radius
    return BodyProperties(position=parent_body.position + offset,velocity=velocity,mass=moon_mass,radius=radius,body_type='moon')

def perform_collisions(bodies: List[BodyProperties], gravity_config: GravityConfig) -> list:
    new_bodies = bodies[:]
    events = detect_physical_collisions(bodies)
    handled = set()
    for i,j in events:
        if i in handled or j in handled:
            continue
        b1,b2 = bodies[i], bodies[j]
        impact_vec = b1.velocity - b2.velocity
        impact_velocity = np.linalg.norm(impact_vec)
        outcome = collision_outcome(b1,b2,gravity_config,impact_velocity)
        final_body = outcome["main"]
        new_bodies[i] = final_body
        new_bodies[j] = None
        if outcome["debris_mass"] > 0.0:
            debris = create_debris(final_body,outcome["debris_mass"])
            new_bodies.extend(debris)
        if outcome["moon_mass"] > 0.0:
            moon = create_moon(final_body,outcome["moon_mass"])
            new_bodies.append(moon)
        handled.add(i)
        handled.add(j)
    return [b for b in new_bodies if b is not None]

def simulate_collisions_step(bodies: List[BodyProperties],gravity_config: GravityConfig) -> list:
    updated_bodies = perform_collisions(bodies,gravity_config)
    return updated_bodies

def detect_and_process_collisions_chain(bodies:List[BodyProperties],gravity_config: GravityConfig, max_chain:int = 5) -> list:
    current_bodies = bodies
    for _ in range(max_chain):
        new_bodies = perform_collisions(current_bodies,gravity_config)
        if len(new_bodies) == len(current_bodies):
            break
        current_bodies = new_bodies
    return current_bodies