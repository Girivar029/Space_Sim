#This module is to calculate midly accurate collision between different astronomical bodies
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

def classify_collision_type(body1: BodyProperties,body2: BodyProperties) -> str:
    types = {body1.body_type, body2.body_type}
    if "star" in types and "planet" in types:
        return "star-planet"
    if "star" in types and "asteroid" in types:
        return "star-asteroid"
    if "planet" in types and "asteroid" in types:
        return "planet-asteroid"
    if "planet" in types and "moon" in types:
        return "planet-moon"
    if "planet" in types and "planet" in types:
        return "planet-planet"
    if "asteroid" in types and "asteroid" in types:
        return "asteroid-asteroid"
    if "comet" in types and "planet" in types:
        return "comet-planet"
    if "comet" in types and "star" in types:
        return "comet-star"
    return "other"

def catastrophic_fragmentation(body1: BodyProperties, body2: BodyProperties, impact_energy: float) -> list:
    total_mass = body1.mass + body2.mass
    num_fragments = min(int(impact_energy // 1e23),100)
    fragments = []
    for _ in range(num_fragments):
        frag_mass = np.random.uniform(1e18,1e20)
        frag_pos = (body1.position + body2.position) / 2 + np.random.normal(0, body1.radius) / 2
        frag_vel = (body1.velocity + body2.velocity) / 2 + np.random.normal(0, body1.radius,2)
        frag_radius = (frag_mass / total_mass) ** (1.0/3.0) * (body1.radius + body2.radius) / 2
        fragments.append(BodyProperties(frag_pos,frag_vel,frag_mass, frag_radius, "debris"))
    return fragments

def star_planet_collison_process(body1: BodyProperties,body2: BodyProperties, impact_energy: float):
    if body1.body_type == "planet":
        planet,star = body1, body2
    else:
        planet,star = body2, body1
    engulfed = True
    remnant_mass = planet.mass * 0.02
    debris_mass = planet.mass * 0.1
    main_star = BodyProperties(star.position,star.velocity,star.mass + planet.mass - debris_mass,star.radius,"star")
    moon = create_moon(main_star, remnant_mass)
    debris = create_debris(main_star,debris_mass)
    return {"main_star": main_star,"moon": moon,"debris": debris, "engulfed": engulfed}

def planet_planet_collision_process(body1: BodyProperties, body2: BodyProperties,impact_energy: float) -> dict:
    total_mass = body1.mass + body2.mass
    moon_mass = total_mass * 0.03
    debris_mass = total_mass * 0.15
    final_mass = total_mass - moon_mass - debris_mass
    centroid = (body1.position * body1.mass + body2.position * body2.mass) / total_mass
    centroid_vel = velocity_after_inelastic_collision(body1,body2)
    planet_remnant = BodyProperties(centroid,centroid_vel,final_mass,(body1.radius**3 + body2.radius**3)**(1.0/3.0),"planet")
    moon = create_moon(planet_remnant,moon_mass)
    debris = create_debris(planet_remnant,debris_mass)
    return {"main_planet": planet_remnant, "moon": moon, "debris": debris}

def asteroid_collision_process(body1: BodyProperties, body2: BodyProperties, impact_energy:float) -> dict:
    total_mass = body1.mass + body2.mass
    largest_chunk = BodyProperties((body1.position + body2.position) / 2, (body1.velocity + body2.velocity)/2,total_mass * 0.7,max(body1.radius,body2.radius), "asteroid")
    small_fragments = catastrophic_fragmentation(body1, body2,impact_energy)
    return {"largest_chunk": largest_chunk,"fragments": small_fragments}

def planet_moon_collision_process(body1: BodyProperties, body2: BodyProperties,impact_energy:float) -> dict:
    types = [body1.body_type, body2.body_type]
    planet = body1 if body1.body_type == "planet" else body2
    moon = body2 if body2.body_type == "moon" else body1
    debris_mass = moon.mass * 0.5
    absorbed_mass = planet.mass + moon.mass - debris_mass
    new_planet = BodyProperties(planet.position,planet.velocity, absorbed_mass, (planet.radius ** 3 + moon.radius**3)**(1.0/3.0),"planet")
    debris = create_debris(new_planet, debris_mass)
    return {"new_planet": new_planet, "debris": debris}

def comet_planet_collision_process(body1: BodyProperties, body2: BodyProperties, impact_energy: float) -> dict:
    comet = body1 if body1.body_type == "comet" else body2
    planet = body2 if body1.body_type == "comet" else body1
    vaporized_mass = comet.mass * 0
    debris_mass = comet.mass * 0.2
    moon_mass = planet.mass * 0.015
    new_planet = BodyProperties(planet.position,planet.velocity,planet.mass + comet.mass - vaporized_mass - debris_mass,planet.radius,"planet")
    moon = create_moon(new_planet, moon_mass)
    debris = create_debris(new_planet, debris_mass)
    return {"new_planet": new_planet,"moon": moon,"debris": debris}

def advanced_collision_outcome(body1: BodyProperties, body2:BodyProperties, gravity_config: GravityConfig) -> dict:
    impact_vec = body1.velocity - body2.velocity
    impact_velocity = np.linalg.norm(impact_vec)
    impact_energy = 0.5 * (body1.mass + body2.mass) * impact_velocity**2
    ctype = classify_collision_type(body1,body2)
    if ctype == "star-planet":
        return star_planet_collison_process(body1,body2,impact_energy)
    elif ctype == "planet-planet":
        return planet_planet_collision_process(body1,body2)
    elif ctype == "asteroid-asteroid":
        return asteroid_collision_process(body1,body2)
    elif ctype == "planet-moon":
        return planet_moon_collision_process(body1,body2)
    elif ctype == "comet-planet":
        return comet_planet_collision_process(body1,body2)
    else: 
        return collision_outcome(body1,body2,gravity_config,impact_velocity)
    
def process_advanced_collisions(bodies: List[BodyProperties], gravity_config: GravityConfig) -> list:
    n = len(bodies)
    events = detect_physical_collisions(bodies)
    new_bodies = bodies[:]
    handled = set()
    for i,j in events:
        if i in handled or j in handled:
            continue
        b1,b2 = bodies[i],bodies[j]
        outcome = advanced_collision_outcome(b1,b2,gravity_config)
        keys = [k for k in outcome if isinstance(outcome[k], BodyProperties)]
        for k in keys:
            new_bodies.append(outcome[k])
        if "debris" in outcome and isinstance(outcome["debris"],list):
            new_bodies.extend(outcome["debris"])
        handled.add(i)
        handled.add(j)
        new_bodies[i] = None
        new_bodies[j] = None
    return [b for b in new_bodies if b is not None]

def simulate_advanced_collision_steps(bodies: List[BodyProperties], gravity_config: GravityConfig, steps: int = 3) -> list:
    current_bodies = bodies[:]
    for _ in range(steps):
        current_bodies = process_advanced_collisions(current_bodies,gravity_config)
    return current_bodies

def sort_bodies_by_mass(bodies: List[BodyProperties]) -> list:
    return sorted([b for b in bodies if b is not None], key = lambda b:b.mass,reverse=True)

def filter_debris_bodies(bodies: List[BodyProperties]) -> list:
    return [b for b in bodies if getattr(b,'body_type',None) == "debris"]

def filter_moon_bodies(bodies: List[BodyProperties]) -> list:
    return [b for b in bodies if getattr(b,'body_type', None) == "moon"]

def filter_planet_bodies(bodies: List[BodyProperties]) -> list:
    return [b for b in bodies if getattr(b,'body_type',None) == "planet"]

def filter_star_bodies(bodies: List[BodyProperties]) -> list:
    return [b for b in bodies if getattr(b,'body_type', None) == "star"]

def filter_asteroid_bodies(bodies: List[BodyProperties]) -> list:
    return [b for b in bodies if getattr(b,'body_type',None) == "asteroid"]

def filter_comet_bodies(bodies: List[BodyProperties]) -> list:
    return [b for b in bodies if getattr(b,'body_type', None) == "comet"]

def colliding_body_indices(events: list) -> set:
    indices = set()
    for i,j in events:
        indices.add(i)
        indices.add(j)
    return indices

def generate_debris_cloud_from_collision(body: BodyProperties, debris_mass: float, cloud_radius: float = 5.0, n_pieces: int = 20) -> list:
    fragments = []
    for _ in range(n_pieces):
        theta = np.random.uniform(0,2*np.pi)
        r = np.random.uniform(0.5,cloud_radius)
        offset = np.array([r * np.cos(theta),r * np.sin(theta)])
        fragment_mass = debris_mass / n_pieces
        fragment_radius = (fragment_mass / body.mass) ** (1/3) * body.radius * 0.4
        velocity_magnitude = np.random.uniform(500,3500)
        fragment_velocity = body.velocity + offset / np.linalg.norm(offset) * velocity_magnitude
        fragment_position = body.position + offset
        fragments.append(BodyProperties(fragment_position,fragment_velocity,fragment_mass,fragment_radius,"debris"))
    return fragments
    
def system_mass_summary(bodies: List[BodyProperties]) -> dict:
    totals = {"star": 0,"planet": 0, "moon": 0, "asteroid": 0, "comet":0, "debris": 0}
    for b in bodies:
        bt = getattr(b,"body_type",None)
        if bt in totals:
            totals[bt] += b.mass
    return totals

def count_body_types(bodies: List[BodyProperties]) -> dict:
    counts = {"star": 0, "planet": 0, "moon": 0, "asteroid": 0, "comet":0, "debris": 0}
    for b in bodies:
        bt = getattr(b,"body_type",None)
        if bt in counts:
            counts[bt] += 1
        else:
            if "other" not in counts:
                counts["other"] = 0
            counts["other"] += 1
    return counts

def eject_debris_from_system(bodies: List[BodyProperties], escape_velocity: float, central_body: BodyProperties) -> list:
    ejected = []
    for b in bodies:
        if getattr(b, "body_type",None) == "debris":
            rel_vel = np.linalg.norm(b.velocity - central_body.velocity)
            if rel_vel > escape_velocity:
                ejected.append(b)
    return ejected

def merge_debris_into_larger_fragments(debris_list: List[BodyProperties], merge_distance: float = 10000) -> list:
    merged = []
    used = set()
    n = len(debris_list)
    for i in range(n):
        if i in used:
            continue
        mass_sum = debris_list[i].mass
        pos_sum = debris_list[i].position * debris_list[i].mass
        vel_sum = debris_list[i].velocity * debris_list[i].mass
        count = 1
        for j in range(i+1,n):
            if j in used:
                continue
            if np.linalg.norm(debris_list[i].position - debris_list[j].position) < merge_distance:
                mass_sum += debris_list[j].mass
                pos_sum += debris_list[j].position * debris_list[j].mass
                vel_sum += debris_list[j].velocity * debris_list[j].mass
                used.add(j)
                count += 1
        position = pos_sum / mass_sum
        velocity = vel_sum / mass_sum
        radius = (mass_sum)**(1/3)
        merged.append(BodyProperties(position, velocity, mass_sum, radius, "debris"))
        used.add(i)
    return merged

def moon_creation_candidate_fragments(debris_list: List[BodyProperties],mass_threshold: float = 1e22) -> list:
    moon_candidates = []
    for debris in debris_list:
        if debris.mass > mass_threshold:
            new_moon = BodyProperties(debris.position,debris.velocity, debris.mass, debris.radius,"moon")
            moon_candidates.append(new_moon)
    return moon_candidates

def track_body_collision_history(collisions_log: dict,body_index:int, collided_with:int,time:float):
    if body_index not in collisions_log:
        collisions_log[body_index] = []
    collisions_log[body_index].append((collided_with,time))

def sample_collision_energy(body1: BodyProperties, body2: BodyProperties) -> float:
    rel_velocity = np.linalg.norm(body1.velocity - body2.velocity)
    return 0.5 * body1.mass * rel_velocity ** 2

def summarize_collision_event(body1: BodyProperties,body2: BodyProperties,outcome: dict,time:float) -> dict:
    return {
        "time":time,
        "body1_type": getattr(body1,"body_type",None),
        "body2_type": getattr(body2,"body_type",None),
        "outcome": {k: getattr(v, "body_type",None)if isinstance(v, BodyProperties) else None for k,v in outcome.items()}
    }

def collision_chain_summary(collisions_log: dict) -> dict:
    summary = {}
    for body_idx, events in collisions_log.items():
        summary[body_idx] = len(events)
    return summary

def estimate_debris_escape_fraction(debris_list: List[BodyProperties], central_body:BodyProperties, escape_velocity: float) -> float:
    ejected = 0
    for debris in debris_list:
        rel_vel = np.linalg.norm(debris.velocity - central_body.velocity)
        if rel_vel > escape_velocity:
            ejected += 1
    return ejected / len(debris_list) if debris_list else 0.0

def update_after_collisions(bodies: List[BodyProperties],debris_list: List[BodyProperties], collision_moons: list) -> list:
    updated = [b for b in bodies if getattr(b, "body_type", None) != "debris"]
    updated += debris_list
    updated += collision_moons
    return updated

def orbital_energy_post_collision(body: BodyProperties, central_body: BodyProperties,gravity_config: GravityConfig) -> float:
    v = np.linalg.norm(body.velocity - central_body.velocity)
    r = np.linalg.norm(body.position - central_body.position)
    mu = gravity_config.G * (body.mass + central_body.mass)
    return 0.5 * body.mass * v ** 2 - mu * body.mass / r

def loss_of_angular_momentum_in_collision(body1: BodyProperties, body2:BodyProperties) -> float:
    l1 = np.cross(body1.position, body1.velocity) * body1.mass
    l2 = np.cross(body2.position,body2.velocity) * body2.mass
    merged_mass = body1.mass + body2.mass
    merged_velocity = velocity_after_inelastic_collision(body1,body2)
    merged_position = (body1.position * body1.mass + body2.position * body2.mass) / merged_mass
    l_merged = np.cross(merged_position,merged_velocity) * merged_mass
    return np.linalg.norm(l1 + l2 - l_merged)

def record_collision_energy_spectrum(collisions_log:dict,bodies:List[BodyProperties]) -> dict:
    spectrum = {}
    for body_idx, events in collisions_log.items():
        energies = []
        for coll_idx, t in events:
            e = sample_collision_energy(bodies[body_idx],bodies[coll_idx])
            energies.append(e)
            spectrum[body_idx] = energies
    return spectrum

def detect_secondary_moon_formation(debris_list: List[BodyProperties],central_body: BodyProperties, gravity_config: GravityConfig) -> list:
    new_moons = []
    for debris in debris_list:
        v = np.linalg.norm(debris.velocity - central_body.velocity)
        r = np.linalg.norm(debris.position - central_body.position)
        mu = gravity_config.G * (debris.mass + central_body.mass)
        energy = 0.5 * debris.mass * v**2 - mu * debris.mass /r
        if energy < 0.0 and debris.mass > 1e21:
            new_moons.append(BodyProperties(debris.position,debris.velocity,debris.mass, debris.radius, "moon"))
    return new_moons

def system_record_state(bodies: List[BodyProperties]) -> dict:
    return {
        "total_mass": sum(b.mass for b in bodies),
        "count": len(bodies),
        "type_counts": count_body_types(bodies),
        "mass_summary": system_mass_summary(bodies)
    }
#Resting after 1000 lines of work!!!

def log_collision_events(events_log: list, body1_idx:int, body2_idx: int,time: float, collision_type: str,outcome:dict):
    log_entry = {
        "time":time,
        "body1_index":body1_idx,
        "body2_index":body2_idx,
        "collision_type": collision_type,
        "outcome_keys": list(outcome.keys()),
        "body1_type": outcome.get('body1_type',None),
        "body2_type": outcome.get('body2_type',None)
    }
    events_log.append(log_entry)

def update_bodies_post_collision_cycle(bodies: List[BodyProperties], gravity_config: GravityConfig, collision_log: list, collision_history: dict,time:float) -> list:
    events = detect_physical_collisions(bodies)
    new_bodies = bodies[:]
    handled = set()
    for i,j in events:
        if i in handled or j in handled:
            continue
        b1,b2 = bodies[i], bodies[j]
        outcome = advanced_collision_outcome(b1,b2,gravity_config)
        type_descr = classify_collision_type(b1,b2)
        log_collision_events(collision_history,i,j,time)
        track_body_collision_history(collision_history,j,i,time)
        new_entries = []
        keys = [k for k in outcome if isinstance(outcome[k], BodyProperties)]
        for k in keys:
            new_entries.append(outcome[k])
        if "debris" in outcome and isinstance(outcome["debris"],list):
            new_entries.extend(outcome["debris"])
        handled.add[i]
        handled.add[j]
        new_bodies[i] = None
        new_bodies[j] = None
        new_bodies.extend(new_entries)
    return [b for b in new_bodies if b is not None]

#Gotta eat some lines while adding logic
def simulate_collisions_advanced_chain(
        bodies: List[BodyProperties],
        gravity_config: GravityConfig,
        start_time: float,
        end_time: float,
        dt: float
) -> tuple:
    time = start_time
    collision_log = []
    collisions_history = {}
    system_record = []
    states_over_time = []
    current_bodies = bodies[:]
    while time < end_time:
        state = system_record_state(current_bodies)
        system_record.append((time,state))
        states_over_time.append(current_bodies[:])
        next_bodies = update_bodies_post_collision_cycle(current_bodies, gravity_config, collision_log, collisions_history,time)
        if len(next_bodies) == len(current_bodies):
            time += dt
            current_bodies = next_bodies
        else:
            current_bodies = next_bodies
    return current_bodies, collision_log, collisions_history, system_record, states_over_time

def simulate_and_track_system(initial_bodies: List[BodyProperties], gravity_config: GravityConfig,start_time: float, end_time: float, dt: float) -> dict:
    results = {}
    bodies, log, history, masslog, timeline = simulate_collisions_advanced_chain(initial_bodies, gravity_config, start_time,end_time,dt)
    results["final_bodies"] = bodies
    results["collision_log"] = log
    results["history"] = history
    results["system_mass_timeline"] = masslog
    results["states_timeline"] = timeline
    return results

def count_debris_mass_in_system(bodies: List[BodyProperties]) -> float:
    return sum(b.mass for b in bodies if getattr(b, 'body_type', None)== "debris")

def bodies_lost_from_system(bodies_start: List[BodyProperties],bodies_final: List[BodyProperties], central_body: BodyProperties, escape_velocity: float) -> list:
    ids_start = {id(b): b for b in bodies_start}
    ids_final = {id(b): b for b in bodies_final}
    lost = []
    for key in ids_start:
        if key not in ids_final:
            b = ids_start[key]
            if getattr(b, "body_type", None) == "debris":
                rel_speed = np.linalg.norm(b.velocity - central_body.velocity)
                if rel_speed > escape_velocity:
                    lost.append(b)
    return lost

def transform_collision_debris_to_population(debris_list: List[BodyProperties], population_size: int) -> list:
    new_population = []
    for i in range(population_size):
        d = np.random.choice(debris_list)
        fluct = np.random.normal(0,0.2,2)
        position = d.position + fluct
        velocity = d.velocity + np.random.normal(0,100,2)
        mass = d.mass * np.random.uniform(0.8,1.2)
        radius = d.radius * np.random.uniform(0.8,1.2)
        new_population.append(BodyProperties(position,velocity, mass, radius, "debris"))
    return new_population

def run_collisions_and_return_debris(bodies: List[BodyProperties], gravity_config: GravityConfig, steps: int = 10) -> tuple:
    all_debris = []
    all_moons = []
    remaining_bodies = bodies[:]
    for _ in range(steps):
        processed = process_advanced_collisions(remaining_bodies,gravity_config)
        debris = filter_debris_bodies(processed)
        moons = filter_moon_bodies(processed)
        processed = [b for b in processed if b not in (debris + moons)]
        if not debris:
            break
        all_debris.extend(debris)
        all_moons.extend(moons)
        remaining_bodies = processed
    return remaining_bodies, all_debris, all_moons

def assign_unique_ids_to_bodies(bodies: List[BodyProperties],start_id: int = 0) -> dict:
    id_map = {}
    for idx, b in enumerate(bodies):
        setattr(b, "unique_id", start_id + idx)
        id_map[start_id + idx] = b
    return id_map

def combine_collision_results(result_1: dict, result_2: dict) -> dict:
    out = {}
    for key in result_1:
        if isinstance(result_1[key],list) and isinstance(result_2[key],list):
            out[key] = result_1[key] + result_2[key]
        elif isinstance(result_1[key],dict) and isinstance(result_2[key],dict):
            out[key] = {**result_1[key],**result_2[key]}
        else:
            out[key] = result_2[key]
    return out

def mark_bodies_as_fragmented(bodies:List[BodyProperties],indices: list):
    for i in indices:
        if i < len(bodies):
            setattr(bodies[i], "fragmented", True)

def disrupt_body(body: BodyProperties, impact_energy: float,energy_threshold:float) -> bool:
    return impact_energy > energy_threshold

def fragment_body(body: BodyProperties, n_fragments: int, max_velocity_dispersion: float) -> list:
    fragments = []
    base_mass = body.mass / n_fragments
    base_radius = (body.radius**3 / n_fragments)**(1/3)
    for _ in range(n_fragments):
        mass = np.random.uniform(0.8 * base_mass,1.2 * base_mass)
        radius = (mass / body.mass)**(1/3)*body.radius
        angle = np.random.uniform(0,2*np.pi)
        offset = np.array([np.cos(angle), np.sin(angle)]) * body.radius * np.random.uniform(0.1,0.3)
        velocity_dispersion = np.random.uniform(-max_velocity_dispersion, max_velocity_dispersion,2)
        pos = body.position + offset
        vel = body.velocity + velocity_dispersion
        frag = BodyProperties(pos, vel, mass, radius,"debris")
        fragments.append(frag)
    return fragments

def perform_fragmentation_check(bodies: List[BodyProperties], gravity_config: GravityConfig, energy_threshold: float) -> list:
    new_bodies = []
    for b in bodies:
        impact_energy = 0
        if disrupt_body(b, impact_energy,energy_threshold):
            frags = fragment_body(b,n_fragments=10,max_velocity_dispersion=2000)
            new_bodies.extend(frags)
        else:
            new_bodies.append(b)
    return new_bodies

def update_system_mass(bodies: List[BodyProperties]) -> float:
    return sum(b.mass for b in bodies)

def track_mass_loss_over_time(system_states: list) -> list:
    mass_over_time = []
    for time, state in system_states:
        total_mass = update_system_mass(state)
        mass_over_time.append((time,total_mass))
    return mass_over_time

def calculate_total_kinetic_energy(bodies: List[BodyProperties], central_body: BodyProperties) -> float:
    total_ke = 0.0
    for b in bodies:
        rel_vel = np.linalg.norm(b.velocity - central_body.velocity)
        total_ke += 0.5 * b.mass * rel_vel**2
    return total_ke

def calculate_total_potential_energy(bodies: List[BodyProperties], central_body: BodyProperties, gravity_constant: float) -> float:
    total_pe = 0.0
    for b in bodies:
        r = np.linalg.norm(b.position - central_body.position)
        total_pe -= gravity_constant * b.mass * central_body.mass / r if r > 0 else 0
    return total_pe

def energy_budget_report(bodies: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig) -> dict:
    ke = calculate_total_kinetic_energy(bodies, central_body)
    pe = calculate_total_potential_energy(bodies, central_body, gravity_config.G)
    total_energy = ke + pe
    return {"kinetic_energy":ke,"potential_energy":pe,"total_energy": total_energy}

def simulate_energy_conservation(bodies: List[BodyProperties],central_body: BodyProperties, gravity_config: GravityConfig, total_time: float,dt: float) -> list:
    steps = int(total_time // dt)
    energy_log = []
    for step in range(steps):
        t = step * dt
        ke = calculate_total_kinetic_energy(bodies,central_body)
        pe = calculate_total_potential_energy(bodies,central_body,gravity_config.G)
        total_e = ke + pe
        energy_log.append((t,total_e))
    return energy_log

def create_energy_histogram(energies: list, bins:int = 50) -> tuple:
    values = [e[1] for e in energies]
    hist, edges = np.histogram(values, bins = bins)
    return hist, edges

def fragmentation_probabiltiy_curve(velocity: float) -> float:
    if velocity < 500: return 0
    elif velocity < 1000: return 0.1
    elif velocity < 2000: return 0.5
    else: return 0.9

def collision_outcome_probability(body1: BodyProperties, body2: BodyProperties) -> float:
    rel_vel = np.linalg.norm(body1.velocity - body2.velocity)
    return fragmentation_probabiltiy_curve(rel_vel)

def assign_outcome_by_probability(body1: BodyProperties, body2: BodyProperties, gravity_config: GravityConfig) -> dict:
    prob = collision_outcome_probability(body1, body2)
    impact_velocity = np.linalg.norm(body1.velocity - body2.velocity)
    if prob > 0.7:
        return collision_outcome(body1, body2,0.5 * (body1.mass + body2.mass) * impact_velocity**2)
    else:
        return collision_outcome(body1,body2,gravity_config,impact_velocity)

def simulate_probabilistic_collision_outcomes(bodies:List[BodyProperties], gravity_config: GravityConfig, max_steps: int = 10) -> list:
    current_bodies = bodies[:]
    for _ in range(max_steps):
        new_bodies = []
        collisions = detect_physical_collisions(current_bodies)
        handled = set()
        for i, j in collisions:
            if i in handled or j in handled:
                continue
            b1,b2 = current_bodies[i],current_bodies[j]
            outcome = assign_outcome_by_probability(b1,b2,gravity_config)
            final_bodies = []
            for k, val in outcome.items():
                if isinstance(val,list):
                    final_bodies.extend(val)
                elif isinstance(val, BodyProperties):
                    final_bodies.append(val)
            new_bodies.extend(final_bodies)
            handled.add(i)
            handled.add(j)
        new_bodies.extend([b for idx, b in enumerate(current_bodies)if idx not in handled])
        if len(new_bodies) == len(current_bodies):
            break
        current_bodies = new_bodies
    return current_bodies
#Break for a while

def calculate_fragment_escape_velocity(body: BodyProperties, central_body: BodyProperties, gravity_config: GravityConfig) -> float:
    r = np.linalg.norm(body.position - central_body.mass)
    mu = gravity_config.G * (body.mass + central_body.mass)
    return np.sqrt(2 * mu / r)

def filter_escpaing_fragments(debris_list: List[BodyProperties], central_body: BodyProperties, gravity_config: GravityConfig) -> list:
    escaping = []
    for fragment in debris_list:
        v_rel = np.linalg.norm(fragment.velocity - central_body.velocity)
        v_escape = calculate_fragment_escape_velocity(fragment,central_body, gravity_config)
        if v_rel > v_escape:
            escaping.append(fragment)
    return escaping

def calculate_accretion_probability(body1: BodyProperties, body2: BodyProperties, impact_velocity:float) -> float:
    relative_mass = min(body1.mass,body2.mass) / max(body1.mass, body2.mass)
    if impact_velocity < 500:
        return 1.0
    elif impact_velocity < 2000:
        return 0.5 + 0.5 * (1 - relative_mass)
    else:
        return 0.1
    
def simulate_accretion_process(bodies:List[BodyProperties],gravity_config: GravityConfig, accretion_threshold: float = 0.75) -> list:
    n = len(bodies)
    updated_bodies = bodies[:]
    for i in range(n):
        for j in range(i+1,n):
            b1, b2 = updated_bodies[i],updated_bodies[j]
            if b1 is None or b2 is None: continue
            dist = euclidean_distance(b1,b2)
            if dist < b1.radius + b2.radius:
                impact_velocity = np.linalg.norm(b1.velocity - b2.velocity)
                prob = calculate_accretion_probability(b1,b2,impact_velocity)
                if prob > accretion_threshold:
                    merged = merge_bodies(b1,b2)
                    updated_bodies[i] = merged
                    updated_bodies[j] = None
    return [b for b in updated_bodies if b is not None]

def create_collision_remnant(body1: BodyProperties, body2: BodyProperties,impulse: float) -> float:
    total_mass = body1.mass + body2.mass
    new_pos = (body1.position * body1.mass + body2.position * body2.mass)
    new_vel = (body1.velocity * body1.mass + body2.velocity * body2.mass)
    new_radius = (body1.radius**2 + body2.radius**3) ** (1/3)
    return BodyProperties(new_pos, new_vel, total_mass, new_radius, "remnant")

def calculate_impact_angular_momentum(body1: BodyProperties, body2: BodyProperties) -> float:
    r_vec = body1.position - body2.position
    v_vec = body1.velocity - body2.velocity
    return np.linalg.norm(np.cross(r_vec,v_vec)) * min(body1.mass,body2.mass)

def sample_impact_parameter(max_radius: float) -> float:
    return np.random.uniform(0, max_radius)

def simulate_fragment_reaccumalation(fragments: List[BodyProperties], gravity_config: GravityConfig, total_time: float,dt:float) -> list:
    n = len(fragments)
    current_fragments = fragments[:]
    steps = int(total_time // dt)
    for _ in range(steps):
        positions = [f.position for f in current_fragments]
        merged_indices = set()
        for i in range(n):
            for j in range(i+1,n):
                if i in merged_indices or j in merged_indices:
                    continue
                dist = np.linalg.norm(positions[i] - positions[j])
                radius_sum = current_fragments[i].radius + current_fragments[j].radius
                if dist < radius_sum:
                    merged = merge_bodies(current_fragments[i], current_fragments[j])
                    current_fragments[i] = merged
                    current_fragments[j] = None
                    merged_indices.add(j)
        current_fragments = [f for f in current_fragments if f is not None]
        n = len(current_fragments)
    return current_fragments

def calculate_collision_velocity_distribution(collisions:list) -> tuple:
    velocities = [np.linalg.norm(c[1].velocity - c[0].velocity)for c in collisions]
    mean_v = np.mean(velocities) if velocities else 0.0
    std_v = np.std(velocities) if velocities else 0.0
    return mean_v, std_v

def simulate_gravitation_binding_evolution(bodies: List[BodyProperties], gravity_config: GravityConfig, total_time: float, dt: float) -> list:
    bonded_clusters = []
    steps = int(total_time // dt)
    for step in range(steps):
        clusters = []
        visited = set()
        for i,b1 in enumerate(bodies):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j,b2 in enumerate(bodies):
                if j != i and j not in visited:
                    e_kin = 0.5 - b2.mass * np.linalg.norm(b2.velocity - b1.velocity)**2
                    r_dist = np.linalg.norm(b2.position - b1.position)
                    e_pot = gravity_config.G * b1.mass * b2.mass / r_dist if r_dist > 0 else float('inf')
                    if e_kin < e_pot:
                        cluster.append(j)
                        visited.add(j)
            clusters.append(cluster)
        bonded_clusters.append(clusters)
    return bonded_clusters

def assign_masses_to_clusters(clusters: list, bodies: List[BodyProperties]) -> list:
    cluster_masses = []
    for cluster in clusters:
        mass_sum = sum(bodies[i].mass for i in cluster)
        cluster_masses.append(mass_sum)
    return cluster_masses

def merge_clustered_bodies(clusters: list, bodies: List[BodyProperties]) -> list:
    merged_bodies = []
    for cluster in clusters:
        if len(cluster) == 1:
            merge_bodies.append(bodies[cluster[0]])
        else:
            cluster_mass = 0.0
            cluster_pos = np.zeros(2)
            cluster_vel = np.zeros(2)
            for idx in cluster:
                b = bodies[idx]
                cluster_pos += b.position * b.mass
                cluster_vel += b.velocity * b.mass
                cluster_mass += b.mass
            cluster_pos /= cluster_mass
            cluster_vel /= cluster_mass
            cluster_radius = sum(bodies[idx].radius for idx in cluster) / len(cluster)
            merged_body = BodyProperties(cluster_pos,cluster_vel,cluster_mass,cluster_radius,"merged")
            merged_bodies.append(merged_body)
    return merged_bodies