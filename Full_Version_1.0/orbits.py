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


