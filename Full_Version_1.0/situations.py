#Creates situations and creates the custom simulation editor
from typing import List, Dict, Tuple
import numpy as np
from gravity import BodyProperties, G
from informations import SimulationInfo
import random

class SolarSystemScenario:
    def __init__(self):
        self.central_body: BodyProperties = None
        self.planets: List[BodyProperties] = []
        self.albedos: List[float] = []
        self.rotation_periods: List[float] = []
        self.core_radius_ratios: List[float] = []
        self.planet_colors: List[tuple] = []
        self.simulation_info: SimulationInfo = None
        self.init_solar_system_constants()
        self.prepare_bodies()
        self.init_orbital_velocities()
        self.init_simulation_info()

    def init_solar_system_constants(self):
        self.sun_mass = 1.989e30  # kg
        self.sun_radius = 6.9634e8  # m
        self.sun_position = np.array([0.0, 0.0, 0.0])

        # Planet data
        self.planet_data = [
            {"name": "Mercury", "mass": 3.3011e23, "radius": 2.4397e6, "distance": 5.79e10, "rotation_period": 58.646 * 86400},
            {"name": "Venus", "mass": 4.8675e24, "radius": 6.0518e6, "distance": 1.082e11, "rotation_period": -243.025 * 86400},
            {"name": "Earth", "mass": 5.97237e24, "radius": 6.371e6, "distance": 1.496e11, "rotation_period": 0.99726968 * 86400},
            {"name": "Mars", "mass": 6.4171e23, "radius": 3.3895e6, "distance": 2.279e11, "rotation_period": 1.025957 * 86400},
            {"name": "Jupiter", "mass": 1.8982e27, "radius": 6.9911e7, "distance": 7.785e11, "rotation_period": 0.41354 * 86400},
            {"name": "Saturn", "mass": 5.6834e26, "radius": 5.8232e7, "distance": 1.433e12, "rotation_period": 0.44401 * 86400},
            {"name": "Uranus", "mass": 8.6810e25, "radius": 2.5362e7, "distance": 2.877e12, "rotation_period": -0.71833 * 86400},
            {"name": "Neptune", "mass": 1.02413e26, "radius": 2.4622e7, "distance": 4.503e12, "rotation_period": 0.67125 * 86400}
        ]

        self.default_albedos = [0.12, 0.75, 0.30, 0.25, 0.52, 0.47, 0.51, 0.41]
        self.default_core_radius_ratios = [0.42, 0.50, 0.55, 0.52, 0.85, 0.82, 0.72, 0.75]

        # Realistic RGB colors scaled 0-255 for each planet (approximate surface colors)
        self.planet_colors = [
            (169, 143, 122),  # Mercury: grayish brown
            (204, 204, 153),  # Venus: pale yellow
            (70, 130, 180),   # Earth: blue
            (188, 39, 50),    # Mars: reddish
            (216, 180, 130),  # Jupiter: beige
            (210, 180, 140),  # Saturn: light tan
            (150, 200, 230),  # Uranus: pale cyan
            (50, 70, 220),    # Neptune: deep blue
        ]

    def prepare_bodies(self):
        self.central_body = BodyProperties(
            mass=self.sun_mass,
            radius=self.sun_radius,
            position=self.sun_position,
            velocity=np.zeros(3),
            name="Sun"
        )
        self.planets = []
        self.albedos = []
        self.rotation_periods = []
        self.core_radius_ratios = []

        for i, pdata in enumerate(self.planet_data):
            position = np.array([pdata["distance"], 0.0, 0.0])
            velocity = np.zeros(3)  # Will be set for orbital velocity
            body = BodyProperties(
                mass=pdata["mass"],
                radius=pdata["radius"],
                position=position,
                velocity=velocity,
                name=pdata["name"]
            )
            self.planets.append(body)
            self.albedos.append(self.default_albedos[i])
            self.rotation_periods.append(pdata["rotation_period"])
            self.core_radius_ratios.append(self.default_core_radius_ratios[i])

    def init_orbital_velocities(self):
        # Calculate circular orbital velocities perpendicular to radius vector
        for planet in self.planets:
            r_vec = planet.position - self.central_body.position
            r = np.linalg.norm(r_vec)
            v_magnitude = np.sqrt(G * self.central_body.mass / r)
            # velocity vector perpendicular to radius vector in XY plane (clockwise)
            vel_direction = np.array([-r_vec[1], r_vec[0], 0])
            vel_direction /= np.linalg.norm(vel_direction)
            planet.velocity = vel_direction * v_magnitude

    def init_simulation_info(self):
        self.simulation_info = SimulationInfo(
            central_body=self.central_body,
            bodies=self.planets,
            albedos=self.albedos,
            rotation_periods=self.rotation_periods,
            core_radius_ratios=self.core_radius_ratios,
        )
        self.simulation_info.update_all()

    def get_planet_color(self, planet_name: str):
        for i, pdata in enumerate(self.planet_data):
            if pdata["name"] == planet_name:
                return self.planet_colors[i]
        return (255, 255, 255)  # default white

    def get_body_infos(self):
        return self.simulation_info.body_infos


class RandomGalaxyScenario:

    def __init__(self, num_stars: int = 1000, seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_stars = num_stars
        self.stars: List[BodyProperties] = []
        self.orbiting_bodies:List[List[BodyProperties]] = []

        self.spectral_colors: Dict[str, Tuple[int,int,int]] = {
            "O": (155,176,255),
            "B": (170,191,255),
            "A": (202,215,255),
            "F": (248,247,255),
            "G": (255,244,234),
            "K": (255,210,161),
            "M": (255,204,111)
        }

        self.spectral_distribution = [
            ("O",0.00003),
            ("B",0.0013),
            ("A", 0.006),
            ("F", 0.03),
            ("G", 0.076),
            ("K", 0.121),
            ("M", 0.7645)
        ]

        self.simulation_info: SimulationInfo = None
        self.albedos: List[float] = []
        self.rotation_periods: List[float] = []
        self.core_radius_ratios: List[float] = []
        self.star_colors: List[Tuple[int,int,int]] = []

        self.generate_galaxy()
#Aiming to finish by 2nd November and as quick as possible!