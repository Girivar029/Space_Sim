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

    def pick_spectral_class(self) -> str:
        p = random.random()
        cumulative = 0
        for spec_class, prob in self.spectral_distribution:
            cumulative += prob
            if p <= cumulative:
                return spec_class
        return "M"
    
    def spectral_to_mass_radius(self, spectral_class:str) -> Tuple[float,float]:

        mapping = {
            "O": (16,6.6),
            "B": (2.1,1.8),
            "A": (1.7,1.7),
            "F": (1.3,1.3),
            "G": (1.0,1.0),
            "K": (0.8,0.7),
            "M": (0.3,0.4)
        }
        mass_solar, radius_solar = mapping.get(spectral_class,(1.0,1.0))
        mass_kg = mass_solar * 1.989e30
        radius_m = radius_solar * 6.9634e8
        return mass_kg, radius_m
    
    def generate_star(self) -> BodyProperties:
        spectral_class = self.pick_spectral_class()
        mass, radius = self.spectral_to_mass_radius(spectral_class)

        galaxy_size_ly = 100000
        position = np.random.uniform(-galaxy_size_ly * 9.461e15, galaxy_size_ly * 9.461e15, 3)
        body = BodyProperties(
            mass=mass,
            radius=radius,
            position=position,
            velocity=np.zeros(3),
            spectral_type = spectral_class,
            name=f"{spectral_class}-Star-{random.randint(1000,9999)}"
        )
        return body
    
    def generate_galaxy(self):
        self.stars.clear()
        self.albedos.clear()
        self.rotation_periods.clear()
        self.core_radius_ratios.clear()
        self.star_colors.clear()

        for _ in range(self.num_stars):
            star = self.generate_star()
            self.stars.append(star)

            self.albedos.append(0.8)
            self.rotation_periods.append(25 * 86400)
            self.core_radius_ratios.append(0.55)

            color = self.spectral_colors.get(star.spectral_type,(255,255,255))

            self.star_colors.append(color)

        self.simulation_info = SimulationInfo(
            central_body=None,
            bodies=self.stars,
            albedos=self.albedos,
            rotation_periods=self.rotation_periods,
            core_radius_ratios=self.core_radius_ratios
        )
        self.simulation_info.update_all()


    def generate_planets_for_star(self, star: BodyProperties, num_planets: int = 5):

        planets = []
        albedos = []
        rotations = []
        cores = []
        base_distance = star.radius * 10
        distance_increment = 5e10

        for i in range(num_planets):
            mass = np.random.uniform(1e23,5e25)
            radius = np.cbrt((3 * mass) / (4 * np.pi * 5500))

            distance = base_distance + i * distance_increment + np.random.uniform(-1e9,1e9)
            position = np.array([star.position[0] + distance, star.position[1], star.position[2]])
            velocity_magnitude = np.sqrt(G * star.mass / distance)
            velocity = np.array([0.0,velocity_magnitude,0.0])
            planet = BodyProperties(
                mass=mass,
                radius=radius,
                position=position,
                velocity=velocity,
                name=f"Planet-{star.name}-{i}"
            )

            planets.append(planet)
            albedos.append(np.random.uniform(0.1,0.7))
            rotations.append(np.random.uniform(-2e5,2e5))
            cores.append(np.random.uniform(0.3,0.8))
        return planets, albedos, rotations, cores
    
    def generate_galaxy_with_planets(self, max_planets_per_star:int = 5):
        self.orbiting_bodies.clear()
        all_bodies = list(self.stars)
        all_albedos = list(self.albedos)
        all_rotations = list(self.rotation_periods)
        all_cores = list(self.core_radius_ratios)

        for star in self.stars:

            num_planets = np.random.randint(1,max_planets_per_star+1)
            planets, albedos, rotations, cores = self.generate_planets_for_star(star,num_planets)
            self.orbiting_bodies.append(planets)
            all_bodies.extend(planets)
            all_albedos.extend(albedos)
            all_rotations.extend(rotations)
            all_cores.extend(cores)

        self.simulation_info = SimulationInfo(
            central_body=None,
            bodies=all_bodies,
            albedos=all_albedos,
            rotation_periods=all_rotations,
            core_radius_ratios=all_cores
        )
        self.simulation_info.update_all()

    def get_body_color(self, body: BodyProperties):
        if hasattr(body, "spectral_type"):
            return 
        self.spectral_colors.get(body.spectral_body, (255,255,255))
        name = getattr(body, "name", "")
        if "Planet" in name:
            star_name = name.split("-")[1]
            for i,star in enumerate(self.stars):
                return self.star_colors[i]
            return (150,150,150)
        return (200,200,200)
    
    def update_orbits(self, time_step: float):
        for body in self.simulation_info.bodies:
            acceleration = np.zeros(3)
            for other_body in self.simulation_info.bodies:
                if other_body is body:
                    continue

                r_vec = other_body.position - body.position
                r = np.linalg.norm(r_vec)
                if r == 0:
                    continue

                force_dir = r_vec / r
                force_mag = G * other_body.mass / (r ** 2)

                acceleration += force_dir * force_mag
            body.velocity += acceleration * time_step
        for body in self.simulation_info.bodies:
            body.position += body.velocity * time_step

    def get_body_info_by_name(self, name:str):
        for body_info in self.simulation_info.body_infos:
            if body_info.body.name == name:
                return body_info
        return None
    
    def get_all_bodies(self):
        return self.simulation_info.bodies
    
    def print_summary(self):
        print(f"Galaxy with {len(self.stars)} stars, total bodies: {len(self.simulation_info.bodies)}")
        for star in self.stars:
            print(f"Star: {star.name} Mass: {star.mass:.2e}kg Position: {star.position}")
        for idx, body in enumerate(self.simulation_info.bodies):
            if body in self.stars:
                continue
            print(f"Body {idx}: Name: {body.name}, Mass: {body.mass}kg, Position: {body.position}")

    
    def reset(self, seed:int = None):
        self.__init__()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)


class BlackHoleSystem:

    def __init__(self):
        self.black_holes: List[BodyProperties] = []
        self.stars: List[BodyProperties] = []
        self.albedos: List[float] = []
        self.rotation_periods: List[float] = []
        self.core_radius_ratios: List[float]= []
        self.star_colors: List[tuple] = []
        self.simulation_info: SimulationInfo = None
        self.init_constants()
        self.prepare_bodies()
        self.set_orbits()
        self.simulation_info = SimulationInfo(central_body=self.black_holes[0] if self.black_holes else None,
                                              bodies=self.stars + self.black_holes, albedos=self.albedos,
                                              rotation_periods=self.rotation_periods,core_radius_ratios=self.core_radius_ratios,
                                              )
        
        self.simulation_info.update_all()

    def init_constants(self):
        self.black_hole_mass = 1e31
        self.black_hole_radius = 1e4
        self.bh_position = np.array([0.0,0.0,0.0])
        self.star_base_mass = 2e30
        self.star_base_radius = 7e8
        self.galaxy_size = 2e12

    def prepare_bodies(self):
        self.black_holes.append(BodyProperties(mass=self.black_hole_mass, radius=self.black_hole_radius, position=self.bh_position, velocity=np.zeros(3),name="Central Black Hole"))
        self.albedos.append(0,0)
        self.rotation_periods.append(1.0)
        self.core_radius_ratios.append(1.0)
        for i in range(10):
            angle = i * (2 * np.pi / 10)
            distance = 1e11 + i * 2e10
            position = np.array([distance * np.cos(angle),distance * np.sin(angle),0.0])
            velocity_magnitude = np.sqrt(G * self.black_hole_mass / distance)
            velocity = np.array([
                -velocity_magnitude * np.sin(angle),
                velocity_magnitude * np.cos(angle),0.0
            ])
            star = BodyProperties(
                mass=self.star_base_mass * (0.7 + 0.6 * np.random.rand()),
                radius=self.star_base_radius * (0.7 + 0.6 * np.random.rand()),
                position=position,
                velocity=velocity,
                name=f"Star{i}"
            )
            self.stars.append(star)
            self.albedos.append(0.7)
            self.rotation_periods.append(86400 * (0.8 + 0.4 * np.random.rand()))
            self.core_radius_ratios.append(0.45 + 0.1 - np.random.rand())

    def set_orbits(self):
        for star in self.stars:
            r = np.linalg.norm(star.position - self.black_holes[0].position)
            v_mag = np.sqrt(G * self.black_holes[0].mass / r)
            r_vec = star.position - self.black_holes[0].position
            v_dir = np.array([-r_vec[1],r_vec[0],0])
            v_dir /= np.linalg.norm(v_dir)
            star.velocity = v_dir * v_mag