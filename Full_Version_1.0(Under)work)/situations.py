from typing import List, Optional
import numpy as np
from gravity import BodyProperties, G
from informations import SimulationInfo


class BlackHoleSystem:
    def __init__(self):
        self.black_holes: List[BodyProperties] = []
        self.stars: List[BodyProperties] = []
        self.albedos: List[float] = []
        self.rotation_periods: List[float] = []
        self.core_radius_ratios: List[float] = []
        self.star_colors: List[tuple] = []
        self.simulation_info: SimulationInfo = None
        self.init_constants()
        self.prepare_bodies()
        self.set_orbits()
        self.simulation_info = SimulationInfo(
            central_body=self.black_holes[0] if self.black_holes else None,
            bodies=self.stars + self.black_holes,
            albedos=self.albedos,
            rotation_periods=self.rotation_periods,
            core_radius_ratios=self.core_radius_ratios,
        )
        self.simulation_info.update_all()

    def init_constants(self):
        self.black_hole_mass = 1e31
        self.black_hole_radius = 1e4
        self.bh_position = np.array([0.0, 0.0, 0.0])
        self.star_base_mass = 2e30
        self.star_base_radius = 7e8
        self.galaxy_size = 2e12

    def prepare_bodies(self):
        self.black_holes.append(BodyProperties(
            mass=self.black_hole_mass,
            radius=self.black_hole_radius,
            position=self.bh_position,
            velocity=np.zeros(3),
            oody_type="blackhole"
        ))
        self.albedos.append(0.0)
        self.rotation_periods.append(1.0)
        self.core_radius_ratios.append(1.0)
        for i in range(10):
            angle = i * (2 * np.pi / 10)
            distance = 1e11 + i * 2e10
            position = np.array([
                distance * np.cos(angle),
                distance * np.sin(angle),
                0.0
            ])
            velocity_magnitude = np.sqrt(
                G * self.black_hole_mass / distance
            )
            velocity = np.array([
                -velocity_magnitude * np.sin(angle),
                velocity_magnitude * np.cos(angle),
                0.0
            ])
            star = BodyProperties(
                mass=self.star_base_mass * (0.7 + 0.6 * np.random.rand()),
                radius=self.star_base_radius * (0.7 + 0.6 * np.random.rand()),
                position=position,
                velocity=velocity,
                body_type="star"
            )
            self.stars.append(star)
            self.albedos.append(0.7)
            self.rotation_periods.append(86400 * (0.8 + 0.4 * np.random.rand()))
            self.core_radius_ratios.append(0.45 + 0.1 * np.random.rand())

    def set_orbits(self):
        for star in self.stars:
            r = np.linalg.norm(star.position - self.black_holes[0].position)
            v_mag = np.sqrt(G * self.black_holes[0].mass / r)
            r_vec = star.position - self.black_holes[0].position
            v_dir = np.array([-r_vec[1], r_vec[0], 0])
            v_dir /= np.linalg.norm(v_dir)
            star.velocity = v_dir * v_mag


    def add_secondary_black_holes(self, num_secondary=2):
        for j in range(num_secondary):
            angle = j * (2 * np.pi / num_secondary)
            distance = self.galaxy_size * (0.5 + 0.3 * np.random.rand())
            pos = np.array([
                distance * np.cos(angle),
                distance * np.sin(angle),
                0.0
            ])
            mass = self.black_hole_mass * (0.7 + 0.5 * np.random.rand())
            radius = self.black_hole_radius * (1 + 0.4 * np.random.rand())
            velocity = np.zeros(3)
            bh = BodyProperties(
                mass=mass,
                radius=radius,
                position=pos,
                velocity=velocity,
                body_type="blackhole"
            )
            self.black_holes.append(bh)
            self.albedos.append(0.0)
            self.rotation_periods.append(1.0)
            self.core_radius_ratios.append(1.0)

    def add_planets_to_stars(self, planets_per_star=2):
        planets = []
        for i, star in enumerate(self.stars):
            for k in range(planets_per_star):
                mass = np.random.uniform(1e24, 8e25)
                radius = np.cbrt((3 * mass) / (4 * np.pi * 5500))
                distance = np.linalg.norm(star.position - self.black_holes[0].position) * (1 + 0.15 * (k + 1))
                angle = np.random.uniform(0, 2 * np.pi)
                planet_pos = np.array([
                    star.position[0] + distance * np.cos(angle),
                    star.position[1] + distance * np.sin(angle),
                    0.0
                ])
                v_magnitude = np.sqrt(G * star.mass / distance)
                v_dir = np.array([-np.sin(angle), np.cos(angle), 0])
                velocity = v_dir * v_magnitude
                planet = BodyProperties(
                    mass=mass,
                    radius=radius,
                    position=planet_pos,
                    velocity=velocity,
                    body_type="planet"
                )
                planets.append(planet)
                self.albedos.append(np.random.uniform(0.1, 0.6))
                self.rotation_periods.append(np.random.uniform(10e3, 8e5))
                self.core_radius_ratios.append(np.random.uniform(0.3, 0.75))
        self.stars.extend(planets)

    def update_bodies(self):
        all_bodies = self.black_holes + self.stars
        self.simulation_info = SimulationInfo(
            central_body=self.black_holes[0] if self.black_holes else None,
            bodies=all_bodies,
            albedos=self.albedos,
            rotation_periods=self.rotation_periods,
            core_radius_ratios=self.core_radius_ratios,
        )
        self.simulation_info.update_all()

    def advance_orbits(self, time_step: float):
        for body in self.simulation_info.bodies:
            acc = np.zeros(3)
            for other in self.simulation_info.bodies:
                if body == other:
                    continue
                r_vec = other.position - body.position
                r = np.linalg.norm(r_vec)
                if r == 0:
                    continue
                dir = r_vec / r
                mag = G * other.mass / (r ** 2)
                acc += dir * mag
            body.velocity += acc * time_step
        for body in self.simulation_info.bodies:
            body.position += body.velocity * time_step

    def get_body_info_by_name(self, name: str):
        infos = self.simulation_info.body_infos
        for info in infos:
            if info.body.name == name:
                return info
        return None

    def summary(self):
        print("Black hole system summary")
        print(f"Black holes: {len(self.black_holes)} stars/planets: {len(self.stars)} bodies: {len(self.simulation_info.bodies)}")
        for bh in self.black_holes:
            print(f"BlackHole: {bh.name} Mass:{bh.mass:.2e} Pos:{bh.position}")
        for s in self.stars:
            print(f"Body: {s.name}, Mass:{s.mass:.2e} Pos:{s.position}")

    def reset(self):
        self.black_holes.clear()
        self.stars.clear()
        self.albedos.clear()
        self.rotation_periods.clear()
        self.core_radius_ratios.clear()
        self.simulation_info = None
        self.init_constants()
        self.prepare_bodies()
        self.set_orbits()
        self.simulation_info = SimulationInfo(
            central_body=self.black_holes[0] if self.black_holes else None,
            bodies=self.stars + self.black_holes,
            albedos=self.albedos,
            rotation_periods=self.rotation_periods,
            core_radius_ratios=self.core_radius_ratios,
        )
        self.simulation_info.update_all()

class SupernovaScenario:
    def __init__(self):
        self.progenitor_star: BodyProperties = None
        self.remnant_shells: List[BodyProperties] = []
        self.albedos: List[float] = []
        self.rotation_periods: List[float] = []
        self.core_radius_ratios: List[float] = []
        self.simulation_info: SimulationInfo = None
        self.init_progenitor()
        self.init_remnant_shells()
        self.init_simulation_info()

    def init_progenitor(self):
        mass = 1.5e31
        radius = 1.2e10
        position = np.array([0.0, 0.0, 0.0])
        velocity = np.zeros(3)
        self.progenitor_star = BodyProperties(mass, radius, position, velocity, name="Progenitor Star")
        self.albedos.append(0.7)
        self.rotation_periods.append(2.5e5)
        self.core_radius_ratios.append(0.7)

    def init_remnant_shells(self):
        shell_count = 30
        expansion_radius_start = 1.0e10
        expansion_radius_end = 3.0e11
        for i in range(shell_count):
            radius = expansion_radius_start + i * (expansion_radius_end - expansion_radius_start) / shell_count
            mass = self.progenitor_star.mass * 0.01
            position = np.array([radius, 0.0, 0.0])
            velocity = np.array([1.5e4, 0.0, 0.0])
            shell = BodyProperties(mass, radius * 0.1, position, velocity, name=f"Shell{i}")
            self.remnant_shells.append(shell)
            self.albedos.append(0.3)
            self.rotation_periods.append(1e6)
            self.core_radius_ratios.append(0.5)

    def init_simulation_info(self):
        bodies = [self.progenitor_star] + self.remnant_shells
        self.simulation_info = SimulationInfo(
            central_body=self.progenitor_star,
            bodies=bodies,
            albedos=self.albedos,
            rotation_periods=self.rotation_periods,
            core_radius_ratios=self.core_radius_ratios
        )
        self.simulation_info.update_all()

    def explode(self):
        for shell in self.remnant_shells:
            shell.velocity[0] *= 1.05
            shell.position += shell.velocity

    def update(self, timestep: float):
        self.explode()
        self.simulation_info.update_all()

    def get_body_infos(self) -> List[BodyProperties]:
        return self.simulation_info.body_infos
    
    def simulate_remnant_cooling(self, time_step: float):
        cooling_rate = 1e-5
        for shell in self.remnant_shells:
            if hasattr(shell, 'surface_temperature'):
                shell.surface_temperature = max(0, shell.surface_temperature - cooling_rate * time_step)

    def fragment_shells(self):
        new_shells = []
        for shell in self.remnant_shells:
            if np.random.rand() < 0.01:
                mass_split = shell.mass * 0.5
                shell.mass *= 0.5
                offset = np.random.uniform(-shell.radius, shell.radius, 3)
                new_pos = shell.position + offset
                new_vel = shell.velocity * (0.5 + np.random.rand())
                new_shell = BodyProperties(mass_split, shell.radius, new_pos, new_vel, name=f"{shell.name}_frag")
                new_shells.append(new_shell)
        if new_shells:
            self.remnant_shells.extend(new_shells)
            self.simulation_info.bodies.extend(new_shells)
            self.albedos.extend([0.3]*len(new_shells))
            self.rotation_periods.extend([1e6]*len(new_shells))
            self.core_radius_ratios.extend([0.5]*len(new_shells))

    def reset(self):
        self.__init__()

class MultiBlackHoleSystem:
    def __init__(self, num_black_holes=3):
        self.num_black_holes = num_black_holes
        self.black_holes: List[BodyProperties] = []
        self.partner_bodies: List[BodyProperties] = []
        self.albedos: List[float] = []
        self.rotation_periods: List[float] = []
        self.core_radius_ratios: List[float] = []
        self.simulation_info: SimulationInfo = None
        self.init_black_holes()
        self.init_companion_bodies()
        self.assign_orbits()
        self.init_simulation_info()

    def init_black_holes(self):
        base_mass = 5e31
        base_radius = 1e4
        distance_factor = 5e11
        self.black_holes.clear()
        self.albedos.clear()
        self.rotation_periods.clear()
        self.core_radius_ratios.clear()
        angle_increment = 2 * np.pi / self.num_black_holes
        for i in range(self.num_black_holes):
            angle = i * angle_increment
            pos = np.array([distance_factor * np.cos(angle), distance_factor * np.sin(angle), 0])
            mass = base_mass * (1 + 0.5 * np.random.rand())
            radius = base_radius * (1 + 0.5 * np.random.rand())
            vel_mag = np.sqrt(self.G() * base_mass * self.num_black_holes / distance_factor)
            vel_dir = np.array([-np.sin(angle), np.cos(angle), 0])
            velocity = vel_dir * vel_mag
            bh = BodyProperties(
                mass=mass,
                radius=radius,
                position=pos,
                velocity=velocity,
                body_type="blackhole"
            )
            self.black_holes.append(bh)
            self.albedos.append(0.0)
            self.rotation_periods.append(1)
            self.core_radius_ratios.append(1)

    def init_companion_bodies(self):
        self.partner_bodies.clear()
        for idx, bh in enumerate(self.black_holes):
            num_companions = np.random.randint(1,4)
            for c in range(num_companions):
                angle = np.random.uniform(0, 2*np.pi)
                distance = bh.radius * 100 * (c+1)
                pos = bh.position + np.array([distance * np.cos(angle), distance * np.sin(angle), 0])
                mass = np.random.uniform(1e29, 5e30)
                radius = np.random.uniform(1e7, 4e7)
                velocity = np.zeros(3)
                companion = BodyProperties(
                    mass=mass, radius=radius, position=pos, velocity=velocity, name=f"Companion{idx}_{c}"
                )
                self.partner_bodies.append(companion)
                self.albedos.append(0.4)
                self.rotation_periods.append(1e5)
                self.core_radius_ratios.append(0.5)

    def assign_orbits(self):
        bodies = self.black_holes + self.partner_bodies
        for body in bodies:
            if body in self.black_holes:
                continue
            closest_bh = min(self.black_holes, key=lambda bh: np.linalg.norm(bh.position - body.position))
            r_vec = body.position - closest_bh.position
            r = np.linalg.norm(r_vec)
            v_mag = np.sqrt(self.G() * closest_bh.mass / r)
            v_dir = np.array([-r_vec[1], r_vec[0], 0])
            if np.linalg.norm(v_dir) == 0:
                continue
            v_dir /= np.linalg.norm(v_dir)
            body.velocity = v_dir * v_mag

    def simulate_step(self, dt: float):
        bodies = self.black_holes + self.partner_bodies
        accelerations = {body: np.zeros(3) for body in bodies}
        for i, body in enumerate(bodies):
            for j, other in enumerate(bodies):
                if i == j:
                    continue
                r_vec = other.position - body.position
                r = np.linalg.norm(r_vec)
                if r == 0:
                    continue
                force_dir = r_vec / r
                force_mag = self.G() * other.mass / (r ** 2)
                accelerations[body] += force_dir * force_mag
        for body in bodies:
            body.velocity += accelerations[body] * dt
        for body in bodies:
            body.position += body.velocity * dt

    def init_simulation_info(self):
        all_bodies = self.black_holes + self.partner_bodies
        self.simulation_info = SimulationInfo(
            central_body=None,
            bodies=all_bodies,
            albedos=self.albedos,
            rotation_periods=self.rotation_periods,
            core_radius_ratios=self.core_radius_ratios
        )
        self.simulation_info.update_all()

    def G(self):
        return 6.67430e-11

    def reset(self):
        self.__init__(self.num_black_holes)

    def add_black_hole(self, mass: float, radius: float, position: np.ndarray, velocity: np.ndarray, name: str):
        bh = BodyProperties(mass, radius, position, velocity, name)
        self.black_holes.append(bh)
        self.albedos.append(0.0)
        self.rotation_periods.append(1.0)
        self.core_radius_ratios.append(1.0)

    def add_companion_body(self, mass: float, radius: float, position: np.ndarray, velocity: np.ndarray, name: str):
        body = BodyProperties(mass, radius, position, velocity, name)
        self.partner_bodies.append(body)
        self.albedos.append(0.4)
        self.rotation_periods.append(1e5)
        self.core_radius_ratios.append(0.5)

    def get_black_holes(self):
        return self.black_holes

    def get_companion_bodies(self):
        return self.partner_bodies

    def get_all_bodies(self):
        return self.black_holes + self.partner_bodies

    def print_system_info(self):
        print(f"MultiBlackHoleSystem with {len(self.black_holes)} black holes and {len(self.partner_bodies)} companions.")
        for bh in self.black_holes:
            print(f"Black Hole {bh.name} Mass: {bh.mass} Position: {bh.position}")
        for body in self.partner_bodies:
            print(f"Companion {body.name} Mass: {body.mass} Position: {body.position}")

    def simulate_interactions(self, timestep: float):
        bodies = self.get_all_bodies()
        accelerations = {b: np.zeros(3) for b in bodies}
        for i, b1 in enumerate(bodies):
            for j, b2 in enumerate(bodies):
                if i == j:
                    continue
                r_vec = b2.position - b1.position
                r = np.linalg.norm(r_vec)
                if r < 1e3:
                    continue
                acc_val = self.G() * b2.mass / (r ** 2)
                acc_vec = r_vec / r * acc_val
                accelerations[b1] += acc_vec
        for b in bodies:
            b.velocity += accelerations[b] * timestep
        for b in bodies:
            b.position += b.velocity * timestep

    def merge_close_black_holes(self, threshold_distance: float = 1e7):
        merged = []
        bodies = self.get_all_bodies()
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                b1, b2 = bodies[i], bodies[j]
                dist = np.linalg.norm(b1.position - b2.position)
                if dist < threshold_distance:
                    new_mass = b1.mass + b2.mass
                    new_pos = (b1.position * b1.mass + b2.position * b2.mass) / new_mass
                    new_vel = (b1.velocity * b1.mass + b2.velocity * b2.mass) / new_mass
                    new_radius = max(b1.radius, b2.radius) * 1.2  # Increased radius post merger
                    merged_bh = BodyProperties(new_mass, new_radius, new_pos, new_vel, name=f"Merged_{b1.name}_{b2.name}")
                    merged.append((b1, b2, merged_bh))
        for b1, b2, merged_bh in merged:
            if b1 in self.black_holes:
                self.black_holes.remove(b1)
            if b2 in self.black_holes:
                self.black_holes.remove(b2)
            if b1 in self.partner_bodies:
                self.partner_bodies.remove(b1)
            if b2 in self.partner_bodies:
                self.partner_bodies.remove(b2)
            self.add_black_hole(merged_bh.mass, merged_bh.radius, merged_bh.position, merged_bh.velocity, merged_bh.name)

    def run_simulation_step(self, timestep: float):
        self.simulate_interactions(timestep)
        self.merge_close_black_holes()
        self.update_simulation_info()

    def update_simulation_info(self):
        all_bodies = self.get_all_bodies()
        self.simulation_info = SimulationInfo(
            central_body=None,
            bodies=all_bodies,
            albedos=self.albedos,
            rotation_periods=self.rotation_periods,
            core_radius_ratios=self.core_radius_ratios
        )
        self.simulation_info.update_all()

class CustomScenario:
    def __init__(self):
        self.bodies: List[BodyProperties] = []
        self.albedos: List[float] = []
        self.rotation_periods: List[float] = []
        self.core_radius_ratios: List[float] = []
        self.simulation_info: Optional[SimulationInfo] = None

    def add_body(self, mass: float, radius: float, position: np.ndarray, velocity: np.ndarray,
                 albedo: float = 0.3, rotation_period: float = 86400, core_radius_ratio: float = 0.5,
                 name: str = "CustomBody"):
        body = BodyProperties(mass, radius, position, velocity, name)
        self.bodies.append(body)
        self.albedos.append(albedo)
        self.rotation_periods.append(rotation_period)
        self.core_radius_ratios.append(core_radius_ratio)

    def remove_body_by_name(self, name: str):
        index = self._find_body_index_by_name(name)
        if index is not None:
            self.bodies.pop(index)
            self.albedos.pop(index)
            self.rotation_periods.pop(index)
            self.core_radius_ratios.pop(index)

    def update_body_velocity(self, name: str, velocity: np.ndarray):
        index = self._find_body_index_by_name(name)
        if index is not None:
            self.bodies[index].velocity = velocity

    def update_body_position(self, name: str, position: np.ndarray):
        index = self._find_body_index_by_name(name)
        if index is not None:
            self.bodies[index].position = position

    def _find_body_index_by_name(self, name: str) -> Optional[int]:
        for i, b in enumerate(self.bodies):
            if b.name == name:
                return i
        return None

    def initialize_simulation_info(self):
        central_body = self.bodies[0] if self.bodies else None
        self.simulation_info = SimulationInfo(
            central_body=central_body,
            bodies=self.bodies,
            albedos=self.albedos,
            rotation_periods=self.rotation_periods,
            core_radius_ratios=self.core_radius_ratios
        )
        self.simulation_info.update_all()

    def list_bodies(self) -> List[str]:
        return [b.name for b in self.bodies]

    def print_summary(self):
        print(f"CustomScenario with {len(self.bodies)} bodies:")
        for b in self.bodies:
            print(f"  Name: {b.name} Mass: {b.mass:.2e} Radius: {b.radius:.2e} Position: {b.position} Velocity: {b.velocity}")

    def simulate_step(self, dt: float):
        if not self.simulation_info:
            return
        bodies = self.simulation_info.bodies
        accelerations = {b: np.zeros(3) for b in bodies}
        for i, b1 in enumerate(bodies):
            for j, b2 in enumerate(bodies):
                if i == j:
                    continue
                r_vec = b2.position - b1.position
                r = np.linalg.norm(r_vec)
                if r == 0:
                    continue
                force_dir = r_vec / r
                force_mag = 6.67430e-11 * b2.mass / (r ** 2)
                accelerations[b1] += force_dir * force_mag
        for b in bodies:
            b.velocity += accelerations[b] * dt
        for b in bodies:
            b.position += b.velocity * dt
        self.simulation_info.update_all()

class SituationManager:
    def __init__(self):
        self.scenarios = {}
        self.current_scenario_key = None
        self.current_scenario = None

    def register_scenario(self, key: str, scenario_instance):
        self.scenarios[key] = scenario_instance

    def set_current(self, key: str):
        if key in self.scenarios:
            self.current_scenario_key = key
            self.current_scenario = self.scenarios[key]
        else:
            self.current_scenario = None
            self.current_scenario_key = None

    def update_current(self, dt: float):
        if self.current_scenario and hasattr(self.current_scenario, 'simulate_step'):
            self.current_scenario.simulate_step(dt)
        elif self.current_scenario and hasattr(self.current_scenario, 'update'):
            self.current_scenario.update(dt)

    def get_current_bodies(self):
        if self.current_scenario:
            if hasattr(self.current_scenario, "get_body_infos"):
                return self.current_scenario.get_body_infos()
            elif hasattr(self.current_scenario, "get_all_bodies"):
                return self.current_scenario.get_all_bodies()
            elif hasattr(self.current_scenario, "bodies"):
                return self.current_scenario.bodies
        return []

    def print_current_summary(self):
        if self.current_scenario and hasattr(self.current_scenario, 'print_summary'):
            self.current_scenario.print_summary()

