import numpy as np
from typing import List, Dict, Optional
from gravity import BodyProperties, G

class HabitabilityMetrics:
    def __init__(self, water_fraction: float, oxygen_level: float,
                 atmospheric_pressure: float, surface_temperature: float,
                 tidal_forces: float, magnetic_field_strength: float):
        self.water_fraction = water_fraction
        self.oxygen_level = oxygen_level
        self.atmospheric_pressure = atmospheric_pressure
        self.surface_temperature = surface_temperature
        self.tidal_forces = tidal_forces
        self.magnetic_field_strength = magnetic_field_strength

    def life_probability(self) -> float:
        score = 0.0
        score += max(0.0, min(1.0, self.water_fraction))
        score += max(0.0, min(1.0, self.oxygen_level))
        score += 1.0 if 0.8 <= self.atmospheric_pressure <= 1.2 else 0.0
        score += 1.0 if 273 <= self.surface_temperature <= 303 else 0.0
        score += 0.5 if self.tidal_forces < 1e15 else 0.0
        score += 0.5 if self.magnetic_field_strength > 0.1 else 0.0
        return score / 5.0

class BodyInfo:
    def __init__(self, body: BodyProperties, albedo: float, rotation_period: float,
                 core_radius_ratio: float):
        self.body = body
        self.mass = body.mass
        self.radius = body.radius
        self.density = self.mass / (4/3 * np.pi * self.radius ** 3)
        self.albedo = albedo
        self.rotation_period = rotation_period
        self.core_radius_ratio = core_radius_ratio
        self.habitability = None
        self.update_properties()

    def calculate_surface_gravity(self) -> float:
        return G * self.mass / (self.radius ** 2)

    def calculate_escape_velocity(self) -> float:
        return np.sqrt(2 * G * self.mass / self.radius)

    def update_surface_temperature(self, luminosity: float, distance: float):
        sigma = 5.670374419e-8
        self.surface_temperature = ((luminosity * (1 - self.albedo)) / (16 * np.pi * sigma * distance ** 2)) ** 0.25

    def estimate_tidal_forces(self, central_body: BodyProperties):
        distance = np.linalg.norm(self.body.position - central_body.position)
        self.tidal_forces = 2 * G * central_body.mass * self.radius / distance ** 3

    def estimate_water_level(self, solar_flux: float):
        self.water_level = min(solar_flux / 1361, 1.0)

    def update_magnetic_field(self):
        core_radius = self.core_radius_ratio * self.radius
        self.magnetic_field = core_radius ** 2 / self.rotation_period

    def calculate_habitability(self):
        self.habitability = HabitabilityMetrics(
            water_fraction=self.water_level,
            oxygen_level=0.21,
            atmospheric_pressure=1.0,
            surface_temperature=self.surface_temperature,
            tidal_forces=self.tidal_forces,
            magnetic_field_strength=self.magnetic_field
        )

    def update_properties(self, luminosity: float = None, central_body: BodyProperties = None):
        if luminosity is not None and central_body is not None:
            distance = np.linalg.norm(self.body.position - central_body.position)
            self.update_surface_temperature(luminosity, distance)
            self.estimate_tidal_forces(central_body)
            solar_flux = luminosity / (4 * np.pi * distance ** 2)
            self.estimate_water_level(solar_flux)
        self.update_magnetic_field()
        self.calculate_habitability()

    def info_dict(self) -> Dict:
        return {
            "mass": self.mass,
            "radius": self.radius,
            "density": self.density,
            "surface_temperature": self.surface_temperature,
            "water_level": self.water_level,
            "tidal_forces": self.tidal_forces,
            "magnetic_field": self.magnetic_field,
            "life_probability": self.habitability.life_probability() if self.habitability else None,
        }

class SimulationInfo:
    def __init__(self, central_body: BodyProperties, bodies: List[BodyProperties],
                 albedos: List[float], rotation_periods: List[float], core_radius_ratios: List[float]):
        self.central_body = central_body
        self.bodies = bodies
        self.albedos = albedos
        self.rotation_periods = rotation_periods
        self.core_radius_ratios = core_radius_ratios
        self.body_infos: List[BodyInfo] = []

    def estimate_luminosity(self, mass: float) -> float:
        solar_luminosity = 3.828e26
        solar_mass = 1.989e30
        return solar_luminosity * (mass / solar_mass) ** 3.5

    def update_all(self):
        self.body_infos.clear()
        luminosity = self.estimate_luminosity(self.central_body.mass)
        for i, body in enumerate(self.bodies):
            albedo = self.albedos[i] if i < len(self.albedos) else 0.3
            rotation_period = self.rotation_periods[i] if i < len(self.rotation_periods) else 86400
            core_radius_ratio = self.core_radius_ratios[i] if i < len(self.core_radius_ratios) else 0.5
            info = BodyInfo(body, albedo, rotation_period, core_radius_ratio)
            info.update_properties(luminosity, self.central_body)
            self.body_infos.append(info)

    def get_info_summary(self) -> List[Dict]:
        return [info.info_dict() for info in self.body_infos]

    def get_body_info(self, index: int) -> Optional[Dict]:
        if 0 <= index < len(self.body_infos):
            return self.body_infos[index].info_dict()
        return None

    def print_summary(self):
        for idx, info in enumerate(self.body_infos):
            data = info.info_dict()
            print(f"Body {idx} summary:")
            for key, value in data.items():
                print(f"  {key}: {value}")
            print("")

class ClimateModel:
    def __init__(self, body_info: BodyInfo, albedo: float = 0.3, co2_level: float = 0.0004, greenhouse_factor: float = 1.0):
        self.body_info = body_info
        self.latitude_temperatures = np.zeros(181)
        self.seasonal_cycle = np.zeros(365)
        self.albedo = albedo
        self.co2_level = co2_level
        self.greenhouse_factor = greenhouse_factor

    def calculate_latitudinal_temperature_gradient(self, solar_constant: float):
        for lat in range(-90, 91):
            angle = np.radians(lat)
            insolation = solar_constant * max(0, np.cos(angle))
            base_temp = (insolation * (1 - self.albedo) / (5.67e-8) / 4) ** 0.25
            self.latitude_temperatures[lat + 90] = base_temp * self.greenhouse_factor

    def get_average_global_temperature(self) -> float:
        return np.mean(self.latitude_temperatures) * self.greenhouse_factor

    def simulate_day(self, day_of_year: int, solar_constant: float):
        self.calculate_latitudinal_temperature_gradient(solar_constant)

    def get_temperature_at_latitude(self, latitude: float) -> float:
        idx = int(latitude) + 90
        if 0 <= idx < len(self.latitude_temperatures):
            return self.latitude_temperatures[idx]
        return 0.0


class HydrologyModel:
    def __init__(self, body_info: BodyInfo, ocean_coverage: float = 0.7):
        self.body_info = body_info
        self.ocean_coverage = ocean_coverage
        self.evaporation_rate = 0.0
        self.precipitation_rate = 0.0
        self.surface_water_volume = self.ocean_coverage * 1.4e18

    def update_evaporation(self, temperature: float):
        self.evaporation_rate = 0.1 * np.exp(0.1 * (temperature - 273))

    def update_precipitation(self, temperature: float):
        self.precipitation_rate = 0.05 * temperature / 300

    def update_water_volume(self):
        net = self.precipitation_rate - self.evaporation_rate
        self.surface_water_volume += net * 1e10
        self.surface_water_volume = max(0, min(self.surface_water_volume, self.ocean_coverage * 2e18))

    def simulate_hydrological_cycle(self, temperature: float):
        self.update_evaporation(temperature)
        self.update_precipitation(temperature)
        self.update_water_volume()

class BiosignatureAnalyzer:
    def __init__(self, body_info: BodyInfo, atmospheric_composition: Dict[str, float]):
        self.body_info = body_info
        self.methane_level = atmospheric_composition.get("CH4", 0.0)
        self.oxygen_level = atmospheric_composition.get("O2", 0.21)
        self.water_vapor_level = atmospheric_composition.get("H2O", 0.0)
        self.other_gas_levels = {gas: level for gas, level in atmospheric_composition.items() if gas not in ("CH4", "O2", "H2O")}
        self.biosignature_score = 0.0

    def calculate_biosignature_score(self):
        score = 0.0
        score += min(1.0, self.methane_level * 10)
        score += min(1.0, self.oxygen_level)
        score += min(1.0, self.water_vapor_level * 5)
        for gas, level in self.other_gas_levels.items():
            score += min(0.5, level)
        self.biosignature_score = min(score, 1.0)

    def get_biosignature_summary(self) -> Dict:
        return {
            "methane_level": self.methane_level,
            "oxygen_level": self.oxygen_level,
            "water_vapor_level": self.water_vapor_level,
            "other_gases": self.other_gas_levels,
            "biosignature_score": self.biosignature_score,
        }


class GeologyModel:
    def __init__(self, body_info: BodyInfo, internal_heat_flux: float = 0.1, crust_thickness: float = 30000, lithosphere_strength: float = 1.0):
        self.body_info = body_info
        self.volcanic_activity_index = 0.0
        self.tectonic_activity_level = 0.0
        self.mountain_height_distribution = []
        self.crust_thickness = crust_thickness
        self.lithosphere_strength = lithosphere_strength
        self.estimate_volcanic_activity(internal_heat_flux)
        self.estimate_tectonic_activity(crust_thickness, lithosphere_strength)

    def estimate_volcanic_activity(self, internal_heat_flux: float):
        self.volcanic_activity_index = internal_heat_flux * 0.1

    def estimate_tectonic_activity(self, crust_thickness: float, lithosphere_strength: float):
        self.crust_thickness = crust_thickness
        self.tectonic_activity_level = max(0.0, min(1.0, 1.0 - crust_thickness / 40000)) * lithosphere_strength

    def update_mountain_distribution(self, num_mountains: int, max_height: float):
        self.mountain_height_distribution = np.random.uniform(0, max_height, size=num_mountains).tolist()

    def geology_summary(self) -> Dict:
        return {
            "volcanic_activity_index": self.volcanic_activity_index,
            "tectonic_activity_level": self.tectonic_activity_level,
            "mountain_height_distribution": self.mountain_height_distribution,
            "crust_thickness": self.crust_thickness,
        }
    
class AtmosphericModel:
    def __init__(self, body_info: BodyInfo, composition: dict = None, pressure: float = 1.0,
                 mean_molecular_weight: float = 28.96):
        self.body_info = body_info
        self.composition = composition if composition is not None else {"N2": 0.78, "O2": 0.21, "CO2": 0.0004}
        self.pressure = pressure
        self.mean_molecular_weight = mean_molecular_weight
        self.scale_height = 8500
        self.surface_density = self.calculate_surface_density()

    def calculate_surface_density(self):
        R = 8.314
        pressure_Pa = self.pressure * 101325
        temperature = self.body_info.surface_temperature if self.body_info.surface_temperature > 0 else 288.0
        density = pressure_Pa / (R * temperature / self.mean_molecular_weight)
        return density

    def update_atmospheric_composition(self, new_composition: dict):
        self.composition = new_composition.copy()

    def update_pressure(self, new_pressure: float):
        self.pressure = new_pressure
        self.surface_density = self.calculate_surface_density()

    def update_scale_height(self, gravity: float, temperature: float):
        self.scale_height = (8.314 * temperature) / (self.mean_molecular_weight * gravity)

    def get_atmospheric_summary(self) -> dict:
        return {
            "composition": self.composition,
            "pressure": self.pressure,
            "scale_height": self.scale_height,
            "mean_molecular_weight": self.mean_molecular_weight,
            "surface_density": self.surface_density,
        }

class MagneticFieldModel:
    def __init__(self, body_info: BodyInfo, core_conductivity: float = 1e5,
                 rotation_rate: float = 2 * np.pi / 86400, age: float = 4.5e9):
        self.body_info = body_info
        self.magnetic_field_strength = 0.0
        self.core_conductivity = core_conductivity
        self.rotation_rate = rotation_rate
        self.age = age

    def simulate_dynamo_activity(self):
        radius = self.body_info.radius
        mass = self.body_info.mass
        conduct = self.core_conductivity
        self.magnetic_field_strength = conduct * (mass / radius ** 2) * self.rotation_rate * np.exp(-self.age / 4e9)

    def update_rotation_rate(self, rotation_rate: float):
        self.rotation_rate = rotation_rate

    def update_core_conductivity(self, conductivity: float):
        self.core_conductivity = conductivity

    def update_age(self, age: float):
        self.age = age

    def get_magnetic_field_summary(self) -> dict:
        return {
            "magnetic_field_strength": self.magnetic_field_strength,
            "core_conductivity": self.core_conductivity,
            "rotation_rate": self.rotation_rate,
            "age": self.age,
        }
    
import math

class RotationModel:
    def __init__(self, body_info: BodyInfo, axial_tilt: float = 23.5, rotation_period: float = 86400):
        self.body_info = body_info
        self.axial_tilt = axial_tilt
        self.rotation_period = rotation_period
        self.precession_rate = 0.0
        self.nutation_amplitude = 0.0
        self.seasonal_variation_amplitude = 0.2

    def calculate_precession(self, gravitational_torques: float):
        self.precession_rate = gravitational_torques / (self.body_info.mass * self.body_info.radius ** 2)

    def calculate_nutation(self, torque_variation: float):
        self.nutation_amplitude = torque_variation / (self.body_info.mass * self.body_info.radius ** 2)

    def effective_axial_tilt(self, day_of_year: int) -> float:
        nutation_effect = self.nutation_amplitude * math.sin(2 * math.pi * day_of_year / 365.25)
        return self.axial_tilt + nutation_effect

    def rotate(self, delta_time: float):
        self.rotation_period -= delta_time * self.precession_rate

    def get_rotation_summary(self) -> dict:
        return {
            "axial_tilt": self.axial_tilt,
            "rotation_period": self.rotation_period,
            "precession_rate": self.precession_rate,
            "nutation_amplitude": self.nutation_amplitude,
            "seasonal_variation_amplitude": self.seasonal_variation_amplitude,
        }

class OceanModel:

    def __init__(self, body_info: BodyInfo, ocean_coverage: float = 0.7,
                 average_depth: float = 3700, salinity: float = 35):
        
        self.body_info = body_info
        self.ocean_coverage = ocean_coverage
        self.average_depth = average_depth
        self.salinity = salinity
        self.temperature_profile = []

    def calculate_average_temperature(self, surface_temperature: float):
        self.temperature_profile = [surface_temperature - depth * 0.02 for depth in range(0,int(self.average_depth),100)]

    def update_ocean_properties(self, temp_profile: list = None, salinity: float = None):
        if temp_profile:
            self.temperature_profile = temp_profile
        if salinity:
            self.salinity = salinity

    def get_ocean_summary(self) -> dict:
        return {
            "ocean_coverage": self.ocean_coverage,
            "average_depth": self.average_depth,
            "salinity": self.salinity,
            "temperature_profile": self.temperature_profile
        }
    

class BiosphereModel:

    def __init__(self, body_info: BodyInfo,initial_life_probability: float = 0.0):
        self.body_info = body_info
        self.life_probability = initial_life_probability
        self.complexity_index = 0.0
        self.biodiversity_index - 0.0
        self.age = 0.0

    def evolve_life(self, time_step: float, environmental_factors: dict):
        growth_factor = environmental_factors.get("growth_factor",0.01)
        self.life_probability = min(1.0, self.life_probability + growth_factor * time_step)

        comlexity_growth = environmental_factors.get("complexity_growth",0.001)
        self.complexity_index = min(1.0, self.complexity_index + comlexity_growth * time_step)

        diversity_growth = environmental_factors.get("diversity_growth", 0.001)
        self.biodiversity_index = min(1.0, self.biodiversity_index + diversity_growth * time_step)

        self.age += time_step
    
    def get_biosphere_summary(self) -> dict:
        return {
            "life_probability": self.life_probability,
            "complexity_index": self.complexity_index,
            "biodiversity_index": self.biodiversity_index,
            "age":self.age
        }
    
class AtmosphericChemistry:

    def __init__(self, body_info: BodyInfo, gas_mixing_ratio: Dict[str, float] = None, photochemical_reactions: Dict[str,float] = None):
        self.body_info = body_info
        self.gax_mixing_ratios = gas_mixing_ratio if gas_mixing_ratio else {"N2":0.78,"O2":0.21,"CO2":0.0004}
        self.photochemical_reactions = photochemical_reactions if photochemical_reactions else {}

        self.ozone_layer_thickness = 0.0

    def simulate_photochemistry(self, solar_flux:float,time_step:float):
        o3_production_rate = self.gax_mixing_ratios.get("O2",0) * solar_flux * 1e-6

        self.ozone_layer_thickness += o3_production_rate * time_step - self.ozone_layer_thickness * 1e-4 * time_step

    def update_gas_mixing_ratios(self, new_ratios: Dict[str, float]):
        self.gax_mixing_ratios = new_ratios.copy()

    def get_chemistry_summary(self) -> Dict:
        return {
            "gas_mixing_ratios": self.gax_mixing_ratios,
            "ozone_layer_thickness": self.ozone_layer_thickness,
            "photochemical_ractions": self.photochemical_reactions
        }
    

class SurfaceFeatures:

    def __init__(self, body_info:BodyInfo,crater_density: float = 0.0, mountain_range_count: int = 0,
                 volcanic_activity_level:float = 0.0):
        self.body_info = body_info
        self.crater_density = crater_density
        self.mountain_range_count = mountain_range_count
        self.volcanic_activity_level = volcanic_activity_level
        self.river_network_density = 0.0
        self.ice_cap_coverage = 0.0

    def update_surface_features(self, tectonic_activity: float, climate_tmep: float):
        self.volcanic_activity_level = tectonic_activity * 0.8

        self.river_network_density = max(0.0,min(1.0, climate_tmep / 300))

        self.ice_cap_coverage = 1.0 - max(0.0,min(1.0,climate_tmep / 280))

    def get_surface_summary(self) -> Dict:
        return {
            "crater_density": self.crater_density,
            "mountain_range_count":self.mountain_range_count,
            "volcanic_activity_level": self.volcanic_activity_level,
            "river_network_density": self.river_network_density,
            "ice_cap_coverage": self.ice_cap_coverage
        }
    

class SpaceWeatherModel:

    def __init__(self, body_info: BodyInfo, solar_wind_intensity: float = 1.0,flare_activity_level: float = 0.0):
        self.body_info = body_info
        self.solar_wind_intensity = solar_wind_intensity
        self.flare_activity_level = flare_activity_level
        self.magnetic_storm_probability = 0.0
        self.radiation_belt_strength = 0.0

    def update_solar_wind(self, new_intensity:float):
        self.solar_wind_intensity = new_intensity

    def update_flare_activity(self, new_level: float):
        self.flare_activity_level = new_level

    def calculate_magnetic_storm_probability(self):
        self.magnetic_storm_probability = 1 - np.exp(-self.solar_wind_intensity * self.flare_activity_level)

    def calculate_radiation_belt_strength(self):
        self.radiation_belt_strength = self.solar_wind_intensity * np.sqrt(self.flare_activity_level)

    def simulate_space_weather_cycle(self, timestep: float):
        self.calculate_magnetic_storm_probability()
        self.calculate_radiation_belt_strength()

    def get_space_weather_summary(self) -> dict:
        return {
            "solar_wind_intensity": self.solar_wind_intensity,
            "flare_activity_level": self.flare_activity_level,
            "magnetic_storm_probability": self.magnetic_storm_probability,
            "radiation_belt_strength": self.radiation_belt_strength
        }
    
class RadiationEnvironment:

    def __init__(self, body_info: BodyInfo, cosmic_ray_flux: float = 1.0, uv_radiation_level: float = 1.0):
        self.body_info = body_info
        self.cosmic_ray_flux = cosmic_ray_flux
        self.uv_radiation_level = uv_radiation_level
        self.surface_radiation_dose = 0.0

    def update_cosmic_ray_flux(self, new_flux: float):
        self.cosmic_ray_flux = new_flux

    def update_uv_radiation_level(self, new_level: float):
        self.uv_radiation_level = new_level

    def calculate_surface_radiation_dose(self, atmospheric_thickness: float):
        atmospheric_shielding_factor = np.exp(-atmospheric_thickness / 1000)
        self.surface_radiation_dose = (self.cosmic_ray_flux + self.uv_radiation_level) * atmospheric_shielding_factor

    def simulate_radiation_environment(self, atmospheric_thickness: float, timestep: float):
        self.calculate_surface_radiation_dose(atmospheric_thickness)

    def get_radiation_summary(self) -> dict:
        return {
            "cosmic_ray_flux": self.cosmic_ray_flux,
            "uv_radiation_level": self.uv_radiation_level,
            "surface_radiation_does":self.surface_radiation_dose
        }
    

class EcosystemModel:

    def __init__(self, body_info: BodyInfo, initial_biomass: float = 1.0):
        self.body_info = body_info
        self.biomass = initial_biomass
        self.species_diversity = 1.0
        self.productivity_rate = 0.1
        self.decomposition_rate = 0.05
        self.nutrient_level = 0.5

    def simulate_growth(self, environmental_stress: float, time_step: float):
        growth_factor = self.productivity_rate * (1 - environmental_stress)
        net_growth = growth_factor - self.decomposition_rate
        self.biomass = max(0.0, self.biomass + net_growth * time_step)
        self.species_diversity = min(10.0, self.species_diversity + net_growth * 0.1 * time_step)

    def nutrient_deletion(self, consumption_rate:float, time_step: float):
        self.nutrient_level = max(0.0,self.nutrient_level - consumption_rate * time_step)

    def nutrient_replenish(self, replenish_rate: float, time_step: float):
        self.nutrient_level = min(1.0, self.nutrient_level + replenish_rate * time_step)

    def get_ecosystem_summary(self) -> dict:
        return {
            "biomass": self.biomass,
            "species_diversity": self.species_diversity,
            "productivity_rate": self.productivity_rate,
            "decomposition_rate": self.decomposition_rate,
            "nutrient_level": self.nutrient_level,
        }