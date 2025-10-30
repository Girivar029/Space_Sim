import pygame
from typing import Optional
from informations import BodyInfo, HabitabilityMetrics


class HUD:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont('Arial', 18)
        self.small_font = pygame.font.SysFont('Arial', 14)
        self.bg_color = (0, 0, 0, 180)
        self.text_color = (0, 200, 0)
        self.highlight_color = (0, 255, 0)
        self.selected_body: Optional[BodyInfo] = None
        self.sections_expanded = {
            "habitability": True,
            "atmosphere": False,
            "climate": False,
            "geology": False,
            "magnetic": False,
            "biosignature": False,
            "space_weather": False,
            "ecosystem": False,
            "hydrology": False,
            "surface": False
        }

    def select_body(self, body_info: Optional[BodyInfo]):
        self.selected_body = body_info

    def panel(self, rect: pygame.Rect):
        surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        surface.fill(self.bg_color)
        self.screen.blit(surface, (rect.x, rect.y))

    def draw_text(self, text: str, pos, font=None, color=None):
        font = font or self.font
        color = color or self.text_color
        label_surf = font.render(text, True, color)
        self.screen.blit(label_surf, pos)

    def draw_bar(self, pos, width, height, fraction, color):
        pygame.draw.rect(self.screen, color, (*pos, width, height))
        fill_width = max(0, min(width, int(width * fraction)))
        pygame.draw.rect(self.screen, color, (*pos, fill_width, height))

    def render_basic_info(self, x, y):
        if not self.selected_body:
            self.draw_text("No Body selected", (x, y))
            return y + 30
        self.draw_text(f"Mass: {self.selected_body.mass:.2E} kg", (x, y))
        y += 25
        self.draw_text(f"Radius: {self.selected_body.radius / 1000:.2f} km", (x, y))
        y += 25
        self.draw_text(f"Density: {self.selected_body.density:.2f} kg/m^3", (x, y))
        y += 30
        return y

    def render_habitability(self, x, y):
        if not self.selected_body or not self.selected_body.habitability:
            return y
        hab = self.selected_body.habitability
        self.draw_text("Habitability Metrics:", (x, y), color=self.highlight_color)
        y += 25

        labels = {
            "water_fraction": "Water Fraction",
            "oxygen_level": "Oxygen Level",
            "atmospheric_pressure": "Atmospheric Pressure",
            "surface_temperature": "Surface Temperature (K)",
            "tidal_forces": "Tidal Forces",
            "magnetic_field_strength": "Magnetic Field Strength"
        }

        for attr in [
            'water_fraction', 'oxygen_level', 'atmospheric_pressure',
            'surface_temperature', 'tidal_forces', 'magnetic_field_strength'
        ]:
            val = getattr(hab, attr, None)
            if val is not None:
                label = labels.get(attr, attr.replace('_', ' ').title())
                self.draw_text(f"{label}: {val:.3f}", (x + 10, y))
                y += 22

        y += 7
        life_prob = hab.life_probability()
        self.draw_text("Life Probability:", (x + 10, y))
        self.draw_bar((x + 140, y), 150, 15, life_prob, self.highlight_color)
        self.draw_text(f"{life_prob:.2f}", (x + 300, y))
        y += 30
        return y

    def render_atmosphere(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, 'atmospheric_model'):
            return y
        atm = self.selected_body.atmospheric_model
        self.draw_text("Atmosphere:", (x, y), color=self.highlight_color)
        y += 25
        comp_str = ', '.join(f"{k}: {v:.3f}" for k, v in atm.composition.items())
        self.draw_text(f"Composition: {comp_str}", (x + 10, y))
        y += 20
        self.draw_text(f"Pressure: {atm.pressure:.3f} atm", (x + 10, y))
        y += 20
        self.draw_text(f"Mean Molecular Weight: {atm.mean_molecular_weight:.2f}", (x + 10, y))
        y += 20
        self.draw_text(f"Surface Density: {atm.surface_density:.4f} kg/m^3", (x + 10, y))
        y += 30
        return y

    def render_climate(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, "climate_model"):
            return y
        climate = self.selected_body.climate_model
        self.draw_text("Climate:", (x, y), color=self.highlight_color)
        y += 25
        avg_temp = climate.get_average_global_temperature()
        self.draw_text(f"Avg Global Temp: {avg_temp:.3f} K", (x + 10, y))
        y += 20
        bar_x, bar_y = x + 10, y + 10
        bar_w, bar_h = 300, 15
        for lat in range(-90, 91, 30):
            temp = climate.get_temperature_at_latitude(lat)
            color_val = max(0, min(255, int((temp - 200) * 2)))
            pygame.draw.rect(self.screen, (color_val, 100, 255 - color_val),
                             (bar_x + ((lat + 90) / 180) * bar_w, bar_y, bar_w / 10, bar_h))
        y += 50
        return y

    def render_hydrology(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, "hydrology_model"):
            return y
        hydrology = self.selected_body.hydrology_model
        self.draw_text("Hydrology:", (x, y), color=self.highlight_color)
        y += 25
        self.draw_text(f"Ocean Coverage: {hydrology.ocean_coverage:.2f}", (x + 10, y))
        y += 20
        self.draw_text(f"Surface Water Volume: {hydrology.surface_water_volume:.2e} m³", (x + 10, y))
        y += 20
        self.draw_text(f"Evaporation Rate: {hydrology.evaporation_rate:.3f}", (x + 10, y))
        y += 20
        self.draw_text(f"Precipitation Rate: {hydrology.precipitation_rate:.3f}", (x + 10, y))
        y += 30
        return y


    def render_biosignature(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, "biosignature_analyzer"):
            return y
        biosig = self.selected_body.biosignature_analyzer
        biosig.calculate_biosignature_score()
        self.draw_text("Biosignature Analyzer:", (x, y), color=self.highlight_color)
        y += 25
        summary = biosig.get_biosignature_summary()
        for name, val in summary.items():
            if isinstance(val, dict):
                continue
            self.draw_text(f"{name.replace('_',' ').title()}: {val:.3f}", (x + 10, y))
        y += 20
        self.draw_text("Other Gases:", (x + 10, y))
        g_str = ', '.join(f"{g}: {v:.3f}" for g, v in summary['other_gases'].items())
        self.draw_text(g_str, (x + 110, y))
        y += 30
        self.draw_bar((x + 10, y), 150, 15, summary['biosignature_score'], (220, 180, 50))
        self.draw_text(f"Score: {summary['biosignature_score']:.2f}", (x + 170, y))
        y += 30
        return y


    def render_geology(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, "geology_model"):
            return y
        geology = self.selected_body.geology_model
        summary = geology.geology_summary()
        self.draw_text("Geology:", (x, y), color=self.highlight_color)
        y += 25
        self.draw_text(f"Volcanic Activity: {summary['volcanic_activity_index']:.4f}", (x + 10, y))
        y += 20
        self.draw_text(f"Tectonic Activity: {summary['tectonic_activity_level']:.3f}", (x + 10, y))
        y += 20
        self.draw_text(f"Crust Thickness: {summary['crust_thickness']:.0f} m", (x + 10, y))
        y += 20
        mt_heights = summary.get("mountain_height_distribution", [])
        if mt_heights:
            self.draw_text(f"Max Mountain Height: {max(mt_heights):.0f} m", (x + 10, y))
            y += 20
        y += 20
        return y


    def render_magnetic_field(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, "magnetic_field_model"):
            return y
        magnetic = self.selected_body.magnetic_field_model
        summary = magnetic.get_magnetic_field_summary()
        self.draw_text("Magnetic Field:", (x, y), color=self.highlight_color)
        y += 25
        self.draw_text(f"Field Strength: {summary['magnetic_field_strength']:.2f}", (x + 10, y))
        y += 20
        self.draw_text(f"Core Conductivity: {summary['core_conductivity']:.2f}", (x + 10, y))
        y += 20
        self.draw_text(f"Rotation Rate: {summary['rotation_rate']:.3e}", (x + 10, y))
        y += 20
        self.draw_text(f"Age: {summary['age']:.2e} yrs", (x + 10, y))
        y += 30
        return y


    def render_space_weather(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, "space_weather_model"):
            return y
        sw = self.selected_body.space_weather_model
        summary = sw.get_space_weather_summary()
        self.draw_text("Space Weather:", (x, y), color=self.highlight_color)
        y += 25
        self.draw_text(f"Solar Wind Intensity: {summary['solar_wind_intensity']:.2f}", (x + 10, y))
        y += 20
        self.draw_text(f"Flare Activity: {summary['flare_activity_level']:.2f}", (x + 10, y))
        y += 20
        self.draw_text(f"Magnetic Storm Probability: {summary['magnetic_storm_probability']:.2f}", (x + 10, y))
        y += 20
        self.draw_text(f"Radiation Belt Strength: {summary['radiation_belt_strength']:.2f}", (x + 10, y))
        y += 30
        return y


    def render_ecosystem(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, "ecosystem_model"):
            return y
        eco = self.selected_body.ecosystem_model
        summary = eco.get_ecosystem_summary()
        self.draw_text("Ecosystem:", (x, y), color=self.highlight_color)
        y += 25
        for name, val in summary.items():
            self.draw_text(f"{name.replace('_', ' ').title()}: {val:.2f}", (x + 10, y))
            y += 20
        y += 30
        return y


    def render_surface(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, "surface_features"):
            return y
        surf = self.selected_body.surface_features
        summary = surf.get_surface_summary()
        self.draw_text("Surface Features:", (x, y), color=self.highlight_color)
        y += 25
        self.draw_text(f"Crater Density: {summary['crater_density']:.2f}", (x + 10, y))
        y += 20
        self.draw_text(f"Mountain Ranges: {summary['mountain_range_count']}", (x + 10, y))
        y += 20
        self.draw_text(f"Volcanic Activity Level: {summary['volcanic_activity_level']:.2f}", (x + 10, y))
        y += 20
        self.draw_text(f"River Network Density: {summary['river_network_density']:.2f}", (x + 10, y))
        y += 20
        self.draw_text(f"Ice Cap Coverage: {summary['ice_cap_coverage']:.2f}", (x + 10, y))
        y += 30
        return y


    def render_selected(self):
        panel_rect = pygame.Rect(10, 10, 420, self.screen.get_height() - 20)
        self.panel(panel_rect)
        y = self.render_basic_info(panel_rect.x + 10, panel_rect.y + 10)
        if self.sections_expanded.get("habitability"): y = self.render_habitability(panel_rect.x + 10, y)
        if self.sections_expanded.get("atmosphere"): y = self.render_atmosphere(panel_rect.x + 10, y)
        if self.sections_expanded.get("climate"): y = self.render_climate(panel_rect.x + 10, y)
        # Call others similarly
        # if self.sections_expanded.get("hydrology"): y = self.render_hydrology(panel_rect.x + 10, y)
        # ...
        # Continue for all expanded sections...

    def update(self):
        if not self.selected_body:
            return
        self.render_selected()

    def toggle_section(self, section_name: str):
        if section_name in self.sections_expanded:
            self.sections_expanded[section_name] = not self.sections_expanded[section_name]

    def handle_click(self, pos):
        panel_rect = pygame.Rect(10, 10, 420, self.screen.get_height() - 20)
        if not panel_rect.collidepoint(pos):
            self.select_body(None)
            return

        x, y = pos
        base_y = 30
        height_per_section = 25

        # Map y-ranges to toggle sections
        if 30 <= y < 30 + height_per_section:
            self.toggle_section("habitability")
        elif 30 + height_per_section <= y < 30 + 2 * height_per_section:
            self.toggle_section("atmosphere")
        elif 30 + 2 * height_per_section <= y < 30 + 3 * height_per_section:
            self.toggle_section("climate")
        # Continue for all sections similarly...

    def display_warning(self, text: str):
        w, h = self.screen.get_size()
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        font = pygame.font.SysFont("Arial", 36)
        label = font.render(text, True, (255, 0, 0))
        rect = label.get_rect(center=(w // 2, h // 2))
        self.screen.blit(label, rect)

    def clear_selection(self):
        self.selected_body = None

    def get_selected_body(self):
        return self.selected_body



import numpy as np
from gravity import G, BodyProperties

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
        self.water_level = 0.0
        self.tidal_forces = 0.0
        self.surface_temp = 0.0
        self.magnetic_field = 0.0
        self.update_properties()

    @property
    def surface_temperature(self):
        return self.surface_temp

    def calculate_surface_gravity(self) -> float:
        return G * self.mass / (self.radius ** 2)

    def calculate_escape_velocity(self) -> float:
        return np.sqrt(2 * G * self.mass / self.radius)

    def update_surface_temperature(self, luminosity: float, distance: float):
        sigma = 5.670374419e-8
        safe_distance = max(distance, 1e-12)
        self.surface_temp = ((luminosity * (1 - self.albedo)) / (16 * np.pi * sigma * safe_distance ** 2)) ** 0.25

    def estimate_tidal_forces(self, central_body: BodyProperties):
        distance = np.linalg.norm(self.body.position - central_body.position)
        safe_distance = max(distance, 1e-12)
        self.tidal_forces = 2 * G * central_body.mass * self.radius / safe_distance ** 3

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
            surface_temperature=self.surface_temp,
            tidal_forces=self.tidal_forces,
            magnetic_field_strength=self.magnetic_field
        )

    def update_properties(self, luminosity: float = None, central_body: BodyProperties = None):
        if luminosity is not None and central_body is not None:
            distance = np.linalg.norm(self.body.position - central_body.position)
            self.update_surface_temperature(luminosity, distance)
            self.estimate_tidal_forces(central_body)
            solar_flux = luminosity / (4 * np.pi * max(distance, 1e-12) ** 2)
            self.estimate_water_level(solar_flux)
        self.update_magnetic_field()
        self.calculate_habitability()

    def info_dict(self) -> dict:
        return {
            "mass": self.mass,
            "radius": self.radius,
            "density": self.density,
            "surface_temperature": self.surface_temp,
            "water_level": self.water_level,
            "tidal_forces": self.tidal_forces,
            "magnetic_field": self.magnetic_field,
            "life_probability": self.habitability.life_probability() if self.habitability else None,
        }
