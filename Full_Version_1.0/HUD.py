#This runs the HUD data into a pygame interface
import pygame
from informations import (SpaceWeatherModel,EcosystemModel,RadiationEnvironment,
                          RotationModel,BodyInfo,SimulationInfo,OceanModel,
                          AtmosphericChemistry,AtmosphericModel,
                          BiosignatureAnalyzer,SurfaceFeatures,MagneticFieldModel,
                          GeologyModel,HabitabilityMetrics,HydrologyModel,
                          ClimateModel,BiosphereModel)
from typing import Optional, Dict
from gravity import BodyProperties, G

class HUD:

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont('Arial',18)
        self.small_font = pygame.font.SysFont('Arial',14)
        self.bg_color = (0,0,0,180)
        self.text_color = (0,200,0)
        self.selected_body: Optional[BodyInfo] = None
        self.sections_expanded = {
            "habitability":True,
            "atmosphere":False,
            "climate": False,
            "geology": False,
            "magnetic": False,
            "biosignature": False,
            "space_weather": False,
            "ecosystem": False
        }

    def select_body(self, body_info: Optional[BodyInfo]):
        self.selected_body = body_info

    def panel(self, rect: pygame.Rect):
        surface = pygame.Surface((rect.width,rect.height), pygame.SRCALPHA)
        surface.fill(self.bg_color)
        self.screen.blit(surface,(rect.x, rect.y))

    def draw_text(self, text:str, pos, font=None,color=None):
        font = font or self.font
        color = color or self.text_color
        label_surf = font.render(text, True, color)
        self.screen.blit(label_surf,pos)

    def draw_bar(self, pos, width, height, fraction, color):
        pygame.draw.rect(self.screen, color, (*pos, width, height))
        fill_width = max(0, min(width, int(width * fraction)))
        pygame.draw.rect(self.screen, color, (*pos, fill_width, height))

    def render_basic_info(self, x, y):
        #Show HUD only when a body is selected
        if not self.selected_body:
            self.draw_text("No Body selected",(x, y))
            return y + 30
        self.draw_text(f"Mass:{self.selected_body.mass:.2E} kg", (x, y))
        y += 25
        self.draw_text(f"Radius: {self.selected_body.radius: .2f} km", (x,y))
        y += 25
        self.draw_text(f"Density: {self.selected_body.density: .2f} kg/m^3", (x, y))
        y += 30
        return y
    
    def render_habitability(self, x, y):
        if not self.selected_body or not self.selected_body.habitability:
            return y
        
        hab = self.selected_body.habitability
        self.draw_text("Habitability Metrics:" , (x,y), color = self.highlight_color)
        y += 25
        for attr in ['water_fraction', 'oxygen_level','atmospheric_pressure','surface_temperature','tidal_forces','magnetic_field_strength']:
            val = getattr(hab, attr, None)
            if val is not None:
                label =  attr.replace('_',' ').title()
                self.draw_text(f"{label}: {val:.3f}", (x + 10, y))
                y += 0
        life_prob = hab.life_probability()
        self.draw_text("Life Probability:", (x+10,y))
        self.draw_bar((x + 130, y), 150, 15, life_prob, self.highlight_color)
        y += 35
        return y
        
    def render_atmosphere(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, 'atmospheric_model'):
             return y
            
        atm = self.selected_body.atmospheric_model
        self.draw_text("Atmosphere:", (x,y), color=self.highlight_color)
        y += 25
        comp_str = ', '.join(f"{k}:{v:.3f}" for k,v in atm.composition.items())
        self.draw_text(f"Composition: {comp_str}", (x + 10, y))
        y += 20
        self.draw_text(f"Pressure: {atm.pressure: .3f} atm",(x+10,y))
        y += 20
        self.draw_text(f"Mean Molecular Wight: {atm.mean_molecular_weight:.2f}", (x+10,y))
        y += 20
        self.draw_text(f"Surface Density: {atm.surface_density:.4f} kg/m^3", (x+10,y))
        y += 30
        return y
    
    def render_climate(self,x,y):
        if not self.selected_body or not hasattr(self.selected_body, "climate_model"):
            return y
        climate: ClimateModel = self.selected_body.climate_model
        self.draw_text("Climate:", (x,y), color = self.highlight_color)
        y += 25
        avg_temp = climate.get_average_global_temperature()
        self.draw_text(f"Avg Global Temp: {avg_temp:.3f} K", (x+10,y))
        y += 20
        bar_x,bar_y = x+10, y+10
        bar_w, bar_h = 300,15
        for lat in range(-90,91,30):
            temp = climate.get_temperature_at_latitude(lat)
            color_val = max(0, min(255, int((temp - 200)*2)))
            pygame.draw.rect(self.screen, (color_val, 100, 255-color_val),(bar_x + ((lat+90)/180)*bar_w, bar_y, bar_w/10,bar_h))
            y += 50
            return y
        
    # More render functions for Hydrology,Biosignature, Geology, MagneticField, SpaceWeather, Ecosystem

    def render(self):
        panel_rect = pygame.Rect(10, 10, 420, self.screen.get_height() - 20)
        self.panel(panel_rect)
        y = self.render_basic_info(panel_rect.x + 10, panel_rect.y + 10)
        y = self.render_habitability(panel_rect.x + 10, y)
        y = self.render_atmosphere(panel_rect.x + 10, y)
        y = self.render_climate(panel_rect.x + 10, y)
        #Render other sections conditionally...


# Usage in main simulation (hypothetical, not included here):
# hud = SimulationHUD(pygame_screen)
# hud.set_selected_body(selected_body_info_instance)
# In rendering loop, call hud.render() to update HUD display.
 
    def render_hydrology(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, "hydrology_modle"):
            return y
        hydrology: HydrologyModel = self.selected_body.hydrology_model
        self.draw_text("Hydrology:", (x, y), color=self.highlight_color)
        y += 25
        self.draw_text(f"Ocean Coverage: {hydrology.ocean_coverage:.2f}", (x + 10, y))
        y += 20
        self.draw_text(f"Surface Water Volume: {hydrology.surface_water_volume:.2e} m^3", (x + 10, y))
        y += 20
        self.draw_text(f"Evaporation Rate: {hydrology.evaporation_rate:.3f}", (x + 10, y))
        y += 20
        self.draw_text(f"Precipitation Rate: {hydrology.precipitation_rate:.3f}", (x + 10, y))
        y += 30
        return y

    def render_hydrology(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, "hydrology_model"):
            return y
        hydrology: HydrologyModel = self.selected_body.hydrology_model
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
        biosig: BiosignatureAnalyzer = self.selected_body.biosignature_analyzer
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
        g_str = ', '.join(f"{g}: {v:.3f}" for g,v in summary['other_gases'].items())
        self.draw_text(g_str, (x + 110, y))
        y += 30
        self.draw_bar((x + 10, y), 150, 15, summary['biosignature_score'], (220,180,50))
        self.draw_text(f"Score: {summary['biosignature_score']:.2f}", (x + 170, y))
        y += 30
        return y

    def render_geology(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, "geology_model"):
            return y
        geology: GeologyModel = self.selected_body.geology_model
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
            self.draw_text("Max Mountain Height: {:.0f} m".format(max(mt_heights)), (x + 10, y))
            y += 20
        y += 20
        return y

    def render_magnetic_field(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, "magnetic_field_model"):
            return y
        magnetic: MagneticFieldModel = self.selected_body.magnetic_field_model
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
        sw: SpaceWeatherModel = self.selected_body.space_weather_model
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
        eco: EcosystemModel = self.selected_body.ecosystem_model
        summary = eco.get_ecosystem_summary()
        self.draw_text("Ecosystem:", (x, y), color=self.highlight_color)
        y += 25
        for name, val in summary.items():
            self.draw_text(f"{name.replace('_',' ').title()}: {val:.2f}", (x + 10, y))
            y += 20
        y += 30
        return y

    def render_surface(self, x, y):
        if not self.selected_body or not hasattr(self.selected_body, "surface_features"):
            return y
        surf: SurfaceFeatures = self.selected_body.surface_features
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
        if self.sections_expanded.get("hydrology"): y = self.render_hydrology(panel_rect.x + 10, y)
        if self.sections_expanded.get("biosignature"): y = self.render_biosignature(panel_rect.x + 10, y)
        if self.sections_expanded.get("geology"): y = self.render_geology(panel_rect.x + 10, y)
        if self.sections_expanded.get("magnetic"): y = self.render_magnetic_field(panel_rect.x + 10, y)
        if self.sections_expanded.get("space_weather"): y = self.render_space_weather(panel_rect.x + 10, y)
        if self.sections_expanded.get("ecosystem"): y = self.render_ecosystem(panel_rect.x + 10, y)
        if self.sections_expanded.get("surface"): y = self.render_surface(panel_rect.x + 10, y)

    def update(self):
        if not self.selected_body:
            return
        self.render_selected()

    def toggle_section(self,section_name:str):
        if section_name in self.sections_expanded:
            self.sections_expanded[section_name] = not self.sections_expanded[section_name]

    def handle_click(self, pos):
        panel_rect = pygame.Rect(10,10,420,self.screen.get_height() - 20)
        if not panel_rect.collidepoint(pos):
            self.select_body(None)
            return 
        
        x,y = pos
        base_y = 30
        height_per_section = 25

        if 30 <= y < 30 + height_per_section:
            self.toggle_section("habitability")
        elif 30 + height_per_section <= y < 30+2*height_per_section:
            self.toggle_section("atmosphere")
        elif 30 + 2 * height_per_section <= y < 30 + 3 * height_per_section:
            self.toggle_section("climate")
        elif 30 + 3 * height_per_section <= y < 30 + 4 * height_per_section:
            self.toggle_section("hydrology")
        elif 30 + 4 * height_per_section <= y < 30 + 5 * height_per_section:
            self.toggle_section("biosignature")
        elif 30 + 5 * height_per_section <= y < 30 + 6 * height_per_section:
            self.toggle_section("geology")
        elif 30 + 6 * height_per_section <= y < 30 + 7 * height_per_section:
            self.toggle_section("magnetic")
        elif 30 + 7 * height_per_section <= y < 30 + 8 * height_per_section:
            self.toggle_section("space_weather")
        elif 30 + 8 * height_per_section <= y < 30 + 9 * height_per_section:
            self.toggle_section("ecosystem")
        elif 30 + 9 * height_per_section <= y < 30 + 10 * height_per_section:
            self.toggle_section("surface")

    def display_warning(self, text: str):
        w, h = self.screen.get_size()
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        font = pygame.font.SysFont("Arial", 36)
        label = font.render(text, True, (255, 0, 0))
        rect = label.get_rect(center=(w//2, h//2))
        self.screen.blit(label, rect)

    def clear_selection(self):
        self.selected_body = None

    def get_selected_body(self):
        return self.selected_body
