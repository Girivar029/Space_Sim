import math
import random
import json
import time
import pygame

# Initialize pygame immediately
pygame.init()

# Display Constants
WIDTH, HEIGHT = 850, 700  # Fixed typo: widht → WIDTH
FPS = 60
BACKGROUND = (18, 22, 23)  # Fixed: needs 3 values for RGB, not 2

# Physics Constants
G = 6.67430e-11
SOFTENING = 1e9  # Capitalized for consistency
TIMESTEP = 3600 * 12
SCALE = 200 / 1.496e11  # AU to pixels

# Simulation Settings
MAX_BODIES = 100
TRAIL_LENGTH = 200  
COLLISION_ENABLED = True
MERGE_ON_COLLISION = True  

# Visual Settings
GLOW_LAYERS = 3
GLOW_FALLOFF = 8
MIN_BODY_RADIUS = 3
MAX_BODY_RADIUS = 40
TRAIL_ALPHA = 100  

# Color Palettes
STAR_COLORS = [(157, 180, 255), (255, 130, 32), (255, 242, 32)]
PLANET_COLORS = [(246, 255, 104), (255, 161, 65), (255, 90, 0), (133, 38, 114), (23, 180, 109), (145, 167, 0), (121, 165, 224)]
UI_TEXT_COLOR = (50, 150, 200)

# Distance Calculator
def distance(x1, y1, x2, y2):
    dx =x2-x1
    dy = y2-y1
    return math.sqrt(dx*dx+dy*dy)

# Speed Calculator
def vector_magnitude(vx,vy):
    return(vx*vx+vy*vy)

# Orbital Velocity
def orbital_velocity(central_mass,semi_major_axis,eccentricity,true_anamoly):
    r = semi_major_axis * (1- eccentricity * eccentricity)/ (1+eccentricity * math.cos(true_anamoly))
    v_sqr = G * central_mass * (2/r-1/semi_major_axis)
    v = math.sqrt(v_sqr)
    velocity_angle = true_anamoly + math.pi / 2 + math.atan(eccentricity * math.sin(true_anamoly)) / (1+eccentricity*math.cos(true_anamoly)) 
    vx = v * math.cos(velocity_angle)
    vy = v * math.sin(velocity_angle)
    return vx, vy

def cricular_orbital_velocity(central_mass, radius):
    return math.sqrt(G * central_mass / radius)

#Mass to Radius Converter
def mass_to_radii(mass):
    base_radius = 4 + math.log10(mass) - 22
    radius = max(MIN_BODY_RADIUS,min(MIN_BODY_RADIUS),base_radius)
    return int(radius)

#Random color genrator
def random_color():
    r =random.randint(80, 225)
    g = random.randint(80, 225)
    b = random.randint(80,255)
    return (r,g,b)

def random_star_color():
    return random.choice(STAR_COLORS)

def random_planet_color():
    return random.choice (PLANET_COLORS)

def speed_to_color(speed):
    max_speed = 50000
    normalized =min(speed/max_speed, 1.0)

    red = int(100 + normalized * 155)
    green = 100(100)
    blue = int(255 - normalized * 155)

    return(blue, green ,red)