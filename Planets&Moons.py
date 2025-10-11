import pygame
import math
pygame.init()

WIDTH = 850
HEIGHT = 700
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Solar System with Moons')

YELLOW = (255, 255, 0)
BLUE = (100, 149, 237)
RED = (188, 39, 50)
LIGHT_GREY = (200, 200, 200)
PINK = (255, 192, 203)
WHITE = (255, 255, 255)
ORANGE = (255, 165, 0)
BG_COLOR = (10, 10, 30)

class Body:
    AU = 149597870700
    G = 6.67430e-11
    SCALE = 210 / AU
    TIMESTEP = 3600 * 6
    
    def __init__(self, name, x, y, vx, vy, mass, radius, color):
        self.name = name
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        self.radius = radius
        self.color = color
        self.trail = []
    
    def update(self, bodies):
        fx_total = 0
        fy_total = 0
        
        for other in bodies:
            if other is self:
                continue
            
            dx = other.x - self.x
            dy = other.y - self.y
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist > 0:
                force = self.G * self.mass * other.mass / (dist * dist)
                theta = math.atan2(dy, dx)
                fx_total += force * math.cos(theta)
                fy_total += force * math.sin(theta)
        
        self.vx += fx_total / self.mass * self.TIMESTEP
        self.vy += fy_total / self.mass * self.TIMESTEP
        self.x += self.vx * self.TIMESTEP
        self.y += self.vy * self.TIMESTEP
        
        self.trail.append((self.x, self.y))
        if len(self.trail) > 400:
            self.trail.pop(0)
    
    def draw(self, win):
        sx = int(self.x * self.SCALE + WIDTH / 2)
        sy = int(self.y * self.SCALE + HEIGHT / 2)
        pygame.draw.circle(win, self.color, (sx, sy), self.radius)
        
        if len(self.trail) > 2:
            points = []
            for px, py in self.trail[-200:]:
                screen_x = int(px * self.SCALE + WIDTH / 2)
                screen_y = int(py * self.SCALE + HEIGHT / 2)
                points.append((screen_x, screen_y))
            if len(points) > 1:
                pygame.draw.lines(win, self.color, False, points, 1)

def calc_orbit_velocity(M, r):
    return math.sqrt(Body.G * M / r)

def main():
    clock = pygame.time.Clock()
    running = True
    bodies = []
    
    sun_mass = 1.989e30
    sun = Body("Sun", 0, 0, 0, 0, sun_mass, 35, YELLOW)
    bodies.append(sun)
    
    mercury_dist = 0.387 * Body.AU
    mercury_v = calc_orbit_velocity(sun_mass, mercury_dist)
    mercury = Body("Mercury", mercury_dist, 0, 0, mercury_v, 3.285e23, 8, LIGHT_GREY)
    bodies.append(mercury)
    
    venus_dist = 0.723 * Body.AU
    venus_v = calc_orbit_velocity(sun_mass, venus_dist)
    venus = Body("Venus", venus_dist, 0, 0, venus_v, 4.867e24, 14, PINK)
    bodies.append(venus)
    
    earth_dist = 1.0 * Body.AU
    earth_mass = 5.972e24
    earth_v = calc_orbit_velocity(sun_mass, earth_dist)
    earth = Body("Earth", earth_dist, 0, 0, earth_v, earth_mass, 16, BLUE)
    bodies.append(earth)
    
    moon_dist = 384400000
    moon_v_relative = calc_orbit_velocity(earth_mass, moon_dist)
    moon_x = earth.x + moon_dist
    moon_y = earth.y
    moon_vx = earth.vx
    moon_vy = earth.vy + moon_v_relative
    moon = Body("Moon", moon_x, moon_y, moon_vx, moon_vy, 7.342e22, 4, WHITE)
    bodies.append(moon)
    
    mars_dist = 1.524 * Body.AU
    mars_mass = 6.39e23
    mars_v = calc_orbit_velocity(sun_mass, mars_dist)
    mars = Body("Mars", mars_dist, 0, 0, mars_v, mars_mass, 12, RED)
    bodies.append(mars)
    
    phobos_dist = 9376000
    phobos_v_relative = calc_orbit_velocity(mars_mass, phobos_dist)
    phobos_x = mars.x + phobos_dist
    phobos_y = mars.y
    phobos_vx = mars.vx
    phobos_vy = mars.vy + phobos_v_relative
    phobos = Body("Phobos", phobos_x, phobos_y, phobos_vx, phobos_vy, 1.0659e16, 3, ORANGE)
    bodies.append(phobos)
    
    while running:
        clock.tick(60)
        window.fill(BG_COLOR)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        for body in bodies:
            body.update(bodies)
        
        for body in bodies:
            body.draw(window)
        
        pygame.display.update()
    
    pygame.quit()

if __name__ == "__main__":
    main()
