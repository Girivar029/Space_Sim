import pygame
import math
import random

pygame.init()

WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('N-Body Gravity Simulation')
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
G = 6.67430e-11
SOFTENING = 1e9
DT = 86400 * 2
SCALE = 200 / (1.496e11)

class Body:
    def __init__(self, x, y, vx, vy, mass, radius, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        self.radius = radius
        self.color = color
        self.ax = 0
        self.ay = 0
        self.trail = []
    
    def calculate_acceleration(self, bodies):
        self.ax = 0
        self.ay = 0
        
        for other in bodies:
            if other is self:
                continue
            
            dx = other.x - self.x
            dy = other.y - self.y
            distance_sq = dx * dx + dy * dy + SOFTENING * SOFTENING
            distance = math.sqrt(distance_sq)
            
            force = G * other.mass / distance_sq
            self.ax += force * dx / distance
            self.ay += force * dy / distance
    
    def update_velocity(self):
        self.vx += self.ax * DT
        self.vy += self.ay * DT
    
    def update_position(self):
        self.x += self.vx * DT
        self.y += self.vy * DT
        
        self.trail.append((self.x, self.y))
        if len(self.trail) > 200:
            self.trail.pop(0)
    
    def draw(self, surface):
        screen_x = int(self.x * SCALE + WIDTH / 2)
        screen_y = int(self.y * SCALE + HEIGHT / 2)
        
        if len(self.trail) > 2:
            trail_points = []
            for px, py in self.trail:
                tx = int(px * SCALE + WIDTH / 2)
                ty = int(py * SCALE + HEIGHT / 2)
                if 0 <= tx < WIDTH and 0 <= ty < HEIGHT:
                    trail_points.append((tx, ty))
            if len(trail_points) > 1:
                pygame.draw.lines(surface, self.color, False, trail_points, 1)
        
        if 0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT:
            for i in range(4):
                glow_radius = self.radius + i * 2
                alpha = max(30 - i * 8, 0)
                s = pygame.Surface((glow_radius * 4, glow_radius * 4), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.color, alpha), (glow_radius * 2, glow_radius * 2), glow_radius)
                surface.blit(s, (screen_x - glow_radius * 2, screen_y - glow_radius * 2), special_flags=pygame.BLEND_RGBA_ADD)
            
            pygame.draw.circle(surface, self.color, (screen_x, screen_y), self.radius)

def create_solar_system():
    bodies = []
    
    sun = Body(0, 0, 0, 0, 1.989e30, 30, (255, 220, 0))
    bodies.append(sun)
    
    earth_dist = 1.496e11
    earth_vel = math.sqrt(G * sun.mass / earth_dist)
    earth = Body(earth_dist, 0, 0, earth_vel, 5.972e24, 16, (100, 149, 237))
    bodies.append(earth)
    
    mars_dist = 2.279e11
    mars_vel = math.sqrt(G * sun.mass / mars_dist)
    mars = Body(mars_dist, 0, 0, mars_vel, 6.39e23, 12, (188, 39, 50))
    bodies.append(mars)
    
    venus_dist = 1.082e11
    venus_vel = math.sqrt(G * sun.mass / venus_dist)
    venus = Body(venus_dist, 0, 0, venus_vel, 4.867e24, 14, (255, 198, 73))
    bodies.append(venus)
    
    return bodies

def create_random_system(n=50):
    bodies = []
    
    center_mass = random.uniform(1e30, 2e30)
    center = Body(0, 0, 0, 0, center_mass, 35, (255, 220, 100))
    bodies.append(center)
    
    for i in range(n):
        distance = random.uniform(5e10, 4e11)
        angle = random.uniform(0, 2 * math.pi)
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        
        orbital_vel = math.sqrt(G * center_mass / distance) * random.uniform(0.8, 1.2)
        vx = -orbital_vel * math.sin(angle)
        vy = orbital_vel * math.cos(angle)
        
        mass = random.uniform(1e22, 1e25)
        radius = int(4 + math.log10(mass) - 20)
        color = (random.randint(80, 255), random.randint(80, 255), random.randint(80, 255))
        
        bodies.append(Body(x, y, vx, vy, mass, radius, color))
    
    return bodies

def main():
    running = True
    
    bodies = create_solar_system()
    
    font = pygame.font.Font(None, 24)
    
    while running:
        clock.tick(60)
        screen.fill(BLACK)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    bodies = create_random_system(50)
                if event.key == pygame.K_s:
                    bodies = create_solar_system()
        
        for body in bodies:
            body.calculate_acceleration(bodies)
        
        for body in bodies:
            body.update_velocity()
        
        for body in bodies:
            body.update_position()
        
        for body in bodies:
            body.draw(screen)
        
        text = font.render(f'Bodies: {len(bodies)} | R: Random | S: Solar System', True, WHITE)
        screen.blit(text, (10, 10))
        
        pygame.display.flip()
    
    pygame.quit()

main()