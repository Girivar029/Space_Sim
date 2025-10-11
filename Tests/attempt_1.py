import pygame
import math
pygame.init()

# Window setup
WIDTH, HEIGHT = 850, 700
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Attempt_1')

# Colors in RGB
YELLOW = (255, 255, 0)
BLUE = (100, 149, 237)    # Earth-like blue
RED = (188, 39, 50)
DARK_GREY = (105, 105, 105)
ORANGE = (155, 115, 150)    # Mercury-like color
WHITE = (255, 255, 255)
BG_COLOR = (10, 10, 30)  # Dark space background

class Planet:
    AU = 149.6e6 * 1000                  # Astronomical Unit in meters
    SCALE = 210 / AU                     # Scale: 1 AU = 250 pixels
    G = 6.67430e-11                     # Gravitational constant
    TIMESTEP = 3600 * 24                # One day per simulation step (seconds)

    def __init__(self, x, y, radius, color, mass):
        self.x = x                      # position in meters
        self.y = y
        self.radius = radius            # radius in pixels for display
        self.color = color
        self.mass = mass                # kg
        self.x_vel = 0                  # velocity in m/s
        self.y_vel = 0
        self.orbit = []

    def draw(self, win):
        # Convert position in meters to pixels and center on screen
        x = self.x * self.SCALE + WIDTH / 2
        y = self.y * self.SCALE + HEIGHT / 2
        pygame.draw.circle(win, self.color, (int(x), int(y)), self.radius)

        # Draw orbit trail
        if len(self.orbit) > 2:
            points = []
            for point in self.orbit[-200:]:  # last 200 points only
                px = point[0] * self.SCALE + WIDTH / 2
                py = point[1] * self.SCALE + HEIGHT / 2
                points.append((int(px), int(py)))
            if len(points) > 1:
                pygame.draw.lines(win, self.color, False, points, 1)

    def attraction(self, other):
        dx = other.x - self.x
        dy = other.y - self.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance == 0:
            return 0, 0

        force = self.G * self.mass * other.mass / distance**2
        theta = math.atan2(dy, dx)
        force_x = math.cos(theta) * force
        force_y = math.sin(theta) * force
        return force_x, force_y

    def update_position(self, planets):
        total_fx = total_fy = 0

        for planet in planets:
            if planet == self:
                continue
            fx, fy = self.attraction(planet)
            total_fx += fx
            total_fy += fy

        # Update velocities based on net force
        self.x_vel += total_fx / self.mass * self.TIMESTEP
        self.y_vel += total_fy / self.mass * self.TIMESTEP

        # Update position based on velocity
        self.x += self.x_vel * self.TIMESTEP
        self.y += self.y_vel * self.TIMESTEP

        # Store position for orbit trail
        self.orbit.append((self.x, self.y))
        if len(self.orbit) > 500:
            self.orbit.pop(0)

def main():
    run = True
    clock = pygame.time.Clock()

    # Sun
    sun_mass = 1.9885e30  # kg
    sun = Planet(0, 0, 30, YELLOW, sun_mass)

    # Planets setup: Distance from sun (meters), radius (pixels), color, mass (kg), initial velocity (m/s)
    # Orbital velocities taken from approximately circular orbit speed formula
    mercury = Planet(0.387 * Planet.AU, 0, 8, ORANGE, 3.3011e23)
    mercury.y_vel = -47.4e3

    venus = Planet(0.723 * Planet.AU, 0, 14, WHITE, 4.8675e24)
    venus.y_vel = 35.0e3

    earth = Planet(1.0 * Planet.AU, 0, 16, BLUE, 5.97237e24)
    earth.y_vel = 29.78e3

    mars = Planet(1.524 * Planet.AU, 0, 12, RED, 6.4171e23)
    mars.y_vel = 24.077e3

    planets = [sun, mercury, venus, earth, mars]

    while run:
        clock.tick(60)
        window.fill(BG_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        for planet in planets:
            planet.update_position(planets)
            planet.draw(window)

        pygame.display.update()
    pygame.quit()

main()
