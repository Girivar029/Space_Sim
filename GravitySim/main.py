import math
import random
import json
import time
import pygame
import asyncio

pygame.init()

WIDTH, HEIGHT = 850, 700
FPS = 60
BACKGROUND = (18, 22, 23)

G = 6.67430e-11
SOFTENING = 1e9
TIMESTEP = 3600 * 12
SCALE = 200 / 1.496e11

MAX_BODIES = 100
TRAIL_LENGTH = 200
COLLISION_ENABLED = True
MERGE_ON_COLLISION = True

GLOW_LAYERS = 3
GLOW_FALLOFF = 8
MIN_BODY_RADIUS = 3
MAX_BODY_RADIUS = 40
TRAIL_ALPHA = 100

STAR_COLORS = [(157, 180, 255), (255, 130, 32), (255, 242, 32)]
PLANET_COLORS = [(246, 255, 104), (255, 161, 65), (255, 90, 0), (133, 38, 114), (23, 180, 109), (145, 167, 0), (121, 165, 224)]
UI_TEXT_COLOR = (50, 150, 200)


def distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)


def vector_magnitude(vx, vy):
    return math.sqrt(vx * vx + vy * vy)


def orbital_velocity(central_mass, semi_major_axis, eccentricity, true_anomaly):
    r = semi_major_axis * (1 - eccentricity * eccentricity) / (1 + eccentricity * math.cos(true_anomaly))
    v_sqr = G * central_mass * (2 / r - 1 / semi_major_axis)
    v = math.sqrt(v_sqr)
    velocity_angle = true_anomaly + math.pi / 2 + math.atan(eccentricity * math.sin(true_anomaly) / (1 + eccentricity * math.cos(true_anomaly)))
    vx = v * math.cos(velocity_angle)
    vy = v * math.sin(velocity_angle)
    return vx, vy


def circular_orbital_velocity(central_mass, radius):
    return math.sqrt(G * central_mass / radius)


def mass_to_radius(mass):
    base_radius = 4 + math.log10(mass) - 22
    radius = max(MIN_BODY_RADIUS, min(MAX_BODY_RADIUS, base_radius))
    return int(radius)


def random_color():
    r = random.randint(80, 255)
    g = random.randint(80, 255)
    b = random.randint(80, 255)
    return (r, g, b)


def random_star_color():
    return random.choice(STAR_COLORS)


def random_planet_color():
    return random.choice(PLANET_COLORS)


def speed_to_color(speed):
    max_speed = 50000
    normalized = min(speed / max_speed, 1.0)
    red = int(100 + normalized * 155)
    green = 100
    blue = int(255 - normalized * 155)
    return (red, green, blue)


class Camera:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.zoom = 1.0
        self.target = None
        self.follow_speed = 0.1

    def world_to_screen(self, world_x, world_y):
        relative_x = world_x - self.x
        relative_y = world_y - self.y
        screen_x = relative_x * self.zoom * SCALE + WIDTH / 2
        screen_y = relative_y * self.zoom * SCALE + HEIGHT / 2
        return int(screen_x), int(screen_y)

    def update(self):
        if self.target is not None:
            dx = self.target.x - self.x
            dy = self.target.y - self.y
            self.x += dx * self.follow_speed
            self.y += dy * self.follow_speed

    def zoom_in(self):
        self.zoom *= 1.1
        if self.zoom > 10.0:
            self.zoom = 10.0

    def zoom_out(self):
        self.zoom *= 0.9
        if self.zoom < 0.1:
            self.zoom = 0.1

    def pan(self, dx, dy):
        world_dx = dx / (self.zoom * SCALE)
        world_dy = dy / (self.zoom * SCALE)
        self.x -= world_dx
        self.y -= world_dy

    def follow(self, body):
        self.target = body

    def unfollow(self):
        self.target = None

    def reset(self):
        self.x = 0
        self.y = 0
        self.zoom = 1.0
        self.target = None


class Body:
    def __init__(self, x, y, vx, vy, mass, radius=None, color=None, is_star=False, name="Body"):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = 0
        self.ay = 0
        self.mass = mass
        
        if radius is None:
            self.radius = mass_to_radius(mass)
        else:
            self.radius = radius
        
        if color is None:
            if is_star:
                self.color = random_star_color()
            else:
                self.color = random_planet_color()
        else:
            self.color = color
        
        self.is_star = is_star
        self.name = name
        self.trail = []

    def calculate_forces(self, other):
        dx = other.x - self.x
        dy = other.y - self.y
        distance_squared = dx * dx + dy * dy + SOFTENING * SOFTENING
        distance = math.sqrt(distance_squared)
        force_magnitude = G * self.mass * other.mass / distance_squared
        fx = force_magnitude * dx / distance
        fy = force_magnitude * dy / distance
        return fx, fy

    def calculate_acceleration(self, all_bodies):
        total_fx = 0
        total_fy = 0
        
        for other in all_bodies:
            if other is self:
                continue
            
            fx, fy = self.calculate_forces(other)
            total_fx += fx
            total_fy += fy
        
        self.ax = total_fx / self.mass
        self.ay = total_fy / self.mass

    def update_velocity(self, dt):
        self.vx += self.ax * dt
        self.vy += self.ay * dt

    def update_position(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.trail.append((self.x, self.y))
        
        if len(self.trail) > TRAIL_LENGTH:
            self.trail.pop(0)

    def draw(self, surface, camera):
        screen_x, screen_y = camera.world_to_screen(self.x, self.y)
        buffer = 100
        
        if not (-buffer < screen_x < WIDTH + buffer and -buffer < screen_y < HEIGHT + buffer):
            return
        
        if len(self.trail) > 2:
            trail_points = []
            for px, py in self.trail:
                tx, ty = camera.world_to_screen(px, py)
                if -100 < tx < WIDTH + 100 and -100 < ty < HEIGHT + 100:
                    trail_points.append((tx, ty))
            
            if len(trail_points) > 1:
                try:
                    pygame.draw.lines(surface, self.color, False, trail_points, 1)
                except:
                    pass
        
        for i in range(GLOW_LAYERS):
            glow_radius = self.radius + i * 3
            alpha = max(60 - i * GLOW_FALLOFF, 0)
            
            if alpha > 0:
                glow_surface = pygame.Surface((glow_radius * 4, glow_radius * 4), pygame.SRCALPHA)
                pygame.draw.circle(
                    glow_surface,
                    (*self.color, alpha),
                    (glow_radius * 2, glow_radius * 2),
                    glow_radius
                )
                surface.blit(
                    glow_surface,
                    (screen_x - glow_radius * 2, screen_y - glow_radius * 2),
                    special_flags=pygame.BLEND_RGBA_ADD
                )
        
        pygame.draw.circle(surface, self.color, (screen_x, screen_y), self.radius)

    def check_collision(self, other):
        dist = distance(self.x, self.y, other.x, other.y)
        radius_world_self = self.radius / SCALE
        radius_world_other = other.radius / SCALE
        return dist < (radius_world_self + radius_world_other)

    def get_speed(self):
        return vector_magnitude(self.vx, self.vy)


def merge_bodies(body1, body2):
    total_mass = body1.mass + body2.mass
    new_vx = (body1.vx * body1.mass + body2.vx * body2.mass) / total_mass
    new_vy = (body1.vy * body1.mass + body2.vy * body2.mass) / total_mass
    new_x = (body1.x * body1.mass + body2.x * body2.mass) / total_mass
    new_y = (body1.y * body1.mass + body2.y * body2.mass) / total_mass
    
    explosion_pos = (new_x, new_y)
    
    body1.mass = total_mass
    body1.vx = new_vx
    body1.vy = new_vy
    body1.x = new_x
    body1.y = new_y
    body1.radius = mass_to_radius(total_mass)
    
    if body1.is_star or body2.is_star:
        body1.is_star = True
        body1.color = random_star_color()
    else:
        body1.color = random_planet_color()
    
    body1.name = f"{body1.name}+{body2.name}"
    body1.trail = []
    return explosion_pos


def check_all_collisions(bodies):
    explosions = []
    i = 0
    
    while i < len(bodies):
        j = i + 1
        while j < len(bodies):
            if bodies[i].check_collision(bodies[j]):
                explosion_pos = merge_bodies(bodies[i], bodies[j])
                explosions.append({
                    'x': explosion_pos[0],
                    'y': explosion_pos[1],
                    'frame': 0,
                    'max_frames': 30
                })
                bodies.pop(j)
            else:
                j += 1
        i += 1
    
    return explosions


def update_explosions(explosions):
    i = 0
    while i < len(explosions):
        explosions[i]['frame'] += 1
        if explosions[i]['frame'] >= explosions[i]['max_frames']:
            explosions.pop(i)
        else:
            i += 1


def draw_explosions(surface, explosions, camera):
    for explosion in explosions:
        screen_x, screen_y = camera.world_to_screen(explosion['x'], explosion['y'])
        progress = explosion['frame'] / explosion['max_frames']
        radius = int(10 + progress * 30)
        alpha = int(255 * (1 - progress))
        
        if 0 < screen_x < WIDTH and 0 < screen_y < HEIGHT:
            explosion_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            color = (255, int(100 + progress * 155), 0, alpha)
            pygame.draw.circle(
                explosion_surf,
                color,
                (radius * 2, radius * 2),
                radius,
                width=3
            )
            
            surface.blit(
                explosion_surf,
                (screen_x - radius * 2, screen_y - radius * 2),
                special_flags=pygame.BLEND_RGBA_ADD
            )


def create_solar_system():
    bodies = []
    
    sun = Body(
        x=0,
        y=0,
        vx=0,
        vy=0,
        mass=1.989e30,
        radius=30,
        color=(255, 220, 0),
        is_star=True,
        name="Sun"
    )
    bodies.append(sun)
    
    mercury_dist = 0.387 * 1.496e11
    mercury_vel = circular_orbital_velocity(sun.mass, mercury_dist)
    mercury = Body(
        x=mercury_dist,
        y=0,
        vx=0,
        vy=mercury_vel,
        mass=3.285e23,
        color=(169, 169, 169),
        name="Mercury"
    )
    bodies.append(mercury)
    
    venus_dist = 0.723 * 1.496e11
    venus_vel = circular_orbital_velocity(sun.mass, venus_dist)
    venus = Body(
        x=venus_dist,
        y=0,
        vx=0,
        vy=venus_vel,
        mass=4.876e24,
        color=(255, 198, 173),
        name="Venus"
    )
    bodies.append(venus)
    
    earth_dist = 1.0 * 1.496e11
    earth_vel = circular_orbital_velocity(sun.mass, earth_dist)
    earth = Body(
        x=earth_dist,
        y=0,
        vx=0,
        vy=earth_vel,
        mass=5.972e24,
        color=(100, 149, 237),
        name="Earth"
    )
    bodies.append(earth)
    
    mars_dist = 1.524 * 1.496e11
    mars_vel = circular_orbital_velocity(sun.mass, mars_dist)
    mars = Body(
        x=mars_dist,
        y=0,
        vx=0,
        vy=mars_vel,
        mass=6.39e23,
        color=(188, 39, 50),
        name="Mars"
    )
    bodies.append(mars)
    
    return bodies


def create_random_chaos(n_bodies=50):
    bodies = []
    num_stars = random.randint(1, 2)
    
    for i in range(num_stars):
        x = random.uniform(-1e10, 1e10)
        y = random.uniform(-1e10, 1e10)
        vx = random.uniform(-1000, 1000)
        vy = random.uniform(-1000, 1000)
        mass = random.uniform(5e29, 2e30)
        
        star = Body(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            mass=mass,
            color=random_star_color(),
            is_star=True,
            name=f"Star{i+1}"
        )
        bodies.append(star)
    
    total_star_mass = sum(body.mass for body in bodies)
    
    for i in range(n_bodies - num_stars):
        distance = random.uniform(5e10, 4e11)
        angle = random.uniform(0, 2 * math.pi)
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        
        orbital_vel = circular_orbital_velocity(total_star_mass, distance)
        velocity_factor = random.uniform(0.7, 1.3)
        
        vx = -orbital_vel * velocity_factor * math.sin(angle)
        vy = orbital_vel * velocity_factor * math.cos(angle)
        
        mass = 10 ** random.uniform(22, 26)
        
        body = Body(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            mass=mass,
            color=random_planet_color(),
            name=f"Body{i+1}"
        )
        bodies.append(body)
    
    return bodies


def create_binary_stars():
    bodies = []
    separation = 3e11
    star_mass = 1e30
    orbital_vel = math.sqrt(G * star_mass / separation)
    
    star1 = Body(
        x=-separation/2,
        y=0,
        vx=0,
        vy=orbital_vel,
        mass=star_mass,
        radius=25,
        color=(255, 200, 100),
        is_star=True,
        name="StarA"
    )
    bodies.append(star1)
    
    star2 = Body(
        x=separation/2,
        y=0,
        vx=0,
        vy=-orbital_vel,
        mass=star_mass,
        radius=25,
        color=(100, 200, 255),
        is_star=True,
        name="StarB"
    )
    bodies.append(star2)
    
    for i in range(15):
        distance = random.uniform(5e11, 8e11)
        angle = random.uniform(0, 2 * math.pi)
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        
        orbital_v = circular_orbital_velocity(star_mass * 2, distance)
        vx = -orbital_v * math.sin(angle)
        vy = orbital_v * math.cos(angle)
        
        mass = 10 ** random.uniform(23, 25)
        
        planet = Body(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            mass=mass,
            color=random_planet_color(),
            name=f"Planet{i+1}"
        )
        bodies.append(planet)
    
    return bodies


def create_galaxy_disk(n_bodies=100):
    bodies = []
    
    black_hole = Body(
        x=0,
        y=0,
        vx=0,
        vy=0,
        mass=5e30,
        radius=35,
        color=(50, 0, 100),
        is_star=True,
        name="BlackHole"
    )
    bodies.append(black_hole)
    
    for i in range(n_bodies):
        distance = random.expovariate(1 / 1e11)
        distance = min(distance, 5e11)
        distance = max(distance, 1e10)
        
        angle = random.uniform(0, 2 * math.pi)
        z_offset = random.uniform(-5e9, 5e9)
        
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        
        orbital_v = circular_orbital_velocity(black_hole.mass, distance)
        vx = -orbital_v * math.sin(angle) + random.uniform(-2000, 2000)
        vy = orbital_v * math.cos(angle) + random.uniform(-2000, 2000)
        
        mass = 10 ** random.uniform(24, 29)
        
        r = int(100 + 155 * (distance / 5e11))
        b = int(255 - 155 * (distance / 5e11))
        color = (r, 100, b)
        
        body = Body(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            mass=mass,
            color=color,
            name=f"Star{i+1}"
        )
        bodies.append(body)
    
    return bodies


def create_figure_eight():
    bodies = []
    mass = 1e30
    
    body1 = Body(
        x=-9.7e10,
        y=0,
        vx=0,
        vy=-9.32e4,
        mass=mass,
        radius=20,
        color=(255, 100, 100),
        name="Body1"
    )
    bodies.append(body1)
    
    body2 = Body(
        x=9.7e10,
        y=0,
        vx=0,
        vy=4.66e4,
        mass=mass,
        radius=20,
        color=(100, 255, 100),
        name="Body2"
    )
    bodies.append(body2)
    
    body3 = Body(
        x=0,
        y=0,
        vx=0,
        vy=4.66e4,
        mass=mass,
        radius=20,
        color=(100, 100, 255),
        name="Body3"
    )
    bodies.append(body3)
    
    return bodies


def generate_background_stars(n=200):
    stars = []
    for _ in range(n):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        stars.append((x, y))
    return stars


def draw_background(surface, stars):
    surface.fill(BACKGROUND)
    for (x, y) in stars:
        surface.set_at((x, y), (255, 255, 255))


def draw_ui(surface, bodies, fps, paused):
    font = pygame.font.SysFont(None, 24)
    
    text1 = font.render(f'Bodies: {len(bodies)}', True, UI_TEXT_COLOR)
    surface.blit(text1, (10, 10))
    
    text2 = font.render(f'FPS: {int(fps)}', True, UI_TEXT_COLOR)
    surface.blit(text2, (10, 30))
    
    if paused:
        text3 = font.render('PAUSED', True, (255, 100, 100))
        surface.blit(text3, (WIDTH//2 - 40, 10))
    
    hints = 'SPACE:Pause  R:Reset  S:Solar  C:Chaos  B:Binary  G:Galaxy'
    text4 = font.render(hints, True, UI_TEXT_COLOR)
    surface.blit(text4, (10, HEIGHT - 30))


async def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Test_Core_Code')
    clock = pygame.time.Clock()
    
    camera = Camera()
    background_stars = generate_background_stars(200)
    bodies = create_solar_system()


    paused = False
    explosions = []
    current_timestep = TIMESTEP
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    
                elif event.key == pygame.K_r:
                    bodies = create_solar_system()
                    camera.reset()
                    explosions = []
                    
                elif event.key == pygame.K_s:
                    bodies = create_solar_system()
                    camera.reset()
                    explosions = []
                    
                elif event.key == pygame.K_c:
                    bodies = create_random_chaos(50)
                    camera.reset()
                    explosions = []
                    
                elif event.key == pygame.K_b:
                    bodies = create_binary_stars()
                    camera.reset()
                    explosions = []
                    
                elif event.key == pygame.K_g:
                    bodies = create_galaxy_disk(80)
                    camera.reset()
                    explosions = []
                    
                elif event.key == pygame.K_f:
                    bodies = create_figure_eight()
                    camera.reset()
                    explosions = []
                    
                elif event.key == pygame.K_LEFT:
                    camera.pan(-50, 0)
                elif event.key == pygame.K_RIGHT:
                    camera.pan(50, 0)
                elif event.key == pygame.K_UP:
                    camera.pan(0, -50)
                elif event.key == pygame.K_DOWN:
                    camera.pan(0, 50)
                    
                elif event.key == pygame.K_EQUALS:
                    current_timestep *= 1.5
                elif event.key == pygame.K_MINUS:
                    current_timestep /= 1.5
                elif event.key == pygame.K_0:
                    current_timestep = TIMESTEP
                    
                elif event.key == pygame.K_ESCAPE:
                    running = False
                
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    camera.zoom_in()
                elif event.y < 0:
                    camera.zoom_out()
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_x, mouse_y = event.pos
                    closest_body = None
                    min_dist = 50
                    
                    for body in bodies:
                        sx, sy = camera.world_to_screen(body.x, body.y)
                        dist = math.sqrt((sx - mouse_x)**2 + (sy - mouse_y)**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_body = body
                    
                    if closest_body:
                        camera.follow(closest_body)
                        # If you want HUD to show info on this selected body, convert & pass it here
                        # hud.select_body(BodyInfo(closest_body, ...)) # fill out parameters
                    else:
                        camera.unfollow()
                        # hud.select_body(None)
        
        if not paused:
            for body in bodies:
                body.calculate_acceleration(bodies)
            
            for body in bodies:
                body.update_velocity(current_timestep)
            
            for body in bodies:
                body.update_position(current_timestep)
            
            if COLLISION_ENABLED and MERGE_ON_COLLISION:
                new_explosions = check_all_collisions(bodies)
                explosions.extend(new_explosions)
            
            update_explosions(explosions)
            camera.update()
        
        draw_background(screen, background_stars)
        
        for body in bodies:
            body.draw(screen, camera)
        
        draw_explosions(screen, explosions, camera)
        draw_ui(screen, bodies, clock.get_fps(), paused)
        
        
        pygame.display.flip()
        clock.tick(FPS)
        await asyncio.sleep(0)
    
    pygame.quit()

asyncio.run(main())