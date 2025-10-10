import pygame
pygame.init()

window = pygame.display.set_mode((600, 600))
pygame.display.set_caption('Attempt_1')
WHITE = (255, 255, 255)
BLUE = (0,0 ,255)


class Planet:
    AU = 149.6e6 * 1000
    G = 6.67428e-11
    SCALE = 250 / AU
    TIMESTEP = 3600 * 24  # 1 Day

    def __init__(self, x, y, radius, color, mass):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.mass = mass  # in KG

        self.sun = False
        self.distance_to_sun = 0
        self.orbit = []

        self.x_vel = 0
        self.y_vel = 0

    def draw(self, win):
        x = int(self.x * self.SCALE + 300)
        y = int(self.y * self.SCALE + 300)
        pygame.draw.circle(win, self.color, (x, y), self.radius)


def main():
    run = True
    clock = pygame.time.Clock()
    sun = Planet(0, 0, 30, (255, 255, 0), 198892 * 10 ** 30)
    sun.sun = True

    earth = Planet(-1 * Planet.AU, 0,16,BLUE,5.9742*10**24)

    # Add the sun and potentially other planets to this list
    planets = [sun]

    while run:
        clock.tick(60)
        window.fill((0, 0, 0))  # Clear screen each frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        for planet in planets:
            planet.draw(window)

        pygame.display.update()

    pygame.quit()


main()
