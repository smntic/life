import matplotlib.pyplot as plt
import numpy as np
import pygame
import time
from math import sin, cos
from scipy.signal import convolve2d
from typing import Callable

class GameOfLife:
    def __init__(self, rows: int, cols: int) -> None:
        # Adjacent cells convolution
        self.kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]], dtype=int)

        # Game parameters
        self.becomes_alive = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
        self.stays_alive = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=bool)

        # Grid size
        self.rows, self.cols = rows, cols
        self.cell_size = 5

        # Game state
        self.state = np.random.randint(0, 2, (self.rows, self.cols), dtype=bool)

        # Pygame setup
        self.screen = None # Wait until show function to set up screen
        self.background_colour = (0, 0, 0)
        self.cell_colour = (255, 255, 255)

    def init_rules_random(self) -> None:
        self.becomes_alive = np.random.randint(0, 2, 9, dtype=bool)
        self.stays_alive = np.random.randint(0, 2, 9, dtype=bool)

    def init_rules_set(self, becomes_alive: set[int], stays_alive: set[int]) -> None:
        for i in becomes_alive.union(stays_alive):
            if i < 0 or i > 8:
                raise ValueError('Rules sets must only contain values from 0 to 8 (inclusive)')
        self.becomes_alive = np.array([i in becomes_alive for i in range(0, 9)], dtype=bool)
        self.stays_alive = np.array([i in stays_alive for i in range(0, 9)], dtype=bool)

    def init_rules_vector(self, becomes_alive: np.ndarray, stays_alive: np.ndarray) -> None:
        if becomes_alive.shape != (9,) or stays_alive.shape != (9,) or becomes_alive.dtype != bool or stays_alive.dtype != bool:
            raise ValueError('Rules vectors must be 1D boolean arrays of size 9')
        self.becomes_alive = becomes_alive
        self.stays_alive = stays_alive

    def init_state(self, distribution: Callable[[int, int], bool]) -> None:
        self.state = np.array([[distribution(r, c) for c in range(self.cols)] for r in range(self.rows)], dtype=int)

    def init_state_uniform(self, p) -> None:
        self.init_state(lambda *_: np.random.rand() < p)

    def update(self, steps: int = 1) -> None:
        for _ in range(steps):
            neighbors = convolve2d(self.state, self.kernel, mode="same", boundary="wrap")
            born = self.becomes_alive[neighbors] & ~self.state
            survive = self.stays_alive[neighbors] & self.state
            self.state = born | survive

    def show(self) -> None:
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.cols * self.cell_size, self.rows * self.cell_size))
            pygame.display.set_caption('Game of Life')

        self.screen.fill(self.background_colour)
        for y in range(self.rows):
            for x in range(self.cols):
                if self.state[y, x]:
                    pygame.draw.rect(
                        self.screen,
                        self.cell_colour,
                        (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    )
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()

    def entropy(self) -> float:
        cells = self.state.flatten()
        total = cells.size

        # Count zeros and ones
        counts = np.bincount(cells, minlength=2)
        probs = counts / total

        # Compute entropy safely
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy


def get_entropies(game: GameOfLife, steps: int) -> list[float]:
    entropies = []
    for _ in range(steps):
        entropies.append(game.entropy())
        game.update()
    return entropies


pygame.init()

def explore():
    game = GameOfLife(100, 100)

    steps = 800

    game.init_rules_set({3}, {2, 3})
    game.init_state_uniform(0.5)
    b3s23_entropies = get_entropies(game, steps)

    rules = []

    for _ in range(100):
        game.init_rules_random()
        game.init_state_uniform(0.5)
        entropies = get_entropies(game, steps)

        r = np.corrcoef(b3s23_entropies, entropies)[0, 1]
        rules.append((r, game.becomes_alive, game.stays_alive))

    rules.sort(reverse=True)

    for i, (r, becomes_alive, stays_alive) in enumerate(rules):
        print(f'#{i+1}: {r} -- { {i for i in range(9) if becomes_alive[i]} }, { {i for i in range(9) if stays_alive[i]} }')

    #     plt.plot(entropies, label=f'{list(game.becomes_alive)} {list(game.stays_alive)}')
    #
    # plt.legend()
    # plt.show()

def test():
    game = GameOfLife(100, 100)
    # game.init_rules_set({4, 7}, {2, 3, 4, 7})
    # game.init_rules_set({8,3,5},{ ,3,4,7})
    # game.init_rules_set({3,6,7},{2,3})
    # game.init_rules_set({0,8,2,7},{0,1,2,4,5,6,7,8})
    # game.init_state(lambda r, c: sin(r)*cos(c) > 0.5)
    game.init_state_uniform(0.5)

    running = True
    while running:
        game.show()
        game.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        time.sleep(0.0167)
        # time.sleep(0.2)

    game.close()

# explore()
test()

