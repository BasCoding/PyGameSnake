import numpy as np
import pygame

from pygamesnake.game.snake import Snake, Food

# clockwise ordering of headings, used to turn left/right relative to the snake
CLOCKWISE = ['right', 'down', 'left', 'up']


class SnakeEnv:
    """
    A thin reinforcement-learning wrapper around the existing Snake/Food game.

    It exposes the classic RL interface:
        - reset()      -> start a new game
        - get_state()  -> an 11-value description of the situation the agent sees
        - step(action) -> apply one relative move and return (reward, done, score)
    """

    def __init__(self, render=True, speed=40):
        # game settings (mirror the rest of the project)
        self.ScreenWide = 800
        self.ScreenHeight = 600
        self.SnakeStartX = 0
        self.SnakeStartY = 0
        self.SnakeWide = 40
        self.SnakeHeight = 40
        self.FoodWide = 40
        self.FoodHeight = 40

        self.render_enabled = render
        self.speed = speed
        self.hud_lines = []

        if self.render_enabled:
            pygame.display.init()
            pygame.font.init()
            self.font = pygame.font.SysFont('comicsansms', 22)
            self.screen = pygame.display.set_mode((self.ScreenWide, self.ScreenHeight))
            pygame.display.set_caption('Snake - reinforcement learning')
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        """Start a fresh game."""
        self.snake = Snake(self.SnakeStartX, self.SnakeStartY, self.SnakeWide, self.SnakeHeight)
        self.food = Food(self.ScreenWide, self.ScreenHeight, self.FoodWide, self.FoodHeight,
                         self.SnakeWide, self.SnakeHeight)
        self.frame_iteration = 0

    def _move_point(self, x, y, direction):
        """Return the grid cell reached by stepping once in the given direction."""
        if direction == 'up':
            return x, y - self.SnakeHeight
        if direction == 'down':
            return x, y + self.SnakeHeight
        if direction == 'left':
            return x - self.SnakeWide, y
        return x + self.SnakeWide, y  # right

    def _is_collision(self, x, y):
        """True if the cell (x, y) is a wall or part of the snake body."""
        if (x < 0 or x > self.ScreenWide - self.SnakeWide or
                y < 0 or y > self.ScreenHeight - self.SnakeHeight):
            return True
        for block in self.snake.SnakeBody:
            if block.x == x and block.y == y:
                return True
        return False

    def get_state(self):
        """
        Build the 11-value state the agent learns from:
            - 3 danger flags (straight, right turn, left turn)
            - 4 current-direction flags (one-hot)
            - 4 food-location flags (left, right, up, down of the head)
        Keeping the state small and relative is what makes this problem easy to learn.
        """
        head = self.snake.SnakeHead
        direction = self.snake.SnakeDirection
        idx = CLOCKWISE.index(direction)

        point_straight = self._move_point(head.x, head.y, CLOCKWISE[idx])
        point_right = self._move_point(head.x, head.y, CLOCKWISE[(idx + 1) % 4])
        point_left = self._move_point(head.x, head.y, CLOCKWISE[(idx - 1) % 4])

        state = [
            # danger straight / right / left
            self._is_collision(*point_straight),
            self._is_collision(*point_right),
            self._is_collision(*point_left),
            # current direction
            direction == 'left',
            direction == 'right',
            direction == 'up',
            direction == 'down',
            # food location relative to the head
            self.food.FoodRect.x < head.x,
            self.food.FoodRect.x > head.x,
            self.food.FoodRect.y < head.y,
            self.food.FoodRect.y > head.y,
        ]
        return np.array(state, dtype=int)

    def _apply_action(self, action):
        """Turn the relative action [straight, right, left] into an absolute heading."""
        idx = CLOCKWISE.index(self.snake.SnakeDirection)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = CLOCKWISE[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = CLOCKWISE[(idx + 1) % 4]
        else:  # [0, 0, 1]
            new_dir = CLOCKWISE[(idx - 1) % 4]
        self.snake.set_direction(new_dir)

    def step(self, action):
        """
        Apply one move and return (reward, done, score).
        Reward: +10 for eating food, -10 for dying, 0 otherwise.
        """
        self.frame_iteration += 1

        if self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        self._apply_action(action)
        self.snake.move(self.SnakeHeight, self.SnakeWide)
        self.snake.check_alive(self.ScreenWide, self.ScreenHeight, self.SnakeWide)

        self.food.check_eaten(self.snake.SnakeHead)
        food_eaten = self.food.FoodEaten
        self.food.move(self.ScreenWide, self.ScreenHeight, self.SnakeWide, self.SnakeHeight)
        self.snake.grow(food_eaten, self.SnakeWide, self.SnakeHeight)

        # died, or spent far too long without eating (stops pointless looping)
        if (not self.snake.SnakeAlive or
                self.frame_iteration > 100 * (self.snake.SnakeLength + 1)):
            return -10, True, self.snake.SnakeLength

        reward = 10 if food_eaten else 0

        if self.render_enabled:
            self._render()

        return reward, False, self.snake.SnakeLength

    def _render(self):
        self.screen.fill((0, 0, 0))
        self.snake.render(self.screen)
        self.food.render(self.screen)
        for i, line in enumerate(self.hud_lines):
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (5, 5 + i * 24))
        pygame.display.update()
        self.clock.tick(self.speed)
