import os
import pygame
import numpy
from tqdm import tqdm
from pygamesnake.game.snake import Snake, Food
from pygamesnake.machine_learning.grid import create_grid

def simple_algorithm(snake,food):
    """
    Purpose
        Simple algorithm to grow the snake to gather training data
    Input
        snake, Class: Snake
        food, Class: Food
    Output
        action,
    """
    if snake.SnakeDirection != "up" and snake.SnakeDirection != "down" and snake.SnakeHead.top >= food.FoodRect.bottom:
        snake.set_direction('up')
        action = [(snake.SnakeLength+1)**0.5*1, 0, 0, 0, 0]
    elif snake.SnakeDirection != "left" and snake.SnakeDirection != "right" and snake.SnakeHead.left >= food.FoodRect.right:
        snake.set_direction('left')
        action = [0, (snake.SnakeLength+1)**0.5*1, 0, 0, 0]
    elif snake.SnakeDirection != "right" and snake.SnakeDirection != "left" and snake.SnakeHead.right <= food.FoodRect.left:
        snake.set_direction('right')
        action = [0, 0, (snake.SnakeLength+1)**0.5*1, 0, 0]
    elif snake.SnakeDirection != "down" and snake.SnakeDirection != "up" and snake.SnakeHead.bottom <= food.FoodRect.top:
        snake.set_direction('down')
        action = [0, 0, 0, (snake.SnakeLength+1)**0.5*1,0]
    else:
        action = [0,0,0,0,(snake.SnakeLength+1)**0.5*1]

    return action


def start_snake(snake,food,ScreenWide,ScreenHeight,FontType,SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight,FoodWide, FoodHeight,ScoreRequirement,TrainingPeriod):
    # initiate game memory
    game_memory = []
    run = 0
    training_data = []
    scores = []

    # initialize pygame (skip audio, we never use sound)
    pygame.display.init()
    pygame.font.init()
    font = pygame.font.SysFont(FontType, 72)
    screen = pygame.display.set_mode((ScreenWide, ScreenHeight))

    # Title
    pygame.display.set_caption('Snake')

    # progress bar over the qualifying runs collected
    progress = tqdm(total=TrainingPeriod, desc='Collecting training data', unit='run')

    # Game Loop
    running = True
    while running:
        # pixels in RGB
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    snake.set_direction('left')
                elif event.key == pygame.K_d:
                    snake.set_direction('right')
                elif event.key == pygame.K_w:
                    snake.set_direction('up')
                elif event.key == pygame.K_s:
                    snake.set_direction('down')

        if snake.SnakeAlive:

            snake.move(SnakeHeight, SnakeWide)
            snake.render(screen)
            food.render(screen)

            snake.check_alive(ScreenWide, ScreenHeight, SnakeWide)
            food.check_eaten(snake.SnakeHead)
            food.move(ScreenWide, ScreenHeight, SnakeWide, SnakeHeight)
            snake.grow(food.FoodEaten, SnakeWide, SnakeHeight)
            if snake.SnakeAlive:
                grid = create_grid(snake, food, ScreenWide, SnakeWide, ScreenHeight, SnakeHeight, FoodWide, FoodHeight)
                action = simple_algorithm(snake, food)
                game_memory.append([grid, action])
            # disable tick for speedy testing
            #pygame.time.Clock().tick(snake.SnakeSpeed)
        else:
            # save the game_memory to the training data if the game was good enough
            if snake.SnakeLength >= ScoreRequirement:
                run += 1
                for data in game_memory:
                    training_data.append(data)
                scores.append(snake.SnakeLength)
                progress.update(1)

            # reset memory and snake until end of training period
            if run < TrainingPeriod:
                game_memory = []
                snake = Snake(SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight)
                food = Food(ScreenWide, ScreenHeight, FoodWide, FoodHeight, SnakeWide, SnakeHeight)
            else:
                progress.close()
                print("average score of simple algorithm: ", numpy.mean(scores))
                running = False

        pygame.display.update()

    return training_data

if __name__ == "__main__":

    # Settings
    ScreenWide = 800
    ScreenHeight = 600
    SnakeStartX = 0
    SnakeStartY = 0
    SnakeWide = 40
    SnakeHeight = 40
    FoodWide = 40
    FoodHeight = 40
    FontType = "comicsansms"
    ScoreRequirement = 10
    TrainingPeriod = 5000


    if ScreenWide % SnakeWide != 0 or ScreenHeight % SnakeHeight != 0:
        print('WARNING: ScreenWide and ScreenHeight are not a multiple of the snake size, grid for snake will be off')
    if SnakeStartX % SnakeWide != 0 or SnakeStartY % SnakeHeight != 0:
        print('WARNING: Start position of snake is not aligned with grid')

    snake = Snake(SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight)
    food = Food(ScreenWide, ScreenHeight, FoodWide, FoodHeight, SnakeWide, SnakeHeight)
    training_data = start_snake(snake,food,ScreenWide,ScreenHeight,FontType,SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight,FoodWide, FoodHeight,ScoreRequirement,TrainingPeriod)

    training_data_save = numpy.array(training_data, dtype=object)
    os.makedirs('data', exist_ok=True)
    numpy.save(os.path.join('data','training_data.npy'), training_data_save)