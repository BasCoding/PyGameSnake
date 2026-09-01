import os
import argparse
import numpy as np
import pygame
from pygamesnake.game.snake import Snake, Food
from pygamesnake.machine_learning.grid import create_grid
from pygamesnake.machine_learning.model import neural_network_model


def execute_action(snake, action):
    """
    Purpose
        Exexute the action predicted by the model
    Input
        snake, Class: Snake
        action,int: action to be performed
    Output

    """
    if action == 0:
        snake.set_direction('up')
    elif action == 1:
        snake.set_direction('left')
    elif action == 2:
        snake.set_direction('right')
    elif action == 3:
        snake.set_direction('down')
    # action 4 is do nothing


def start_snake(model,model_name,snake,food,normal_speed=False):
    # initialize pygame (skip audio, we never use sound)
    pygame.display.init()
    # font = pygame.font.SysFont("comicsansms", 72)
    screen = pygame.display.set_mode((ScreenWide, ScreenHeight))

    # Title
    pygame.display.set_caption('Snake')

    # initiate scores
    scores = []
    time_kills = 0
    run = 0
    steps_since_food = 0

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

            snake.move(SnakeHeight,SnakeWide)
            snake.render(screen)
            food.render(screen)

            snake.check_alive(ScreenWide, ScreenHeight, SnakeWide)
            food.check_eaten(snake.SnakeHead)
            food.move(ScreenWide,ScreenHeight,SnakeWide,SnakeHeight)
            snake.grow(food.FoodEaten,SnakeWide,SnakeHeight)
            if snake.SnakeAlive:
                grid = create_grid(snake, food, ScreenWide, SnakeWide, ScreenHeight, SnakeHeight, FoodWide, FoodHeight)
                action = np.argmax(model.predict(grid.reshape(-1,15,20)))
                execute_action(snake, action)

            # render at playable speed when requested, otherwise run as fast as possible
            if normal_speed:
                pygame.time.Clock().tick(snake.SnakeSpeed)

            # requirement to get the food in time, counted in game steps so it is
            # independent of the frame rate (fast vs --normal-speed)
            steps_since_food = 0 if food.FoodEaten else steps_since_food + 1
            if steps_since_food > max_steps_without_food:
                snake.SnakeAlive = False
                time_kills += 1
        else:
            scores.append(snake.SnakeLength)
            run += 1
            steps_since_food = 0
            # reset memory and snake until end of training period
            if run < testing_period:
                snake = Snake(SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight)
                food = Food(ScreenWide, ScreenHeight, FoodWide, FoodHeight, SnakeWide, SnakeHeight)
            else:
                running = False
                print(model_name+': '+'mean: '+str(np.mean(scores))+', min: '+str(np.min(scores))+', max: ' +
                      str(np.max(scores))+', time fails: '+str(time_kills))

        pygame.display.update()


def test_model(model_name,input_size,normal_speed=False):
    model = neural_network_model(input_size)
    model.load(model_name)
    snake = Snake(SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight)
    food = Food(ScreenWide, ScreenHeight, FoodWide, FoodHeight, SnakeWide, SnakeHeight)
    start_snake(model,model_name,snake,food,normal_speed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test a trained snake model')
    parser.add_argument('model_version', help='Name of the model version to test (e.g. trained_model_head_5)')
    parser.add_argument('--normal-speed', action='store_true', help='Watch the game at normal playable speed instead of running as fast as possible')
    args = parser.parse_args()

    # Settings
    ScreenWide = 800
    ScreenHeight = 600
    SnakeStartX = 0
    SnakeStartY = 0
    SnakeWide = 40
    SnakeHeight = 40
    FoodWide = 40
    FoodHeight = 40
    testing_period = 100
    max_steps_without_food = 60  # game steps allowed to reach the next food
    LR = 1e-3

    if ScreenWide % SnakeWide != 0 or ScreenHeight % SnakeHeight != 0:
        print('WARNING: ScreenWide and ScreenHeight are not a multiple of the snake size, grid for snake will be off')
    if SnakeStartX % SnakeWide != 0 or SnakeStartY % SnakeHeight != 0:
        print('WARNING: Start position of snake is not aligned with grid')

    training_data = np.load(os.path.join('data', 'training_data_algorithm.npy'), allow_pickle=True)
    X = np.array([i[0] for i in training_data])
    input_size = X[0].size

    test_model(args.model_version, input_size, args.normal_speed)
