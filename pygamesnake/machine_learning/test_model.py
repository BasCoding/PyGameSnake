import numpy as np
import time
import tensorflow as tf
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


def start_snake(model,model_name,snake,food):
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

            # disable tick for speedy testing
            #pygame.time.Clock().tick(snake.SnakeSpeed)

            # requirement to get the food in time
            if time.time() - food.FoodStart > time_req:
                snake.SnakeAlive = False
                time_kills += 1
        else:
            scores.append(snake.SnakeLength)
            run += 1
            # reset memory and snake until end of training period
            if run < testing_period:
                snake = Snake(SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight)
                food = Food(ScreenWide, ScreenHeight, FoodWide, FoodHeight, SnakeWide, SnakeHeight)
            else:
                running = False
                print(model_name+': '+'mean: '+str(np.mean(scores))+', min: '+str(np.min(scores))+', max: ' +
                      str(np.max(scores))+', time fails: '+str(time_kills))

        pygame.display.update()


def test_model(model_name,input_size):
    tf.compat.v1.reset_default_graph()
    model = neural_network_model(input_size)
    model.load(model_name)
    snake = Snake(SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight)
    food = Food(ScreenWide, ScreenHeight, FoodWide, FoodHeight, SnakeWide, SnakeHeight)
    start_snake(model,model_name,snake,food)


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
    testing_period = 100
    time_req = 2.5  # in seconds
    LR = 1e-3

    if ScreenWide % SnakeWide != 0 or ScreenHeight % SnakeHeight != 0:
        print('WARNING: ScreenWide and ScreenHeight are not a multiple of the snake size, grid for snake will be off')
    if SnakeStartX % SnakeWide != 0 or SnakeStartY % SnakeHeight != 0:
        print('WARNING: Start position of snake is not aligned with grid')

    training_data = np.load('training_data.npy', allow_pickle=True)
    X = np.array([i[0] for i in training_data])
    input_size = len(X[0])

    model_v = ['trained_model_normal_7_125','trained_model_normal','trained_model_normal_3','trained_model_normal_5_250',
               'trained_model_normal_7','trained_model_weighted','trained_model_weighted_3','trained_model_weighted_5_250',
               'trained_model_weighted_7','trained_model_head_5_500','trained_model_head_7_250']
    for model_name in model_v:
        test_model(model_name,input_size)
