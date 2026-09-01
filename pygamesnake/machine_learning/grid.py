import numpy


def create_grid(snake, food, ScreenWide, SnakeWide, ScreenHeight, SnakeHeight, FoodWide, FoodHeight):
    """
    Purpose
        Create a grid from the positions of the food and snake in the game
    Input
        snake, Class: Snake
        food, Class: Food
        ScreenWide, int: wide of the screen
        SnakeWide, int: wide of the snake
        ScreenHeight, int: height of the screen
        SnakeHeight, int: height of the snake
        FoodWide, int: wide of the food
        FoodHeight, int: height of the food
    Output
        grid, array: grid with positions of the food (-1) and snake (1)
    """
    grid = numpy.zeros([int(ScreenHeight / SnakeHeight), int(ScreenWide / SnakeWide)])
    # 1 for the head of the snake
    grid[int((snake.SnakeHead.centery - SnakeHeight / 2) / SnakeHeight)][int((snake.SnakeHead.centerx - SnakeWide / 2) / SnakeWide)] = 1
    # 1 for each part of the body of the snake
    for block in snake.SnakeBody:
        grid[int((block.centery - SnakeHeight / 2) / SnakeHeight)][int((block.centerx - SnakeWide / 2) / SnakeWide)] = 1
    # -1 for the food
    grid[int((food.FoodRect.centery - FoodHeight / 2) / FoodHeight)][int((food.FoodRect.centerx - FoodWide / 2) / FoodWide)] = -1

    return grid
