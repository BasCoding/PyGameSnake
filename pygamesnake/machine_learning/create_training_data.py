import os
import pygame
import numpy
from tqdm import tqdm
from pygamesnake.game.snake import Snake, Food
from pygamesnake.machine_learning.grid import create_grid

# action vector layout: [up, left, right, down, no-op]
ACTION_LABELS = ['UP', 'LEFT', 'RIGHT', 'DOWN', 'NONE']


def sample_weight(snake):
    """
    Purpose
        Importance weight of a sample; longer (more successful) snakes count
        more heavily during training.
    Input
        snake, Class: Snake
    Output
        weight, float
    """
    return (snake.SnakeLength + 1) ** 0.5


def direction_to_action(snake):
    """
    Purpose
        Encode the snake's current direction as a one-hot action vector
        matching ACTION_LABELS.
    Input
        snake, Class: Snake
    Output
        action, list: one-hot action matching ACTION_LABELS
    """
    actions = {
        'up':    [1, 0, 0, 0, 0],
        'left':  [0, 1, 0, 0, 0],
        'right': [0, 0, 1, 0, 0],
        'down':  [0, 0, 0, 1, 0],
    }
    return actions.get(snake.SnakeDirection, [0, 0, 0, 0, 1])


def draw_manual_overlay(screen, font, sample_count, last_action, last_label, last_weight):
    """
    Purpose
        Draw a live overlay showing the training data being collected.
    Input
        screen, pygame.Surface: the game screen (the grid input the model sees)
        font, pygame.font.Font: font used for the overlay text
        sample_count, int: number of samples collected so far
        last_action, list: most recently recorded one-hot action (output)
        last_label, str: human readable label of the last action (output)
        last_weight, float: importance weight of the last recorded sample
    """
    lines = [
        "MANUAL MODE - move with W/A/S/D, close window to save",
        f"output: {last_label} {last_action} (weight {round(last_weight, 2)})",
        f"samples collected: {sample_count}",
    ]
    for i, line in enumerate(lines):
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (5, 5 + i * 22))


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
        action = [1, 0, 0, 0, 0]
    elif snake.SnakeDirection != "left" and snake.SnakeDirection != "right" and snake.SnakeHead.left >= food.FoodRect.right:
        snake.set_direction('left')
        action = [0, 1, 0, 0, 0]
    elif snake.SnakeDirection != "right" and snake.SnakeDirection != "left" and snake.SnakeHead.right <= food.FoodRect.left:
        snake.set_direction('right')
        action = [0, 0, 1, 0, 0]
    elif snake.SnakeDirection != "down" and snake.SnakeDirection != "up" and snake.SnakeHead.bottom <= food.FoodRect.top:
        snake.set_direction('down')
        action = [0, 0, 0, 1, 0]
    else:
        action = [0, 0, 0, 0, 1]

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
                game_memory.append([grid, action, sample_weight(snake)])
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


def collect_manual_training_data(snake, food, ScreenWide, ScreenHeight, FontType, SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight, FoodWide, FoodHeight):
    """
    Purpose
        Collect training data from a human player instead of the algorithm.
        Every frame the current grid (input) and the direction the player is
        steering (output) are recorded, and shown live on screen and console.
    Input
        snake, Class: Snake
        food, Class: Food
        ScreenWide, ScreenHeight, FontType, SnakeStartX, SnakeStartY,
        SnakeWide, SnakeHeight, FoodWide, FoodHeight: game settings
    Output
        training_data, list: collected [grid, action, weight] samples
    """
    game_memory = []
    training_data = []
    scores = []

    # initialize pygame (skip audio, we never use sound)
    pygame.display.init()
    pygame.font.init()
    info_font = pygame.font.SysFont(FontType, 20)
    screen = pygame.display.set_mode((ScreenWide, ScreenHeight))

    # Title
    pygame.display.set_caption('Snake - manual training data collection')

    last_label = "-"
    last_weight = 0.0
    last_action = [0, 0, 0, 0, 0]

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
                action = direction_to_action(snake)
                weight = sample_weight(snake)
                game_memory.append([grid, action, weight])
                last_label = ACTION_LABELS[int(numpy.argmax(action))]
                last_weight = weight
                last_action = action
                sample_count = len(training_data) + len(game_memory)
                # live console feedback of the sample being added
                print(f"sample {sample_count}: output={last_label} {action} weight={round(weight, 2)}")
            # keep a playable speed so a human can control the snake
            pygame.time.Clock().tick(snake.SnakeSpeed)
        else:
            # a human run is always kept: append it and start a fresh game
            for data in game_memory:
                training_data.append(data)
            scores.append(snake.SnakeLength)
            game_memory = []
            snake = Snake(SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight)
            food = Food(ScreenWide, ScreenHeight, FoodWide, FoodHeight, SnakeWide, SnakeHeight)

        draw_manual_overlay(screen, info_font, len(training_data) + len(game_memory), last_action, last_label, last_weight)
        pygame.display.update()

    # keep whatever was collected in the final (unfinished) run
    for data in game_memory:
        training_data.append(data)

    if scores:
        print("average score of manual play: ", numpy.mean(scores))
    print("total manual samples collected: ", len(training_data))

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
    TrainingPeriod = 10000


    if ScreenWide % SnakeWide != 0 or ScreenHeight % SnakeHeight != 0:
        print('WARNING: ScreenWide and ScreenHeight are not a multiple of the snake size, grid for snake will be off')
    if SnakeStartX % SnakeWide != 0 or SnakeStartY % SnakeHeight != 0:
        print('WARNING: Start position of snake is not aligned with grid')

    snake = Snake(SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight)
    food = Food(ScreenWide, ScreenHeight, FoodWide, FoodHeight, SnakeWide, SnakeHeight)

    mode = input("Select training data mode - [a]lgorithm or [m]anual (default a): ").strip().lower()
    if mode == 'm':
        training_data = collect_manual_training_data(snake, food, ScreenWide, ScreenHeight, FontType, SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight, FoodWide, FoodHeight)
        save_path = os.path.join('data', 'training_data_manual.npy')
    else:
        training_data = start_snake(snake,food,ScreenWide,ScreenHeight,FontType,SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight,FoodWide, FoodHeight,ScoreRequirement,TrainingPeriod)
        save_path = os.path.join('data', 'training_data_algorithm.npy')

    os.makedirs('data', exist_ok=True)

    # append to existing training data instead of overwriting it
    if os.path.exists(save_path):
        existing_data = list(numpy.load(save_path, allow_pickle=True))
        print(f"appending {len(training_data)} new samples to {len(existing_data)} existing samples")
        training_data = existing_data + training_data

    training_data_save = numpy.array(training_data, dtype=object)
    numpy.save(save_path, training_data_save)
    print(f"saved {len(training_data)} total samples to {save_path}")