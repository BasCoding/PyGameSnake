import os
import pygame
import numpy
from pygamesnake.machine_learning.directkeys import PressKey, ReleaseKey, W, A, S, D

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
        PressKey(W)
        ReleaseKey(W)
        action = [(snake.SnakeLength+1)**0.5*1, 0, 0, 0, 0]
    elif snake.SnakeDirection != "left" and snake.SnakeDirection != "right" and snake.SnakeHead.left >= food.FoodRect.right:
        PressKey(A)
        ReleaseKey(A)
        action = [0, (snake.SnakeLength+1)**0.5*1, 0, 0, 0]
    elif snake.SnakeDirection != "right" and snake.SnakeDirection != "left" and snake.SnakeHead.right <= food.FoodRect.left:
        PressKey(D)
        ReleaseKey(D)
        action = [0, 0, (snake.SnakeLength+1)**0.5*1, 0, 0]
    elif snake.SnakeDirection != "down" and snake.SnakeDirection != "up" and snake.SnakeHead.bottom <= food.FoodRect.top:
        PressKey(S)
        ReleaseKey(S)
        action = [0, 0, 0, (snake.SnakeLength+1)**0.5*1,0]
    else:
        action = [0,0,0,0,(snake.SnakeLength+1)**0.5*1]

    return action


def round_food(f,r_v):
    """"
    Purpose:
        round a number to a multiple of the rounding value
    Input:
        f, float: value to be rounded
        r_v, int: rounding value
    Output:
        f, float: the rounded value
    """
    r = f % r_v
    if r / r_v > 0.5:
        f += (r_v - r)
    else:
        f -= r
    return f

#create the snake class
class Snake:
    def __init__(self,SnakeStartX,SnakeStartY,SnakeWide,SnakeHeight):
        self.SnakeSpeed = 5
        self.SnakeHead = pygame.Rect(SnakeStartX,SnakeStartY,SnakeWide,SnakeHeight)
        self.SnakeBody = []
        self.SnakeDirection = 'right'
        self.SnakeAlive = True
        self.SnakeColor = (200,116,200) #yellow
        self.SnakeLength = 0
        self.SnakeTrail = [[self.SnakeHead.copy()]]


    def render(self,screen):
        pygame.draw.rect(screen, self.SnakeColor, self.SnakeHead)
        for block in self.SnakeBody:
            pygame.draw.rect(screen, self.SnakeColor, block)

    def move(self,SnakeHeight,SnakeWide):
        #leave trail for SnakeBody to follow
        self.SnakeTrail[0] = self.SnakeHead.copy()
        for i in range(self.SnakeLength):
            self.SnakeTrail[i+1] = self.SnakeBody[i].copy()
        #move head (body will follow by trail)
        if self.SnakeDirection == 'up':
            self.SnakeHead.move_ip(0, -SnakeHeight)
        elif self.SnakeDirection == 'down':
            self.SnakeHead.move_ip(0, SnakeHeight)
        elif self.SnakeDirection == 'left':
            self.SnakeHead.move_ip(-SnakeWide, 0)
        elif self.SnakeDirection == 'right':
            self.SnakeHead.move_ip(SnakeWide, 0)
        #move body
        for i in range(self.SnakeLength):
            self.SnakeBody[i] = self.SnakeTrail[i]

    def grow(self,FoodEaten,SnakeWide,SnakeHeight):
        if FoodEaten:
            if self.SnakeDirection == 'up':
                self.SnakeBody += [pygame.Rect(self.SnakeHead.x,self.SnakeHead.y + SnakeHeight,SnakeWide,SnakeHeight)]
            elif self.SnakeDirection == 'down':
                self.SnakeBody += [pygame.Rect(self.SnakeHead.x, self.SnakeHead.y - SnakeHeight, SnakeWide, SnakeHeight)]
            elif self.SnakeDirection == 'left':
                self.SnakeBody += [pygame.Rect(self.SnakeHead.x + SnakeWide, self.SnakeHead.y, SnakeWide, SnakeHeight)]
            elif self.SnakeDirection == 'right':
                self.SnakeBody += [pygame.Rect(self.SnakeHead.x - SnakeWide, self.SnakeHead.y, SnakeWide, SnakeHeight)]
            self.SnakeLength += 1
            self.SnakeTrail += [self.SnakeBody[self.SnakeLength-1].copy()]
            self.SnakeSpeed += 1

    def set_direction(self,NewDirection):
            self.SnakeDirection = NewDirection

    def check_alive(self,ScreenWide,ScreenHeight,SnakeWide):
        #side of screen
        if self.SnakeHead.right > ScreenWide or self.SnakeHead.left < 0 or self.SnakeHead.top < 0 or self.SnakeHead.bottom > ScreenHeight:
            self.SnakeAlive = False
        #snake itself
        for block in self.SnakeBody:
            if self.SnakeHead.colliderect(block):
                self.SnakeAlive = False


class Food():
    def __init__(self,ScreenWide,ScreenHeight,FoodWide,FoodHeight,SnakeWide,SnakeHeight):
        self.FoodRect = pygame.Rect(round_food((numpy.random.random() * 0.95*ScreenWide),SnakeWide),
                                               round_food((numpy.random.random() * 0.95* ScreenHeight),SnakeHeight),FoodWide,FoodHeight)
        self.FoodColor = (255,255,0) #yellow
        self.FoodEaten = False

    def render(self,screen):
        pygame.draw.rect(screen, self.FoodColor, self.FoodRect)

    def move(self,ScreenWide,ScreenHeight,SnakeWide,SnakeHeight):
        if self.FoodEaten == True:
            self.FoodRect = pygame.Rect(round_food((numpy.random.random() * 0.95*ScreenWide),SnakeWide),
                                        round_food((numpy.random.random() * 0.95* ScreenHeight),SnakeHeight),FoodWide,FoodHeight)

    def check_eaten(self,SnakeHead):
        if SnakeHead.colliderect(self.FoodRect):
            self.FoodEaten = True
        else:
            self.FoodEaten = False


def create_grid(snake,food,ScreenWide,SnakeWide,ScreenHeight,SnakeHeight):
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
    Output
        grid, array: grid with positions of the food (-1) and snake (1)
    """
    grid = numpy.zeros([int(ScreenHeight/SnakeHeight),int(ScreenWide/SnakeWide)])
    # 1 for the head of the snake
    grid[int((snake.SnakeHead.centery - SnakeHeight/2) / SnakeHeight)][int((snake.SnakeHead.centerx - SnakeWide/2) / SnakeWide)] = 1 #5
    # 1 for each part of the body of the snake
    for block in snake.SnakeBody:
        grid[int((block.centery-SnakeHeight/2)/SnakeHeight)][int((block.centerx-SnakeWide/2)/SnakeWide)] = 1
    # -5 for the food
    grid[int((food.FoodRect.centery - FoodHeight / 2) / FoodHeight)][int((food.FoodRect.centerx - FoodWide / 2) / FoodWide)] = -1 #-5

    return grid


def start_snake(snake,food,ScreenWide,ScreenHeight,SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight,FoodWide, FoodHeight,ScoreRequirement,TrainingPeriod):
    # initiate game memory
    game_memory = []
    run = 0
    training_data = []
    scores = []

    # initialize the pygame
    pygame.init()
    font = pygame.font.SysFont(FontType, 72)
    screen = pygame.display.set_mode((ScreenWide, ScreenHeight))

    # Title
    pygame.display.set_caption('Snake')

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
                grid = create_grid(snake, food, ScreenWide, SnakeWide, ScreenHeight, SnakeHeight)
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

            # reset memory and snake until end of training period
            if run < TrainingPeriod:
                game_memory = []
                snake = Snake(SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight)
                food = Food(ScreenWide, ScreenHeight, FoodWide, FoodHeight, SnakeWide, SnakeHeight)
            else:
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
    TrainingPeriod = 15000


    if ScreenWide % SnakeWide != 0 or ScreenHeight % SnakeHeight != 0:
        print('WARNING: ScreenWide and ScreenHeight are not a multiple of the snake size, grid for snake will be off')
    if SnakeStartX % SnakeWide != 0 or SnakeStartY % SnakeHeight != 0:
        print('WARNING: Start position of snake is not aligned with grid')

    snake = Snake(SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight)
    food = Food(ScreenWide, ScreenHeight, FoodWide, FoodHeight, SnakeWide, SnakeHeight)
    training_data = start_snake(snake,food,ScreenWide,ScreenHeight,SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight,FoodWide, FoodHeight,ScoreRequirement,TrainingPeriod)

    training_data_save = numpy.array(training_data, dtype=object)
    numpy.save(os.path.join('data','training_data.npy'), training_data_save)