import numpy as np
import time
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import pygame
from pygamesnake.machine_learning.directkeys import PressKey, ReleaseKey, W, A, S, D


def neural_network_model(input_size):

    network = input_data(shape=[None, 15,20], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 5, activation='softmax')      #size of y!!!
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


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
        self.FoodRect = pygame.Rect(round_food((np.random.random() * 0.95*ScreenWide),SnakeWide),
                                               round_food((np.random.random() * 0.95* ScreenHeight),SnakeHeight),FoodWide,FoodHeight)
        self.FoodColor = (255,255,0) #yellow
        self.FoodEaten = False
        self.FoodStart = time.time() #to measure how long it is taking to find the food

    def render(self,screen):
        pygame.draw.rect(screen, self.FoodColor, self.FoodRect)

    def move(self,ScreenWide,ScreenHeight,SnakeWide,SnakeHeight):
        if self.FoodEaten == True:
            self.FoodRect = pygame.Rect(round_food((np.random.random() * 0.95*ScreenWide),SnakeWide),
                                        round_food((np.random.random() * 0.95* ScreenHeight),SnakeHeight),FoodWide,FoodHeight)
            self.FoodStart = time.time()

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
    grid = np.zeros([int(ScreenHeight/SnakeHeight),int(ScreenWide/SnakeWide)])
    # 1 for each part of the snake
    grid[int((snake.SnakeHead.centery - SnakeHeight/2) / SnakeHeight)][int((snake.SnakeHead.centerx - SnakeWide/2) / SnakeWide)] = 1
    for block in snake.SnakeBody:
        grid[int((block.centery-SnakeHeight/2)/SnakeHeight)][int((block.centerx-SnakeWide/2)/SnakeWide)] = 1
    # -1 for the food
    grid[int((food.FoodRect.centery - FoodHeight / 2) / FoodHeight)][int((food.FoodRect.centerx - FoodWide / 2) / FoodWide)] = -1

    return grid


def execute_action(action):
    """
    Purpose
        Exexute the action predicted by the model
    Input
        action,int: action to be performed
    Output

    """
    if action == 0:
        PressKey(W)
        ReleaseKey(W)
    elif action == 1:
        PressKey(A)
        ReleaseKey(A)
    elif action == 2:
        PressKey(D)
        ReleaseKey(D)
    elif action == 3:
        PressKey(S)
        ReleaseKey(S)
    # action 4 is do nothing


def start_snake(model,model_name,snake,food):
    # initialize the pygame
    pygame.init()
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
                grid = create_grid(snake, food, ScreenWide, SnakeWide, ScreenHeight, SnakeHeight)
                action = np.argmax(model.predict(grid.reshape(-1,15,20)))
                execute_action(action)

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
