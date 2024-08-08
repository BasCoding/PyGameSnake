import pygame
import numpy

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


def start_snake(ScreenWide,ScreenHeight,FontType,SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight,FoodWide, FoodHeight):
    # initialize the pygame
    pygame.init()
    font = pygame.font.SysFont(FontType, 72)
    screen = pygame.display.set_mode((ScreenWide, ScreenHeight))

    # Title
    pygame.display.set_caption('Snake')

    # Game Loop
    running = True
    startup = True
    while running:
        # start = time.time()
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

            if event.type == pygame.MOUSEBUTTONDOWN:
                if startup:
                    if StartRect.collidepoint(event.pos):
                        snake = Snake(SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight)
                        food = Food(ScreenWide, ScreenHeight, FoodWide, FoodHeight, SnakeWide, SnakeHeight)
                        startup = False
                else:
                    if retryRect.collidepoint(event.pos):
                        snake = Snake(SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight)
                        food = Food(ScreenWide, ScreenHeight, FoodWide, FoodHeight, SnakeWide, SnakeHeight)

            # if event.type == pygame.KEYUP:
            #     speed = 0.0
        if startup:
            StartText = font.render("Start", True, (255, 0, 0))
            StartRect = pygame.Rect(ScreenWide / 2 - StartText.get_width() / 2,
                                    ScreenHeight / 2 - StartText.get_height() / 2 - ScreenHeight * 0.2,
                                    StartText.get_width() * 1.1, StartText.get_height() * 1.1)
            if StartRect.collidepoint(pygame.mouse.get_pos()):
                StartText = font.render("Start", True, (255, 255, 255))

            pygame.draw.rect(screen, (0, 255, 0), StartRect, 4)
            screen.blit(StartText, (StartRect.x * 1.05, StartRect.y))
        else:
            if snake.SnakeAlive:

                snake.move(SnakeHeight, SnakeWide)
                snake.render(screen)
                food.render(screen)

                snake.check_alive(ScreenWide, ScreenHeight, SnakeWide)
                food.check_eaten(snake.SnakeHead)
                food.move(ScreenWide, ScreenHeight, SnakeWide, SnakeHeight)
                snake.grow(food.FoodEaten, SnakeWide, SnakeHeight)
                pygame.time.Clock().tick(snake.SnakeSpeed)
            else:

                LoserText = font.render("You lost", True, (255, 0, 0))
                ScoreText = font.render("Score: " + str(snake.SnakeLength), True, (255, 0, 0))
                RetryText = font.render("Try again", True, (0, 255, 0))
                retryRect = pygame.Rect(ScreenWide / 2 - RetryText.get_width() / 2,
                                        ScreenHeight / 2 - RetryText.get_height() / 2 - ScreenHeight * 0.2,
                                        RetryText.get_width() * 1.1, RetryText.get_height() * 1.1)
                if retryRect.collidepoint(pygame.mouse.get_pos()):
                    RetryText = font.render("Try again", True, (255, 255, 255))
                screen.blit(LoserText,
                            (ScreenWide / 2 - LoserText.get_width() / 2, ScreenHeight / 2 - LoserText.get_height() / 2))
                screen.blit(ScoreText,
                            (ScreenWide / 2 - ScoreText.get_width() / 2, ScreenHeight / 1.5 - ScoreText.get_height() / 2))
                pygame.draw.rect(screen, (0, 255, 0), retryRect, 4)
                screen.blit(RetryText, (retryRect.x * 1.05, retryRect.y))

        pygame.display.update()


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


    if ScreenWide % SnakeWide != 0 or ScreenHeight % SnakeHeight != 0:
        print('WARNING: ScreenWide and ScreenHeight are not a multiple of the snake size, grid for snake will be off')
    if SnakeStartX % SnakeWide != 0 or SnakeStartY % SnakeHeight != 0:
        print('WARNING: Start position of snake is not aligned with grid')

    start_snake(ScreenWide,ScreenHeight,FontType,SnakeStartX, SnakeStartY, SnakeWide, SnakeHeight,FoodWide, FoodHeight)