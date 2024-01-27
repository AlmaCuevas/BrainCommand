# Build Pac-Man from Scratch in Python with PyGame!!
import copy
from datetime import datetime

from board_execution import execution_boards, start_execution_positions
import pygame
import math
import time
import pylsl
import os

ASSETS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))

dev_mode = True
# TODO: In theory is done, check this is true: The data should be saved only if game_over, not if I just quit the game
if not dev_mode:
    # LSL COMMUNICATION
    def lsl_mrk_outlet(name):
       info = pylsl.stream_info(name, 'Markers', 1, 0, pylsl.cf_string, 'ID66666666')
       outlet = pylsl.stream_outlet(info, 1, 1)
       print('pacman created result outlet.')
       return outlet
    mrkstream_allowed_turn_out = lsl_mrk_outlet('Allowed_Turn_Markers') # important this is first

# GAME
pygame.init()
current_level = 0  # Inicialmente, el nivel 0 está en juego

# Dimensions
display_info = pygame.display.Info() # Get the monitor's display info
WIDTH = int(display_info.current_h)
HEIGHT = int(display_info.current_h)

level = copy.deepcopy(execution_boards[current_level])
div_width = len(level[0])
div_height = len(level)
yscale = HEIGHT // div_height
xscale = WIDTH // div_width

# 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN

screen = pygame.display.set_mode([WIDTH, HEIGHT])
timer = pygame.time.Clock()
fps = 60  # This decides how fast the game goes. Including pacman and ghosts.
font = pygame.font.Font("freesansbold.ttf", 30)
color = "white"
PI = math.pi


## Images import
player_images = [pygame.transform.scale(pygame.image.load(f'assets/extras_images/right.png'), (xscale, yscale)),
                 pygame.transform.scale(pygame.image.load(f'assets/extras_images/left.png'), (xscale, yscale)),
                 pygame.transform.scale(pygame.image.load(f'assets/extras_images/forward.png'), (xscale, yscale)),
                 pygame.transform.scale(pygame.image.load(f'assets/extras_images/back.png'), (xscale, yscale))] # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
arrow = pygame.transform.scale(
    pygame.image.load(f"assets/extras_images/arrow.png"), (xscale, yscale)
)
arrow_images = [
    pygame.transform.rotate(arrow, -90),
    pygame.transform.rotate(arrow, 90),
    arrow,
    pygame.transform.rotate(arrow, 180),
]  # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
cookie = pygame.transform.scale(pygame.image.load(f'assets/extras_images/cookie.png'), (xscale, yscale))

sound_thud = pygame.mixer.Sound(os.path.join(ASSETS_PATH, 'sounds', 'thud.mp3'))
sound_go = pygame.mixer.Sound(os.path.join(ASSETS_PATH, 'sounds', 'go.mp3'))
sound_win = pygame.mixer.Sound(os.path.join(ASSETS_PATH, 'sounds', 'finish a level.mp3'))

## Positions
start = start_execution_positions[current_level]
player_x = int(start[0] * xscale)
player_y = int(start[1] * yscale)
direction = start[2]
last_direction = start[2]

# Other
turns_allowed = [False, False, False, False]  # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
direction_command = start[2]
if dev_mode:
    player_speed = 5
    original_speed = 5
else:
    player_speed = 1
    original_speed = 1
moving = False
startup_counter = 0
counter = 0
flicker = False
game_over = False
game_won = False
last_activate_turn_tile = [4, 4] # Check that in all levels this is a 0 pixel


def draw_misc():
    gameover_text = font.render("¡Nivel Completado!", True, "red")
    if game_over:
        sound_win.play()
        pygame.draw.rect(screen, "gray", [WIDTH*.05, HEIGHT*.1, WIDTH*.9, HEIGHT*.8], 0, 10)
        pygame.draw.rect(screen, "green", [WIDTH*.1, HEIGHT*.2, WIDTH*.8, HEIGHT*.6], 0, 10)
        gameover_text2 = font.render("¡Gracias por participar!", True, "red")
        gameover_text3 = font.render("Ya puedes cerrar la ventana.", True, "red")
        screen.blit(gameover_text, (xscale*13, HEIGHT//3))
        screen.blit(gameover_text2, (xscale*12, HEIGHT//2))
        screen.blit(gameover_text3, (xscale * 11, xscale * 20))
    elif game_won:
        pygame.draw.rect(screen, "gray", [WIDTH*.05, HEIGHT*.1, WIDTH*.9, HEIGHT*.8], 0, 10)
        pygame.draw.rect(screen, "green", [WIDTH*.1, HEIGHT*.2, WIDTH*.8, HEIGHT*.6], 0, 10)
        gameover_text2 = font.render("¡Prepárate para el siguiente nivel!", True, "red")
        screen.blit(gameover_text, (xscale*13, HEIGHT//3))
        screen.blit(gameover_text2, (xscale*9, HEIGHT//2))

def check_collisions(last_activate_turn_tile, player_speed):
    if 0 < player_x < WIDTH-xscale*2:
        corner_check=copy.deepcopy(turns_allowed)
        corner_check[direction] = False
        if 1 <= level[center_y // yscale][center_x // xscale] <= 2:
            level[center_y // yscale][center_x // xscale] = 0
        if sum(corner_check)>=2 or corner_check==turns_allowed:
            if level[last_activate_turn_tile[0]][last_activate_turn_tile[1]] != -1:
                print('thud')
                sound_thud.play()
                level[center_y // yscale][center_x // xscale] = -1
                last_activate_turn_tile = [center_y // yscale, center_x // xscale]
                player_speed = 0
        elif level[last_activate_turn_tile[0]][last_activate_turn_tile[1]] == -1:
            print('play')
            sound_go.play()
            level[last_activate_turn_tile[0]][last_activate_turn_tile[1]] = 0
            player_speed = original_speed
    return last_activate_turn_tile, player_speed



def draw_board(color):
    for i in range(len(level)):
        for j in range(len(level[i])):
            if level[i][j] == 1:
                pygame.draw.circle(
                    screen,
                    "white",
                    (j * xscale + (0.5 * xscale), i * yscale + (0.5 * yscale)),
                    4,
                )
            if level[i][j] == 2: # and not flicker: # The flicker could affect the brain frequency
                screen.blit(cookie, (j * xscale, i * yscale))
            if level[i][j] == 3:
                pygame.draw.line(
                    screen,
                    color,
                    (j * xscale + (0.5 * xscale), i * yscale),
                    (j * xscale + (0.5 * xscale), i * yscale + yscale),
                    3,
                )
            if level[i][j] == 4:
                pygame.draw.line(
                    screen,
                    color,
                    (j * xscale, i * yscale + (0.5 * yscale)),
                    (j * xscale + xscale, i * yscale + (0.5 * yscale)),
                    3,
                )
            if level[i][j] == 5:
                pygame.draw.arc(
                    screen,
                    color,
                    [
                        (j * xscale - (xscale * 0.4)),
                        (i * yscale + (0.5 * yscale)),
                        xscale,
                        yscale,
                    ],
                    0,
                    PI / 2,
                    3,
                )
            if level[i][j] == 6:
                pygame.draw.arc(
                    screen,
                    color,
                    [(j * xscale + (xscale * 0.5)), (i * yscale + (0.5 * yscale)), xscale, yscale],
                    PI / 2,
                    PI,
                    3,
                )
            if level[i][j] == 7:
                pygame.draw.arc(
                    screen,
                    color,
                    [(j * xscale + (xscale * 0.5)), (i * yscale - (0.4 * yscale)), xscale, yscale],
                    PI,
                    3 * PI / 2,
                    3,
                )
            if level[i][j] == 8:
                pygame.draw.arc(
                    screen,
                    color,
                    [
                        (j * xscale - (xscale * 0.4)),
                        (i * yscale - (0.4 * yscale)),
                        xscale,
                        yscale,
                    ],
                    3 * PI / 2,
                    2 * PI,
                    3,
                )
            if level[i][j] == 9:
                pygame.draw.line(
                    screen,
                    "white",
                    (j * xscale, i * yscale + (0.5 * yscale)),
                    (j * xscale + xscale, i * yscale + (0.5 * yscale)),
                    3,
                )
            if level[i][j] == -1:
                pygame.draw.rect(
                    screen,
                    "yellow",
                    [center_x - xscale, center_y - yscale, 60, 60],
                    border_radius=10,
                )

def draw_player(last_direction):
    # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
    for direction_idx in range(0,4):
        if direction_idx == direction:
            last_direction=direction
            screen.blit(player_images[direction], (player_x, player_y))
    if direction == 4:
        screen.blit(player_images[last_direction], (player_x, player_y))
    return last_direction

def check_position(centerx, centery):
    turns = [False, False, False, False] # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
    half_scale = xscale // 2
    # check collisions based on center x and center y of player +/- fudge number
    if centerx // div_width < div_width-1:
        if direction == 0:
            if level[centery // yscale][(centerx - half_scale) // xscale] < 3:
                turns[1] = True
        elif direction == 1:
            if level[centery // yscale][(centerx + half_scale) // xscale] < 3:
                turns[0] = True
        elif direction == 2:
            if level[(centery + half_scale) // yscale][centerx // xscale] < 3:
                turns[3] = True
        elif direction == 3:
            if level[(centery - half_scale) // yscale][centerx // xscale] < 3:
                turns[2] = True

        if direction == 2 or direction == 3:
            if xscale//3 <= centerx % xscale <= xscale:
                if level[(centery + half_scale) // yscale][centerx // xscale] < 3:
                    turns[3] = True
                if level[(centery - half_scale - 10) // yscale][centerx // xscale] < 3:
                    turns[2] = True
            if yscale//3 <= centery % yscale <= yscale:
                if level[centery // yscale][(centerx - xscale) // xscale] < 3:
                    turns[1] = True
                if level[centery // yscale][(centerx + xscale) // xscale] < 3:
                    turns[0] = True
        if direction == 0 or direction == 1:
            if xscale//3 <= centerx % xscale <= xscale:
                if level[(centery + yscale) // yscale][centerx // xscale] < 3:
                    turns[3] = True
                if level[(centery - yscale) // yscale][centerx // xscale] < 3:
                    turns[2] = True
            if yscale//3 <= centery % yscale <= yscale:
                if level[centery // yscale][(centerx - half_scale - 10) // xscale] < 3:
                    turns[1] = True
                if level[centery // yscale][(centerx + half_scale) // xscale] < 3:
                    turns[0] = True
    else:
        turns[0] = True
        turns[1] = True
    return turns


def move_player(play_x, play_y):
    # r, l, u, d
    # If current direction is right and right is allowed, move right
    if direction == 0 and turns_allowed[0]: 
        play_x += player_speed
    elif direction == 1 and turns_allowed[1]:
        play_x -= player_speed
    if direction == 2 and turns_allowed[2]:
        play_y -= player_speed
    elif direction == 3 and turns_allowed[3]:
        play_y += player_speed
    return play_x, play_y

run = True
first_movement = True
start_time = time.time()
while run:
    timer.tick(fps)
    if counter < 19:
        counter += 1
        if counter > 3:
            flicker = False
    else:
        counter = 0
        flicker = True
    if startup_counter < 180 and not game_over and not game_won:
        moving = False
        startup_counter += 1
    elif startup_counter < fps*20 and not game_won and not dev_mode:
        moving = False
        startup_counter += 1
    else:
        moving = True


    screen.fill("black")
    draw_board(color)
    center_x = player_x + xscale // 2
    center_y = player_y + yscale // 2

    game_won = False
    flat_level_list = [
        x
        for xs in level
        for x in xs
    ]
    if 2 not in flat_level_list:
        if len(start_execution_positions) == current_level+1:
            game_over = True
        game_won = True

    mods = [xscale // 2, xscale // -2]

    if dev_mode:
        # Collider viewer
        player_circle = pygame.draw.circle(screen, 'pink', (center_x+mods[0], center_y+mods[0]), 20, 10)
        player_circle_2 = pygame.draw.circle(screen, 'pink', (center_x + mods[0], center_y+mods[1]), 20, 10)
        player_circle_3 = pygame.draw.circle(screen, 'pink', (center_x + mods[1], center_y + mods[0]), 20, 10)
        player_circle_4 = pygame.draw.circle(screen, 'pink', (center_x + mods[1], center_y + mods[1]), 20, 10)
    last_direction = draw_player(last_direction)
    draw_misc()

    turns_allowed = check_position(center_x, center_y)
    # mrkstream_allowed_turn_out.push_sample(pylsl.vectorstr([str(turns_allowed)]))
    if moving:
        player_x, player_y = move_player(player_x, player_y)

    last_activate_turn_tile, player_speed = check_collisions(last_activate_turn_tile, player_speed)


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYDOWN and player_speed == 0:
            if event.key == pygame.K_RIGHT and turns_allowed[0]:
                direction_command = 0
                player_speed = original_speed
            elif event.key == pygame.K_LEFT and turns_allowed[1]:
                direction_command = 1
                player_speed = original_speed
            elif event.key == pygame.K_UP and turns_allowed[2]:
                direction_command = 2
                player_speed = original_speed
            elif event.key == pygame.K_DOWN and turns_allowed[3]:
                direction_command = 3
                player_speed = original_speed
            if event.key == pygame.K_SPACE and game_over:
                end_time = time.time()
                total_game_time = '{:.2f}'.format(end_time - start_time)

                filename = datetime.now().strftime('game_variables_%H%M_%m%d%Y.txt')

                file = open(os.path.join(ASSETS_PATH, 'game_saved_files', filename), 'w')
                file.write(f'Time: {total_game_time} s')
                file.close()
                run = False
            elif event.key == pygame.K_SPACE and game_won:
                first_movement = True
                draw_misc()
                pygame.display.flip()
                startup_counter = 0
                current_level += 1
                if current_level < len(execution_boards):
                    level = copy.deepcopy(execution_boards[current_level])
                    start = start_execution_positions[current_level]
                    player_x = int(start[0] * xscale)
                    player_y = int(start[1] * yscale)
                    direction = start[2]
                    direction_command = start[2]
                game_won = False

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT and direction_command == 0:
                direction_command = direction
            elif event.key == pygame.K_LEFT and direction_command == 1:
                direction_command = direction
            elif event.key == pygame.K_UP and direction_command == 2:
                direction_command = direction
            elif event.key == pygame.K_DOWN and direction_command == 3:
                direction_command = direction

    for direction_index in range(0, 4):
        if direction_command == direction_index and turns_allowed[direction_index]:
            direction = direction_index
    pygame.display.flip()
pygame.quit()