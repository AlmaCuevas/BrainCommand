# Build Pac-Man from Scratch in Python with PyGame!!
import copy
from board_calibration import prompts_paradigm_SI,boards_paradigm_SI, start_positions_paradigm_SI, commands_list_board
import pygame
import math
import time
import pylsl


def lsl_mrk_outlet(name):
    info = pylsl.stream_info(name, 'Markers', 1, 0, pylsl.cf_string, 'ID66666666');
    outlet = pylsl.stream_outlet(info, 1, 1)
    print('Calibration created result outlet.')
    return outlet

def bci_calibration(dev_mode: bool = False):
    if not dev_mode:
        # LSL COMMUNICATION
        mrkstream_allowed_turn_out = lsl_mrk_outlet('PyGame - Paradigma Experimental') # important this is first

    # GAME
    pygame.init()
    current_level:int = 0  # Inicialmente, el nivel 0 está en juego

    # Dimensions
    display_info = pygame.display.Info() # Get the monitor's display info
    WIDTH:int = int(display_info.current_h)
    HEIGHT:int = int(display_info.current_h)

    level:int = copy.deepcopy(boards_paradigm_SI[current_level])
    prompts = copy.deepcopy(prompts_paradigm_SI[current_level])
    div_width:int = len(level[0])  # 31
    div_height:int = len(level)  # 38
    num1:int = HEIGHT // div_height #23
    num2:int = WIDTH // div_width #29


    commands_list = commands_list_board.pop(0)
     # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN, 4-STOP

    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    timer = pygame.time.Clock()
    fps:int = 60  # This decides how fast the game goes.
    font = pygame.font.Font("RetroFont.ttf", 30)
    color = "white"
    PI = math.pi


    ## Images import
    image_xscale = num2
    image_yscale = num1
    player_images = []
    player_images = [pygame.transform.scale(pygame.image.load(f'assets/extras_images/right_1.png'), (image_xscale, image_yscale)),
                     pygame.transform.scale(pygame.image.load(f'assets/extras_images/left_1.png'), (image_xscale, image_yscale)),
                     pygame.transform.scale(pygame.image.load(f'assets/extras_images/forward_1.png'), (image_xscale, image_yscale)),
                     pygame.transform.scale(pygame.image.load(f'assets/extras_images/back_1.png'), (image_xscale, image_yscale)),
                     pygame.transform.scale(pygame.image.load(f'assets/extras_images/back_1.png'), (image_xscale*10, image_yscale*10))] # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
    arrow = pygame.transform.scale(
        pygame.image.load(f"assets/extras_images/arrow.png"), (image_xscale, image_yscale)
    )
    arrow_images = [
        pygame.transform.rotate(arrow, -90),
        pygame.transform.rotate(arrow, 90),
        arrow,
        pygame.transform.rotate(arrow, 180),
    ]  # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
    cookie = pygame.transform.scale(pygame.image.load(f'assets/extras_images/cookie.png'), (image_xscale, image_yscale))

    ## Positions
    start = start_positions_paradigm_SI.pop(0)
    player_x = int(start[0] * num2)
    player_y = int(start[1]* num1)
    direction = start[2]
    last_direction = start[2]

    #Other
    turns_allowed = [False, False, False, False]  # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
    direction_command = start[2]
    player_speed = 1
    moving = False
    startup_counter = 0
    game_won = False
    last_activate_turn_tile = [1, 1]
    button_pressed = False

    def draw_misc():
        if game_won:
            pygame.draw.rect(screen, "gray", [WIDTH*.05, HEIGHT*.1, WIDTH*.9, HEIGHT*.8], 0, 10)
            pygame.draw.rect(screen, "green", [WIDTH*.1, HEIGHT*.2, WIDTH*.8, HEIGHT*.6], 0, 10)
            level_done = font.render("¡Nivel Completado!", True, "red")
            prepare_for_next_level = font.render("¡Prepárate para el siguiente nivel!", True, "red")
            screen.blit(level_done,
                        (WIDTH / 2 - level_done.get_width() / 2, HEIGHT / 2 - level_done.get_height() / 2))
            screen.blit(prepare_for_next_level,
                        (WIDTH / 2 - prepare_for_next_level.get_width() / 2,
                         HEIGHT / 2 - prepare_for_next_level.get_height() / 2 + 100))

    def command_leader(current_command, player_y, player_x):
        goal_x=player_x
        goal_y=player_y
        if current_command == 'right':  # Right
            goal_x = player_x + num2 * 3
        elif current_command == 'left':  # Left
            goal_x = player_x - num2 * 3
        elif current_command == 'up':  # Up
            goal_y = player_y - num1 * 3
        elif current_command == 'down':  # Down
            goal_y = player_y + num1 * 3
        return goal_x, goal_y

    def check_collisions (last_activate_turn_tile):
        level[last_activate_turn_tile[0]][last_activate_turn_tile[1]] = 0
        if level[center_y // num1][center_x // num2] == 1:
            level[center_y // num1][center_x // num2] = 0
        if level[center_y // num1][center_x // num2] == 2:
            level[center_y // num1][center_x // num2] = 0
        return last_activate_turn_tile

    def draw_board(color):
        for i in range(len(level)):
            for j in range(len(level[i])):
                if level[i][j] == 1:
                    pygame.draw.circle(
                        screen,
                        "white",
                        (j * num2 + (0.5 * num2), i * num1 + (0.5 * num1)),
                        4,
                    )
                if level[i][j] == 2:
                    screen.blit(cookie, (j * num2, i * num1))
                if level[i][j] == 3:
                    pygame.draw.line(
                        screen,
                        color,
                        (j * num2 + (0.5 * num2), i * num1),
                        (j * num2 + (0.5 * num2), i * num1 + num1),
                        3,
                    )
                if level[i][j] == 4:
                    pygame.draw.line(
                        screen,
                        color,
                        (j * num2, i * num1 + (0.5 * num1)),
                        (j * num2 + num2, i * num1 + (0.5 * num1)),
                        3,
                    )
                if level[i][j] == 5:
                    pygame.draw.arc(
                        screen,
                        color,
                        [
                            (j * num2 - (num2 * 0.4)) - 2,
                            (i * num1 + (0.5 * num1)),
                            num2,
                            num1,
                        ],
                        0,
                        PI / 2,
                        3,
                    )
                if level[i][j] == 6:
                    pygame.draw.arc(
                        screen,
                        color,
                        [(j * num2 + (num2 * 0.5)), (i * num1 + (0.5 * num1)), num2, num1],
                        PI / 2,
                        PI,
                        3,
                    )
                if level[i][j] == 7:
                    pygame.draw.arc(
                        screen,
                        color,
                        [(j * num2 + (num2 * 0.5)), (i * num1 - (0.4 * num1)), num2, num1],
                        PI,
                        3 * PI / 2,
                        3,
                    )
                if level[i][j] == 8:
                    pygame.draw.arc(
                        screen,
                        color,
                        [
                            (j * num2 - (num2 * 0.4)) - 2,
                            (i * num1 - (0.4 * num1)),
                            num2,
                            num1,
                        ],
                        3 * PI / 2,
                        2 * PI,
                        3,
                    )
                if level[i][j] == 9:
                    pygame.draw.line(
                        screen,
                        "white",
                        (j * num2, i * num1 + (0.5 * num1)),
                        (j * num2 + num2, i * num1 + (0.5 * num1)),
                        3,
                    )
                if level[i][j] < 0:
                    number_text = font.render(str(abs(level[i][j])), True, "white")
                    cell_x = j * num2 + (0.5 * num2) - 10
                    cell_y = i * num1 + (0.5 * num1) - 10
                    screen.blit(number_text, (cell_x, cell_y))



    def draw_player(last_direction):
        # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
        for direction_idx in range(0,4):
            if direction_idx == direction:
                last_direction=direction
                screen.blit(player_images[direction], (player_x, player_y))
        if direction == 4:
            screen.blit(player_images[last_direction], (player_x, player_y))
        return last_direction



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

    def change_colors(color):

        if len(commands_list)> 0:
            if first_movement==True:
                movement_command = current_command
                if current_command == 'right':  # Right
                    screen.blit(arrow_images[0],(player_x+num2, player_y))
                elif current_command == 'left':  # Left
                    screen.blit(arrow_images[1],(player_x-num2, player_y))
                elif current_command == 'up':  # Up
                    screen.blit(arrow_images[2],(player_x, player_y-num1))
                elif current_command == 'down':  # Down
                    screen.blit(arrow_images[3],(player_x, player_y+num1))
            else:
                movement_command = commands_list[0]
                if commands_list[0] == 'right':  # Right
                    screen.blit(arrow_images[0],(player_x+num2, player_y))
                elif commands_list[0] == 'left':  # Left
                    screen.blit(arrow_images[1],(player_x-num2, player_y))
                elif commands_list[0] == 'up':  # Up
                    screen.blit(arrow_images[2],(player_x, player_y-num1))
                elif commands_list[0] == 'down':  # Down
                    screen.blit(arrow_images[3],(player_x, player_y+num1))

            pygame.display.flip()
            said_command = prompts.pop(0)

            # print(last_direction)
            # print(movement_command)
            # print(said_command)
            time.sleep(1.4)
            # Green (Imagined Speech)
            color = "green"
            draw_board(color)
            draw_player(last_direction)
            pygame.display.flip()
            if not dev_mode: mrkstream_allowed_turn_out.push_sample(pylsl.vectorstr([str(said_command)]))
            time.sleep(1.4)

            # Blue (Auditory Speech)
            color = "blue"
            draw_board(color)
            draw_player(last_direction)
            pygame.display.flip()
            if not dev_mode: mrkstream_allowed_turn_out.push_sample(pylsl.vectorstr([str("Spoken " + said_command)]))
            time.sleep(1.4)
            color = "white"
            draw_board(color)

            return color


    # Commands
    current_command = commands_list.pop(0)
    goal_x, goal_y = command_leader(current_command, player_y, player_x)

    run = True
    first_movement = True
    while run:
        timer.tick(fps)
        if startup_counter < fps*20 and not game_won:
            moving = False
            startup_counter += 1
        else:
            moving = True

        if moving and first_movement:
           color = change_colors(color)
           first_movement = False


        screen.fill("black")
        draw_board(color)
        center_x = int(player_x + image_xscale//2)
        center_y = int(player_y + image_yscale//2)


        last_direction = draw_player(last_direction)
        turns_allowed = [True,True,True,True]

        if moving:
            player_x, player_y = move_player(player_x, player_y)

        last_activate_turn_tile = check_collisions (last_activate_turn_tile)


        if goal_x == player_x and goal_y == player_y:
            change_colors(color)
            # Change Command
            if len(commands_list) > 0:
                current_command = commands_list.pop(0)
            else:
                current_command = 'None'
                game_won = True
            # Change Direction
            if current_command == "right":
                direction = 0
            if current_command == "left":
                direction = 1
            if current_command == "up":
                direction = 2
            if current_command == "down":
                direction = 3

            goal_x, goal_y = command_leader(current_command, player_y, player_x)


        if  game_won:
            first_movement = True
            draw_misc()
            pygame.display.flip()
            time.sleep(10)
            startup_counter = 0
            start = start_positions_paradigm_SI.pop(0)
            player_x = int(start[0] * num2)
            player_y = int(start[1]* num1)
            direction = start[2]
            direction_command = start[2]
            current_level += 1
            if current_level < len(boards_paradigm_SI):
                level = copy.deepcopy(boards_paradigm_SI[current_level])
                prompts = copy.deepcopy(prompts_paradigm_SI[current_level])
            game_won = False
            commands_list = commands_list_board.pop(0)
            current_command = commands_list.pop(0)
            goal_x, goal_y = command_leader(current_command, player_y, player_x)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if not dev_mode:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_j:
                    button_pressed = True
                if event.type == pygame.KEYUP and event.key == pygame.K_j and button_pressed:
                    button_pressed = False
                    mrkstream_allowed_turn_out.push_sample(pylsl.vectorstr(["Mistake"]))

        pygame.display.flip()
    pygame.quit()
