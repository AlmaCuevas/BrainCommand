# Build Pac-Man from Scratch in Python with PyGame!!
import copy
import random

from board_execution import multiplayer_execution_boards, multiplayer_player_1_start_execution_positions, multiplayer_player_2_start_execution_positions, singleplayer_start_execution_positions, singleplayer_execution_boards, tutorial_player_1_start_execution_positions, tutorial_execution_boards
import pygame
import math
import time
import pylsl
import numpy as np
import joblib
import pandas as pd


# The report file will only be saved when the game finishes without quitting.
# You don't have to close or open a new game to select a different mode.

# TODO: Training should happen after finishing calibration (CALL CUTOMIZED_PROBS IN A FILE THAT RETURNS THE PKL CLF)
#       TODO: Right after saving the file with calibration, the training should load it and start immediately.
            # TODO: I can't do that until I know which device I am using and what output (EDF? MAT?) they provide.

# todo: pull_sample all available during the blue may be better than pull_chuk if cant get the pull_chunk to just do it once at the right time

# LSL COMMUNICATIONtlet

def lsl_inlet(name:str, number_subject: int = 1):
    info = pylsl.resolve_stream('name', name + str(number_subject))
    inlet = pylsl.stream_inlet(info[0], recover=False)
    print(f'Brain Command has received the {info[0].type()} inlet: {name}, for Player {number_subject}.')
    return inlet

def play_game(game_mode: str, dev_mode: bool = False, process_mode: bool = True, player1_subject_id: int = 0, player2_subject_id: int = 0):
    fs: int = 250

    player1_eeg_data: dict = {'time': [], 'class':[], 'movement index':[0], 'game index':[]}
    player1_subject_id = int(player1_subject_id)

    if game_mode=='calibration1' or game_mode=='calibration2':
        player_1_start_execution_positions = tutorial_player_1_start_execution_positions
        player_2_start_execution_positions = multiplayer_player_2_start_execution_positions # It doesn't matter
        execution_boards = tutorial_execution_boards
    elif game_mode == 'singleplayer':
        player_1_start_execution_positions = singleplayer_start_execution_positions
        player_2_start_execution_positions = multiplayer_player_2_start_execution_positions # It doesn't matter
        execution_boards = singleplayer_execution_boards
    else: # game_mode == 'multiplayer'
        player_1_start_execution_positions = multiplayer_player_1_start_execution_positions
        player_2_start_execution_positions = multiplayer_player_2_start_execution_positions
        execution_boards = multiplayer_execution_boards

        player2_eeg_data: dict = {'time': [], 'class': [], 'movement index': [0], 'game index': []}
        player2_subject_id = int(player2_subject_id)  # It becomes str during the main menu parsing


    if game_mode=='calibration2':
        clf_loading_game_mode = 'calibration1'
    elif game_mode=='singleplayer' or game_mode=='singleplayer':
        clf_loading_game_mode = 'calibration2'
    else:
        clf_loading_game_mode = game_mode

    if process_mode:
        if game_mode=='calibration2' or game_mode == 'singleplayer':
            clf_1 = joblib.load(open(f'assets/classifier_data/classifier_{clf_loading_game_mode}_sub{player1_subject_id:02d}.joblib', 'rb'))
        elif game_mode == 'multiplayer':
            clf_2 = joblib.load(open(f'assets/classifier_data/classifier_{clf_loading_game_mode}_sub{player2_subject_id:02d}.joblib', 'rb'))
    else:
        clf_1 = None
        clf_2 = None

    if not dev_mode:
        eeg_1_in = lsl_inlet('player', 1)  # Don't use special characters or uppercase for the name
        if game_mode == 'multiplayer':
            eeg_2_in = lsl_inlet('player', 2)  # Don't use special characters or uppercase for the name

    # GAME
    pygame.init()
    current_level: int = 0
    # Dimensions
    display_info = pygame.display.Info()  # Get the monitor's display info
    WIDTH: int = int(display_info.current_h)
    HEIGHT: int = int(display_info.current_h)

    level: list[list[int]] = copy.deepcopy(execution_boards[current_level])
    flat_level_list = [
        x
        for xs in level
        for x in xs
    ]
    cookies_at_the_beginning = flat_level_list.count(2)
    div_width: int = len(level[0])
    div_height: int = len(level)
    yscale: int = HEIGHT // div_height
    xscale: int = WIDTH // div_width

    # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN

    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    timer = pygame.time.Clock()
    fps = 60  # This decides how fast the game goes.
    font = pygame.font.Font("RetroFont.ttf", 30)
    color = "white"
    PI = math.pi
    total_game_time: list = []
    player_1_total_game_turns: list = []
    player_1_level_turns: list[int] = []

    ## All Player 2
    player_2_total_game_turns: list = []
    player_2_level_turns: list[int] = []

    ## Images import
    player_2_images: list = [
        pygame.transform.scale(pygame.image.load('assets/extras_images/right_2.png'), (xscale*2, yscale*2)),
        pygame.transform.scale(pygame.image.load('assets/extras_images/left_2.png'), (xscale*2, yscale*2)),
        pygame.transform.scale(pygame.image.load('assets/extras_images/forward_2.png'), (xscale*2, yscale*2)),
        pygame.transform.scale(pygame.image.load('assets/extras_images/back_2.png'), (xscale*2, yscale*2))]

    start_2: list = player_2_start_execution_positions[current_level]
    player_2_player_x: int = int(start_2[0] * xscale)
    player_2_player_y: int = int(start_2[1] * yscale)
    player_2_direction: int = start_2[2]
    player_2_last_direction: int = start_2[2]
    player_2_direction_command: int = start_2[2]
    player_2_last_activate_turn_tile: list[int] = [4, 4]
    player_2_time_to_corner: int = 0
    corner_color = 'blue'

    ## Images import
    player_1_images: list = [
        pygame.transform.scale(pygame.image.load('assets/extras_images/right_1.png'), (xscale*2, yscale*2)),
        pygame.transform.scale(pygame.image.load('assets/extras_images/left_1.png'), (xscale*2, yscale*2)),
        pygame.transform.scale(pygame.image.load('assets/extras_images/forward_1.png'), (xscale*2, yscale*2)),
        pygame.transform.scale(pygame.image.load('assets/extras_images/back_1.png'),
                               (xscale*2, yscale*2))]  # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN

    cookie = pygame.transform.scale(pygame.image.load('assets/extras_images/cookie.png'), (xscale, yscale))

    ## Sounds import
    sound_thud = pygame.mixer.Sound('assets/sounds/thud.mp3')
    sound_go = pygame.mixer.Sound('assets/sounds/go.mp3')
    sound_win = pygame.mixer.Sound('assets/sounds/finish a level.mp3')
    sound_win.set_volume(0.3)

    channel = pygame.mixer.Channel(0)

    ## Positions
    start_1: list = player_1_start_execution_positions[current_level]
    player_1_player_x: int = int(start_1[0] * xscale)
    player_1_player_y: int = int(start_1[1] * yscale)
    player_1_direction: int = start_1[2]
    player_1_last_direction: int = start_1[2]
    player_1_turns_allowed: list[bool] = [False, False, False, False]

    ## Other
    player_1_direction_command: int = start_1[2]
    if dev_mode:
        player_1_speed: int = 5
        original_speed: int = 5
        player_2_speed: int = 5
    else:
        player_1_speed: int = 5
        original_speed: int = 5
        player_2_speed: int = 5
    moving: bool = False
    startup_counter: int = 0
    game_over: bool = False
    game_won: bool = False
    play_won_flag: bool = True
    player_1_last_activate_turn_tile: list[int] = [4, 4]  # Check that in all levels this is a 0 pixel
    player_1_time_to_corner: int = 0
    cookie_winner: list = []
    cookie_winner_2_num: int = 0
    moving_flag_1: bool = True
    start_time_eeg_1: float = 0
    start_time_eeg_2: float = 0

    def draw_text(text: str):
        font = pygame.font.Font("RetroFont.ttf", 300)
        txt_render = font.render(text, True, "lightgrey")
        screen.blit(txt_render,
                    (WIDTH / 2 - txt_render.get_width() / 2, HEIGHT / 2 - txt_render.get_height() / 2))

    def draw_misc(player_num: int, game_mode: str):
        level_done = font.render("¡Nivel Completado!", True, "lightgrey")
        if game_mode == 'multiplayer':
            congrats_winner_str = f"¡Felicidades jugador {player_num}!"
        else:
            congrats_winner_str = "¡Felicidades!"
        congrats_winner = font.render(congrats_winner_str, True, "lightgrey")
        if game_over:
            pygame.draw.rect(screen, "lightgrey", [WIDTH * .05, HEIGHT * .1, WIDTH * .9, HEIGHT * .8], 0, 10)
            pygame.draw.rect(screen, "lightpink4", [WIDTH * .1, HEIGHT * .2, WIDTH * .8, HEIGHT * .6], 0, 10)
            thanks_for_participating = font.render("¡Gracias por jugar!", True, "lightgrey")

            screen.blit(thanks_for_participating,
                        (WIDTH / 2 - thanks_for_participating.get_width() / 2, HEIGHT / 2 - thanks_for_participating.get_height() / 2 + 100))
            screen.blit(congrats_winner,
                            (WIDTH / 2 - congrats_winner.get_width() / 2, HEIGHT / 2 - congrats_winner.get_height() / 2 - 100))
            screen.blit(level_done,
                        (WIDTH / 2 - level_done.get_width() / 2, HEIGHT / 2 - level_done.get_height() / 2))
        elif game_won:
            pygame.draw.rect(screen, "lightgrey", [WIDTH * .05, HEIGHT * .1, WIDTH * .9, HEIGHT * .8], 0, 10)
            pygame.draw.rect(screen, "lightpink4", [WIDTH * .1, HEIGHT * .2, WIDTH * .8, HEIGHT * .6], 0, 10)
            prepare_for_next_level = font.render("¡Prepárate para el siguiente nivel!", True, "lightgrey")
            screen.blit(prepare_for_next_level,
                        (WIDTH / 2 - prepare_for_next_level.get_width() / 2, HEIGHT / 2 - prepare_for_next_level.get_height() / 2 + 100))
            screen.blit(congrats_winner,
                            (WIDTH / 2 - congrats_winner.get_width() / 2, HEIGHT / 2 - congrats_winner.get_height() / 2 - 100))
            screen.blit(level_done,
                        (WIDTH / 2 - level_done.get_width() / 2, HEIGHT / 2 - level_done.get_height() / 2))


    def check_collisions(last_activate_turn_tile:list, player_speed:int, time_to_corner:int, turns_allowed, direction:int, center_x:float,
                         center_y:float, level:list, player_num:int, start_player_time, calibration_moving_flag: bool = True):
        cookie_winner_num = 0
        if player_num == 2:
            right_volume = 0
            left_volume = 1
        else:
            right_volume = 1
            left_volume = 0
        corner_check = copy.deepcopy(turns_allowed)
        corner_check[direction] = False
        if level[center_y // yscale][center_x // xscale] == 1:
            level[center_y // yscale][center_x // xscale] = 0
        elif level[center_y // yscale][center_x // xscale] == 2:
            cookie_winner_num = player_num
            level[center_y // yscale][center_x // xscale] = 0
        if sum(corner_check) >= 2 or corner_check == turns_allowed:
            if level[last_activate_turn_tile[0]][last_activate_turn_tile[1]] != -1 * player_num and time_to_corner > 10:
                channel.play(sound_thud)
                start_player_time = time.time()
                channel.set_volume(right_volume, left_volume)
                level[center_y // yscale][center_x // xscale] = -1 * player_num
                last_activate_turn_tile = [center_y // yscale, center_x // xscale]
                player_speed = 0
        elif level[last_activate_turn_tile[0]][last_activate_turn_tile[1]] == -1 * player_num and calibration_moving_flag:
            channel.play(sound_go)
            channel.set_volume(right_volume, left_volume)
            level[last_activate_turn_tile[0]][last_activate_turn_tile[1]] = 0
            player_speed = original_speed
            time_to_corner = 0
        return last_activate_turn_tile, player_speed, time_to_corner, level, cookie_winner_num, start_player_time

    def draw_player(direction:int, last_direction:int, player_x:float, player_y:float, player_images, moving_flag: bool = True):
        # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
        if moving_flag:
            for direction_idx in range(0, 4):
                if direction_idx == direction:
                    last_direction = direction
                    screen.blit(player_images[direction], (player_x - xscale/2, player_y - yscale/2))
        else:
            screen.blit(player_images[last_direction], (player_x - xscale/2, player_y - yscale/2))
        return last_direction

    def check_position(direction: int, center_x: int, center_y: int, level: list) -> list[bool]:
        turns = [False, False, False, False]  # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
        half_scale = xscale // 2 + 5
        if direction == 2 or direction == 3:
            if xscale // 3 <= center_x % xscale <= xscale:
                if level[(center_y + half_scale) // yscale][center_x // xscale] < 3:
                    if dev_mode: pygame.draw.circle(screen, 'pink', (center_x, (center_y + half_scale)), 20, 1)
                    turns[3] = True
                if level[(center_y - half_scale - 10) // yscale][center_x // xscale] < 3:
                    if dev_mode: pygame.draw.circle(screen, 'pink', (center_x, (center_y - half_scale - 10)), 20, 1)
                    turns[2] = True
            if yscale // 3 <= center_y % yscale <= yscale:
                if level[center_y // yscale][(center_x - xscale) // xscale] < 3:
                    if dev_mode: pygame.draw.circle(screen, 'pink', (center_x - xscale, center_y), 20, 1)
                    turns[1] = True
                if level[center_y // yscale][(center_x + xscale) // xscale] < 3:
                    if dev_mode: pygame.draw.circle(screen, 'pink', (center_x + xscale, center_y), 20, 1)
                    turns[0] = True
        elif direction == 0 or direction == 1:
            if xscale // 3 <= center_x % xscale <= xscale:
                if level[(center_y + yscale) // yscale][center_x // xscale] < 3:
                    if dev_mode: pygame.draw.circle(screen, 'green', (center_x, center_y + yscale), 20, 1)
                    turns[3] = True
                if level[(center_y - yscale) // yscale][center_x // xscale] < 3:
                    if dev_mode: pygame.draw.circle(screen, 'green', (center_x, center_y - yscale), 20, 1)
                    turns[2] = True
            if yscale // 3 <= center_y % yscale <= yscale:
                if level[center_y // yscale][(center_x - half_scale - 8) // xscale] < 3:
                    if dev_mode: pygame.draw.circle(screen, 'white', ((center_x - half_scale - 8), center_y), 20, 1)
                    turns[1] = True
                if level[center_y // yscale][(center_x + half_scale) // xscale] < 3:
                    if dev_mode: pygame.draw.circle(screen, 'red', ((center_x + half_scale), center_y), 20, 1)
                    turns[0] = True
        return turns

    def move_player(direction:int, turns_allowed, play_x:int, play_y:int, player_speed:int):
        # r, l, u, d
        # If current direction is right and right is allowed, move right
        if direction == 0 and turns_allowed[0]:
            play_x += player_speed
        elif direction == 1 and turns_allowed[1]:
            play_x -= player_speed
        elif direction == 2 and turns_allowed[2]:
            play_y -= player_speed
        elif direction == 3 and turns_allowed[3]:
            play_y += player_speed
        return play_x, play_y

    def draw_board(level: list, color: str, corner_color: str, center_1_x: int, center_1_y: int, center_2_x: int = 0, center_2_y: int = 0):
        for i in range(len(level)):
            for j in range(len(level[i])):
                if level[i][j] == 1:
                    pygame.draw.circle(
                        screen,
                        "white",
                        (j * xscale + (0.5 * xscale), i * yscale + (0.5 * yscale)),
                        4,
                    )
                if level[i][j] == 2:
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
                        corner_color,
                        [center_1_x - xscale, center_1_y - yscale, 60, 60],
                        border_radius=10,
                    )
                if level[i][j] == -2:
                    pygame.draw.rect(
                        screen,
                        corner_color,
                        [center_2_x - xscale, center_2_y - yscale, 60, 60],
                        border_radius=10,
                    )
        return level

    def decision_maker(eeg_in, player_speed: int, start_time_eeg: float, player_turns_allowed: list[bool], player_eeg_data: dict, player_direction_command: int, player_level_turns: list[int], clf):
        ## Section to process direction prediction with the EEG.
        movement_option = [0, 1, 2, 3]
        if time.time() - start_time_eeg > 1.4 and player_speed == 0:
            eeg, t_eeg = eeg_in.pull_chunk(timeout=0, max_samples=int(1.4 * fs))  # 1.4 seconds by Fs
            if eeg:
                if process_mode:
                    data = np.transpose(np.asarray([eeg]), (0, 2, 1))
                    probs_array = clf.predict_proba(data)[0]
                    valid_array = [0 if not flag else x for x, flag in zip(probs_array, player_turns_allowed)]
                    prediction_movement_1 = np.argmax(
                        valid_array)  # this one just chooses the highest value from available, if you want to add a difference threshold between the highest and the second highest, you have to do it before this,
                else:
                    allowed_movement_random = [x for x, flag in zip(movement_option, player_turns_allowed) if
                                                 flag]
                    prediction_movement_1 = allowed_movement_random[
                        random.randint(0, len(allowed_movement_random) - 1)]  # exclusive range

                player_eeg_data['time'].append(eeg)
                player_eeg_data['class'].append(player_1_direction)
                player_eeg_data['game index'].append(len(player_1_total_game_turns))
                player_eeg_data['movement index'].append(len(player_level_turns))
                print(f'Classifier returned: {prediction_movement_1}')

                player_level_turns.append(player_direction_command) # Save before changing it
                player_direction_command = prediction_movement_1
                player_speed = original_speed  # Otherwise speed doesn't return, that it's only for arrow key

        return player_speed, player_direction_command, player_level_turns, player_eeg_data

    run = True
    start_time = time.time() # In case you are running dev_mode
    while run:
        timer.tick(fps)
        screen.fill("black")
        if startup_counter < 200 and not game_over and not game_won and not dev_mode:
            moving = False
            startup_counter += 1
            if startup_counter < 60:
                draw_text('3')
            elif startup_counter < 120:
                draw_text('2')
            elif startup_counter < 180:
                draw_text('1')
            else:
                draw_text('GO!')
                start_time = time.time()
        else:
            moving = True

        player_1_center_x: int = player_1_player_x + xscale // 2
        player_1_center_y: int = player_1_player_y + yscale // 2
        if game_mode == 'multiplayer':
            player_2_center_x: int = player_2_player_x + xscale // 2
            player_2_center_y: int = player_2_player_y + yscale // 2
            level = draw_board(level, color, corner_color, player_1_center_x, player_1_center_y, player_2_center_x, player_2_center_y)
        else:
            level = draw_board(level, color, corner_color, player_1_center_x, player_1_center_y)

        draw_misc(cookie_winner[-1:], game_mode)

        if moving and not game_won and not game_over:
            player_1_last_direction = draw_player(player_1_direction, player_1_last_direction, player_1_player_x, player_1_player_y, player_1_images, moving_flag_1)
            if game_mode == 'multiplayer': player_2_last_direction = draw_player(player_2_direction, player_2_last_direction, player_2_player_x, player_2_player_y, player_2_images)



            player_1_turns_allowed = check_position(player_1_direction, player_1_center_x, player_1_center_y, level)
            if game_mode == 'multiplayer': player_2_turns_allowed = check_position(player_2_direction,
                                                                                   player_2_center_x, player_2_center_y,
                                                                                   level)

            if game_mode == 'calibration1':
                if player1_eeg_data['movement index'][-1] != len(player_1_level_turns) or not moving_flag_1:  # Valid direction or already EEG caption in process
                    # Right after the person click the arrow, we get the last 1.4s when they also thought the movement
                    eeg_1, t_eeg_1 = eeg_1_in.pull_chunk(timeout=0,
                                                         max_samples=int(1.4 * fs))  # Take the last 1.4 seconds
                    if eeg_1:
                        player1_eeg_data['game index'].append(len(player_1_total_game_turns))
                        player1_eeg_data['movement index'].append(len(player_1_level_turns))
                        player1_eeg_data['class'].append(player_1_direction)
                        player1_eeg_data['time'].append(eeg_1)
                else:
                    corner_color = 'blue'

            if not dev_mode and game_mode != 'calibration1':
                player_1_speed, player_1_direction_command, player_1_level_turns, player1_eeg_data = decision_maker(
                    eeg_1_in, player_1_speed, start_time_eeg_1, player_1_turns_allowed, player1_eeg_data,
                    player_1_direction_command, player_1_level_turns, clf_1)
                if game_mode == 'multiplayer':
                    player_2_speed, player_2_direction_command, player_2_level_turns, player2_eeg_data = decision_maker(
                        eeg_2_in, player_2_speed, start_time_eeg_2, player_2_turns_allowed, player2_eeg_data,
                        player_2_direction_command, player_2_level_turns, clf_2)

            if moving_flag_1:
                player_1_player_x, player_1_player_y = move_player(player_1_direction, player_1_turns_allowed, player_1_player_x, player_1_player_y, player_1_speed)
            if game_mode == 'multiplayer': player_2_player_x, player_2_player_y = move_player(player_2_direction, player_2_turns_allowed, player_2_player_x, player_2_player_y, player_2_speed)

            player_1_last_activate_turn_tile, player_1_speed, player_1_time_to_corner, level, cookie_winner_1_num, start_time_eeg_1 = check_collisions(player_1_last_activate_turn_tile, player_1_speed, player_1_time_to_corner, player_1_turns_allowed, player_1_direction, player_1_center_x, player_1_center_y, level, 1, start_time_eeg_1, moving_flag_1)
            if game_mode == 'multiplayer': player_2_last_activate_turn_tile, player_2_speed, player_2_time_to_corner, level, cookie_winner_2_num, start_time_eeg_2 = check_collisions(player_2_last_activate_turn_tile, player_2_speed, player_2_time_to_corner, player_2_turns_allowed, player_2_direction, player_2_center_x, player_2_center_y, level, 2, start_time_eeg_2)


            ## Section to decide if the game is finished.
            # Suggestion: Put it in a Function and call it instead
            game_won = False
            flat_level_list = [
                x
                for xs in level
                for x in xs
            ]
            if cookies_at_the_beginning != flat_level_list.count(2):
                if play_won_flag:
                    if cookie_winner_1_num:
                        cookie_winner.append(cookie_winner_1_num)
                    elif cookie_winner_2_num and game_mode == 'multiplayer':
                        cookie_winner.append(cookie_winner_2_num)
                    sound_win.play()
                    total_game_time.append('{:.2f}'.format(time.time() - start_time))
                    player_1_total_game_turns.append(player_1_level_turns[1:])
                    if game_mode == 'multiplayer': player_2_total_game_turns.append(player_2_level_turns[1:])
                    play_won_flag = False
                if len(player_1_start_execution_positions) == current_level+1:
                    game_over = True
                game_won = True

            player_1_time_to_corner += 1
            if game_mode == 'multiplayer': player_2_time_to_corner += 1

            for direction_index in range(0, 4):
                if player_1_direction_command == direction_index and player_1_turns_allowed[direction_index]:
                    player_1_direction = direction_index
                if game_mode == 'multiplayer':
                    if player_2_direction_command == direction_index and player_2_turns_allowed[direction_index]:
                        player_2_direction = direction_index

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if ((player_1_speed == 0 or player_2_speed == 0) and dev_mode) or game_mode == 'calibration1':
                    if event.key == pygame.K_RIGHT and player_1_turns_allowed[0]:
                        player_1_level_turns.append(player_1_direction)
                        player_1_direction_command = 0
                        player_1_speed = original_speed
                    elif event.key == pygame.K_LEFT and player_1_turns_allowed[1]:
                        player_1_level_turns.append(player_1_direction)
                        player_1_direction_command = 1
                        player_1_speed = original_speed
                    elif event.key == pygame.K_UP and player_1_turns_allowed[2]:
                        player_1_level_turns.append(player_1_direction)
                        player_1_direction_command = 2
                        player_1_speed = original_speed
                    elif event.key == pygame.K_DOWN and player_1_turns_allowed[3]:
                        player_1_level_turns.append(player_1_direction)
                        player_1_direction_command = 3
                        player_1_speed = original_speed

                    if game_mode == 'multiplayer':
                        if event.key == pygame.K_d and player_2_turns_allowed[0]:
                            player_2_level_turns.append(player_2_direction)
                            player_2_direction_command = 0
                            player_2_speed = original_speed
                        elif event.key == pygame.K_a and player_2_turns_allowed[1]:
                            player_2_level_turns.append(player_2_direction)
                            player_2_direction_command = 1
                            player_2_speed = original_speed
                        elif event.key == pygame.K_w and player_2_turns_allowed[2]:
                            player_2_level_turns.append(player_2_direction)
                            player_2_direction_command = 2
                            player_2_speed = original_speed
                        elif event.key == pygame.K_s and player_2_turns_allowed[3]:
                            player_2_level_turns.append(player_2_direction)
                            player_2_direction_command = 3
                            player_2_speed = original_speed

                if event.key == pygame.K_SPACE and game_over:
                    if game_mode == 'multiplayer': player_2_total_game_turns.append(player_2_level_turns[1:])
                    run = False
                elif event.key == pygame.K_SPACE and game_won:
                    play_won_flag = True
                    startup_counter = 0
                    current_level += 1
                    if dev_mode:
                        player_1_speed = original_speed
                        if game_mode == 'multiplayer': player_2_speed = original_speed
                    else:
                        player_1_speed = original_speed
                        if game_mode == 'multiplayer': player_2_speed = original_speed
                    if current_level < len(execution_boards):
                        level = copy.deepcopy(execution_boards[current_level])
                        flat_level_list = [
                            x
                            for xs in level
                            for x in xs
                        ]
                        cookies_at_the_beginning = flat_level_list.count(2)
                        start_1 = player_1_start_execution_positions[current_level]
                        if game_mode == 'multiplayer':
                            start_2 = player_2_start_execution_positions[current_level]
                            player_2_direction = start_2[2]
                            player_2_player_x = int(start_2[0] * xscale)
                            player_2_player_y = int(start_2[1] * yscale)
                            player_2_direction_command: int = start_2[2]
                        player_1_direction = start_1[2]
                        player_1_player_x = int(start_1[0] * xscale)
                        player_1_player_y = int(start_1[1] * yscale)
                        player_1_direction_command: int = start_1[2]
                    game_won = False
                    player_1_level_turns = []
                    if game_mode == 'multiplayer': player_2_level_turns = []
            if dev_mode or game_mode == 'calibration1':
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_RIGHT and player_1_direction_command == 0:
                        player_1_direction_command = player_1_direction
                    elif event.key == pygame.K_LEFT and player_1_direction_command == 1:
                        player_1_direction_command = player_1_direction
                    elif event.key == pygame.K_UP and player_1_direction_command == 2:
                        player_1_direction_command = player_1_direction
                    elif event.key == pygame.K_DOWN and player_1_direction_command == 3:
                        player_1_direction_command = player_1_direction

                    if game_mode == 'multiplayer':
                        if event.key == pygame.K_d and player_2_direction_command == 0:
                            player_2_direction_command = player_2_direction
                        elif event.key == pygame.K_a and player_2_direction_command == 1:
                            player_2_direction_command = player_2_direction
                        elif event.key == pygame.K_w and player_2_direction_command == 2:
                            player_2_direction_command = player_2_direction
                        elif event.key == pygame.K_s and player_2_direction_command == 3:
                            player_2_direction_command = player_2_direction

        pygame.display.flip()

    if not dev_mode:
        player1_eeg_data['movement index'] = player1_eeg_data['movement index'][1:] # Avoid the init
        pd.DataFrame(player1_eeg_data).to_csv(
            f'assets/game_saved_files/eeg_data_{game_mode}_sub{player1_subject_id:02d}.csv')
        if game_mode == 'multiplayer':
            player2_eeg_data['movement index'] = player2_eeg_data['movement index'][1:]  # Avoid the init
            pd.DataFrame(player2_eeg_data).to_csv(
                f'assets/game_saved_files/eeg_data_{game_mode}_sub{player2_subject_id:02d}.csv')

        if game_mode != 'multiplayer':
            file = open(f'assets/game_saved_files/time_and_movement_{game_mode}_sub{player1_subject_id:02d}.txt', 'w')
        else:
            file = open(f'assets/game_saved_files/time_and_movement_{game_mode}_sub{player1_subject_id:02d}_and_sub_{player2_subject_id:02d}.txt', 'w')
        file.write(f'game_mode, {game_mode}\n')
        file.write(f'total_game_time, {total_game_time}\n')
        file.write(f'cookie_winner, {cookie_winner}\n')
        file.write(f'player_1_turns, {player_1_total_game_turns}\n')
        file.write(f'player_2_turns, {player_2_total_game_turns}\n')
        file.close()
    print("Congrats! Game finished :D")
