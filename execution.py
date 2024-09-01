import copy
import random

from processing_eeg_methods.data_dataclass import ProcessingMethods
from processing_eeg_methods.data_utils import flat_a_list
from BrainCommand_classification import BrainCommand_train, BrainCommand_test

# from plot_current_trial import check_eeg
from board_execution import (
    multiplayer_execution_boards,
    multiplayer_player1_start_execution_positions,
    multiplayer_player2_start_execution_positions,
    singleplayer_start_execution_positions,
    singleplayer_execution_boards,
    calibration_player1_start_execution_positions,
    calibration_execution_boards,
)
from board_execution_closed_map import (
    closed_map_boards,
    closed_map_player1_positions,
    closed_map_player2_positions,
    desired_directions,
)
from board_execution_short_maps import (
    short_map_boards,
    short_map_positions,
    short_map_directions,
    execution_short_map_boards,
    execution_short_map_positions,
    execution_short_map_directions,
)
import pygame
import math
import time
import pylsl
import numpy as np
import pandas as pd

# The report file will only be saved when the game finishes without quitting.
# You don't have to close or open a new game to select a different mode.


# LSL COMMUNICATION
def lsl_inlet(name: str, number_subject: int = 1):
    info = pylsl.resolve_stream("name", name + str(number_subject))
    inlet = pylsl.stream_inlet(info[0], recover=False)
    print(
        f"Brain Command has received the {info[0].type()} inlet: {name}, for Player {number_subject}."
    )
    return inlet


def get_samples_from_certain_timestamps(eeg_in, start_timestamp, end_timestamp):
    data = []
    timestamps = []
    while True:
        sample, timestamp = eeg_in.pull_sample()
        if start_timestamp <= timestamp <= end_timestamp:
            data.append(sample)
            timestamps.append(timestamp)
        if timestamp > end_timestamp:
            break
    return data, timestamps


def play_game(
    game_mode: str,
    player1_subject_id,
    player2_subject_id,
    dev_mode: bool = False,
    process_mode: bool = True,
):  # Process mode will train after a calibration and test during an execution
    fs: int = 250  # Unicorn Hybrid Black
    calibration_style: str = (
        "only_blue"  # or "green_blue" . "green_blue" doesn't make sense with the closed-map: calibration 2.
    )
    automatic_movement_classes: list = [2, 3]  # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
    movement_option = [0, 1, 2, 3]  # The two words that the person can choose

    player1_eeg_data: dict = {
        "time": [],
        "class": [],
        "movement index": [0],
        "game index": [],
    }  # [0] for init only, I delete it in the saving
    player1_subject_id = int(player1_subject_id)

    player2_eeg_data: dict = {
        "time": [],
        "class": [],
        "movement index": [0],
        "game index": [],
    }
    player2_subject_id = int(
        player2_subject_id
    )  # It becomes str during the main menu parsing

    if game_mode == "calibration1":  # Maze
        player1_start_execution_positions = (
            calibration_player1_start_execution_positions
        )
        player2_start_execution_positions = (
            multiplayer_player2_start_execution_positions  # It doesn't matter
        )
        execution_boards = calibration_execution_boards
    elif game_mode == "calibration2":  # closed-map
        player1_start_execution_positions = closed_map_player1_positions
        player2_start_execution_positions = closed_map_player2_positions
        execution_boards = closed_map_boards
    elif game_mode == "calibration3":  # short map (small mazes)
        player1_start_execution_positions = short_map_positions
        player2_start_execution_positions = (
            multiplayer_player2_start_execution_positions  # It doesn't matter
        )
        execution_boards = short_map_boards
    elif game_mode == "singleplayer":
        player1_start_execution_positions = execution_short_map_positions
        player2_start_execution_positions = (
            multiplayer_player2_start_execution_positions  # It doesn't matter
        )
        execution_boards = execution_short_map_boards
    elif game_mode == "free singleplayer":
        player1_start_execution_positions = singleplayer_start_execution_positions
        player2_start_execution_positions = (
            multiplayer_player2_start_execution_positions  # It doesn't matter
        )
        execution_boards = singleplayer_execution_boards
    else:  # game_mode == 'multiplayer'
        player1_start_execution_positions = (
            multiplayer_player1_start_execution_positions
        )
        player2_start_execution_positions = (
            multiplayer_player2_start_execution_positions
        )
        execution_boards = multiplayer_execution_boards

    if "singleplayer" in game_mode or game_mode == "multiplayer":
        processing_function_loading_game_mode = (
            "calibration3"  # 'calibration2' it can be either.
        )
    else:
        processing_function_loading_game_mode = game_mode

    player1_start_mrk_time = 0
    player1_end_mrk_time = 0
    player2_end_mrk_time = 0

    player1_processing_function = None
    player2_processing_function = None
    if process_mode and not dev_mode:
        if "singleplayer" in game_mode or game_mode == "multiplayer":
            print("loaded processing_function 1")
            player1_processing_function = ProcessingMethods()
            player1_processing_function.activate_methods(
                spatial_features=True,  # Training is over-fitted. Training accuracy >90
                simplified_spatial_features=False,
                # Simpler than selected_transformers, only one transformer and no frequency bands. No need to activate both at the same time
                ShallowFBCSPNet=False,
                LSTM=False,  # Training is over-fitted. Training accuracy >90
                GRU=False,  # Training is over-fitted. Training accuracy >90
                diffE=False,  # It doesn't work if you only use one channel in the data
                feature_extraction=True,
                number_of_classes=len(movement_option),
            )
            player1_processing_function.load_models(
                f"assets/classifier_data/classifier_{processing_function_loading_game_mode}_sub{player1_subject_id:02d}"
            )
        if game_mode == "multiplayer":
            print("loaded processing_function 2")
            player2_processing_function = ProcessingMethods()
            player2_processing_function.activate_methods(
                spatial_features=True,  # Training is over-fitted. Training accuracy >90
                simplified_spatial_features=False,
                # Simpler than selected_transformers, only one transformer and no frequency bands. No need to activate both at the same time
                ShallowFBCSPNet=False,
                LSTM=False,  # Training is over-fitted. Training accuracy >90
                GRU=False,  # Training is over-fitted. Training accuracy >90
                diffE=False,  # It doesn't work if you only use one channel in the data
                feature_extraction=True,
                number_of_classes=len(movement_option),
            )
            player2_processing_function.load_models(
                f"assets/classifier_data/classifier_{processing_function_loading_game_mode}_sub{player2_subject_id:02d}"
            )
    if not dev_mode:
        player1_start_mrk_time = 0
        player1_eeg_in = lsl_inlet(
            "player", 1
        )  # Don't use special characters or uppercase for the name
        if game_mode == "multiplayer" or game_mode == "calibration2":
            player2_start_mrk_time = 0
            player2_eeg_in = lsl_inlet(
                "player", 2
            )  # Don't use special characters or uppercase for the name

    # GAME
    pygame.init()
    current_level: int = 0
    # Dimensions
    display_info = pygame.display.Info()  # Get the monitor's display info
    WIDTH: int = int(display_info.current_h)
    HEIGHT: int = int(display_info.current_h)

    level: list[list[int]] = copy.deepcopy(execution_boards[current_level])

    cookies_at_the_beginning = flat_a_list(level).count(2)
    div_width: int = len(level[0])
    div_height: int = len(level)
    yscale: int = HEIGHT // div_height
    xscale: int = WIDTH // div_width

    # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN

    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    timer = pygame.time.Clock()
    fps = 60  # This decides how fast the game goes.
    font = pygame.font.Font("assets/RetroFont.ttf", 30)
    large_font = pygame.font.Font("assets/RetroFont.ttf", 50)
    small_font = pygame.font.Font("assets/RetroFont.ttf", 20)
    color = "white"
    PI = math.pi
    total_game_time: list = []
    player1_total_game_turns: list = []
    player1_level_turns: list[int] = []

    ## All Player 2
    player2_total_game_turns: list = []
    player2_level_turns: list[int] = []

    ## Images import
    player2_images: list = [
        pygame.transform.scale(
            pygame.image.load("assets/extras_images/right_2.png"),
            (xscale * 2, yscale * 2),
        ),
        pygame.transform.scale(
            pygame.image.load("assets/extras_images/left_2.png"),
            (xscale * 2, yscale * 2),
        ),
        pygame.transform.scale(
            pygame.image.load("assets/extras_images/forward_2.png"),
            (xscale * 2, yscale * 2),
        ),
        pygame.transform.scale(
            pygame.image.load("assets/extras_images/back_2.png"),
            (xscale * 2, yscale * 2),
        ),
    ]

    player2_start: list = player2_start_execution_positions[current_level]
    player2_player_x: int = int(player2_start[0] * xscale)
    player2_player_y: int = int(player2_start[1] * yscale)
    player2_direction: int = player2_start[2]
    player2_last_direction: int = player2_start[2]
    player2_direction_command: int = player2_start[2]
    player2_last_activate_turn_tile: list[int] = [4, 4]
    player2_time_to_corner: int = 0
    corner_color = "blue"

    ## Images import
    player1_images: list = [
        pygame.transform.scale(
            pygame.image.load("assets/extras_images/right_1.png"),
            (xscale * 2, yscale * 2),
        ),
        pygame.transform.scale(
            pygame.image.load("assets/extras_images/left_1.png"),
            (xscale * 2, yscale * 2),
        ),
        pygame.transform.scale(
            pygame.image.load("assets/extras_images/forward_1.png"),
            (xscale * 2, yscale * 2),
        ),
        pygame.transform.scale(
            pygame.image.load("assets/extras_images/back_1.png"),
            (xscale * 2, yscale * 2),
        ),
    ]  # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN

    cookie = pygame.transform.scale(
        pygame.image.load("assets/extras_images/cookie.png"), (xscale, yscale)
    )

    ## Sounds import
    sound_thud = pygame.mixer.Sound("assets/sounds/thud.mp3")
    sound_go = pygame.mixer.Sound("assets/sounds/go.mp3")
    sound_win = pygame.mixer.Sound("assets/sounds/finish a level.mp3")
    sound_check = pygame.mixer.Sound("assets/sounds/check_audio.mp3")
    sound_win.set_volume(0.3)

    channel = pygame.mixer.Channel(0)

    ## Positions
    player1_start: list = player1_start_execution_positions[current_level]
    player1_player_x: int = int(player1_start[0] * xscale)
    player1_player_y: int = int(player1_start[1] * yscale)
    player1_direction: int = player1_start[2]
    player1_last_direction: int = player1_start[2]
    player1_turns_allowed: list[bool] = [False, False, False, False]

    ## Other
    start_time = 0 # Init
    player1_direction_command: int = player1_start[2]
    if dev_mode:
        player1_speed: int = 5
        original_speed: int = 5
        player2_speed: int = 5
    else:
        player1_speed: int = 15
        original_speed: int = 15
        player2_speed: int = 15

    failed_movements = 0
    minimum_total_trials_per_movement = 20
    startup_counter: int = 0
    game_over: bool = False
    game_won: bool = False
    play_won_flag: bool = True
    player1_last_activate_turn_tile: list[int] = [4, 4]  # Check that in all levels this is a 0 pixel
    player1_time_to_corner: int = 0
    cookie_winner: list = []
    player2_cookie_winner_num: int = 0
    player1_moving_flag: bool = True
    player1_start_time_eeg: float = 0
    player2_start_time_eeg: float = 0
    misc_color: str = "lightpink4"

    player1_desired_direction_player = 0  # default to avoid missing variable
    player2_desired_direction_player = 1  # default to avoid missing variable
    desired_directions_map = []  # default to avoid missing variable
    if game_mode == "calibration2":
        desired_directions_map = desired_directions[current_level]
        player1_desired_direction_player = desired_directions_map[0]
        player2_desired_direction_player = desired_directions_map[1]
    elif game_mode == "calibration3":
        desired_directions_map = short_map_directions[current_level]
    elif game_mode == "singleplayer":
        desired_directions_map = execution_short_map_directions[current_level]

    def reset_game(
            current_level,
            player2_direction,
            player2_player_x,
            player2_player_y,
            player2_direction_command,
            player1_direction,
            player1_player_x,
            player1_player_y,
            player1_direction_command,
            player2_speed,
            level,
            player2_level_turns,
            cookies_at_the_beginning,
            player1_desired_direction_player,
            player2_desired_direction_player,
    ):
        play_won_flag = True
        startup_counter = 0
        desired_directions_map = []
        failed_movements = 0
        current_level += 1
        if dev_mode:
            player1_speed = original_speed
            if game_mode == "multiplayer" or game_mode == "calibration2":
                player2_speed = original_speed
        else:
            player1_speed = original_speed
            if game_mode == "multiplayer" or game_mode == "calibration2":
                player2_speed = original_speed
        if current_level < len(execution_boards):
            level = copy.deepcopy(execution_boards[current_level])
            cookies_at_the_beginning = flat_a_list(level).count(2)
            player1_start = player1_start_execution_positions[current_level]
            if game_mode == "calibration2":
                desired_directions_map = desired_directions[current_level]
                player1_desired_direction_player = desired_directions_map[0]
                player2_desired_direction_player = desired_directions_map[1]
            elif game_mode == "calibration3":
                desired_directions_map = short_map_directions[current_level]
            elif game_mode == "singleplayer":
                desired_directions_map = execution_short_map_directions[current_level]
            if game_mode == "multiplayer" or game_mode == "calibration2":
                player2_start = player2_start_execution_positions[current_level]
                player2_direction = player2_start[2]
                player2_player_x = int(player2_start[0] * xscale)
                player2_player_y = int(player2_start[1] * yscale)
                player2_direction_command: int = player2_start[2]
            player1_direction = player1_start[2]
            player1_player_x = int(player1_start[0] * xscale)
            player1_player_y = int(player1_start[1] * yscale)
            player1_direction_command: int = player1_start[2]
        game_won = False
        player1_level_turns = []
        if game_mode == "multiplayer" or game_mode == "calibration2":
            player2_level_turns = []

        return (
            current_level,
            player2_direction,
            player2_player_x,
            player2_player_y,
            player2_direction_command,
            player1_direction,
            player1_player_x,
            player1_player_y,
            player1_direction_command,
            game_won,
            play_won_flag,
            startup_counter,
            player1_speed,
            player1_level_turns,
            player2_speed,
            level,
            player2_level_turns,
            cookies_at_the_beginning,
            player1_desired_direction_player,
            player2_desired_direction_player,
            desired_directions_map,
            failed_movements,
        )

    def toggle_direction(player_direction):
        if player_direction == 2:
            player_direction_command = 3
        elif player_direction == 3:
            player_direction_command = 2
        elif player_direction == 1:
            player_direction_command = 0
        elif player_direction == 0:
            player_direction_command = 1
        return player_direction_command

    def draw_text(text: str):
        font = pygame.font.Font("assets/RetroFont.ttf", 300)
        txt_render = font.render(text, True, "lightgrey")
        screen.blit(
            txt_render,
            (
                WIDTH / 2 - txt_render.get_width() / 2,
                HEIGHT / 2 - txt_render.get_height() / 2,
            ),
        )

    def draw_misc(player_num: int, game_mode: str, misc_color: str):
        level_done = font.render("¡Nivel Completado!", True, "lightgrey")
        press_space_to_continue = small_font.render(
            "[Press space to continue]", True, "lightgrey"
        )
        if game_mode == "multiplayer":
            congrats_winner_str = f"¡Felicidades jugador {player_num}!"
        else:
            congrats_winner_str = "¡Felicidades!"
        congrats_winner = large_font.render(congrats_winner_str, True, "lightgrey")
        if game_over:
            pygame.draw.rect(
                screen,
                "lightgrey",
                [WIDTH * 0.05, HEIGHT * 0.1, WIDTH * 0.9, HEIGHT * 0.8],
                0,
                10,
            )
            pygame.draw.rect(
                screen,
                misc_color,
                [WIDTH * 0.1, HEIGHT * 0.2, WIDTH * 0.8, HEIGHT * 0.6],
                0,
                10,
            )
            thanks_for_participating = font.render(
                "¡Gracias por jugar!", True, "lightgrey"
            )

            screen.blit(
                thanks_for_participating,
                (
                    WIDTH / 2 - thanks_for_participating.get_width() / 2,
                    HEIGHT / 2 - thanks_for_participating.get_height() / 2 + 100,
                ),
            )
            screen.blit(
                congrats_winner,
                (
                    WIDTH / 2 - congrats_winner.get_width() / 2,
                    HEIGHT / 2 - congrats_winner.get_height() / 2 - 150,
                ),
            )
            screen.blit(
                level_done,
                (
                    WIDTH / 2 - level_done.get_width() / 2,
                    HEIGHT / 2 - level_done.get_height() / 2,
                ),
            )
            screen.blit(
                press_space_to_continue,
                (
                    WIDTH / 2 - press_space_to_continue.get_width() / 2,
                    HEIGHT / 2 - press_space_to_continue.get_height() / 2 + 200,
                ),
            )
        elif game_won:
            pygame.draw.rect(
                screen,
                "lightgrey",
                [WIDTH * 0.05, HEIGHT * 0.1, WIDTH * 0.9, HEIGHT * 0.8],
                0,
                10,
            )
            pygame.draw.rect(
                screen,
                misc_color,
                [WIDTH * 0.1, HEIGHT * 0.2, WIDTH * 0.8, HEIGHT * 0.6],
                0,
                10,
            )
            prepare_for_next_level = font.render(
                "¡Prepárate para el siguiente nivel!", True, "lightgrey"
            )
            screen.blit(
                prepare_for_next_level,
                (
                    WIDTH / 2 - prepare_for_next_level.get_width() / 2,
                    HEIGHT / 2 - prepare_for_next_level.get_height() / 2 + 100,
                ),
            )
            screen.blit(
                congrats_winner,
                (
                    WIDTH / 2 - congrats_winner.get_width() / 2,
                    HEIGHT / 2 - congrats_winner.get_height() / 2 - 150,
                ),
            )
            screen.blit(
                level_done,
                (
                    WIDTH / 2 - level_done.get_width() / 2,
                    HEIGHT / 2 - level_done.get_height() / 2,
                ),
            )
            screen.blit(
                press_space_to_continue,
                (
                    WIDTH / 2 - press_space_to_continue.get_width() / 2,
                    HEIGHT / 2 - press_space_to_continue.get_height() / 2 + 200,
                ),
            )

    def check_collisions(
            start_mrk_time,
            last_activate_turn_tile: list,
            player_speed: int,
            time_to_corner: int,
            turns_allowed,
            direction: int,
            center_x: int,
            center_y: int,
            level: list,
            player_num: int,
            start_player_time,
            calibration_moving_flag: bool = True,
    ):
        cookie_winner_num = 0
        if player_num == 2:  # Change values in case you want to control volume per ear
            right_volume = 1
            left_volume = 1
        else:  # player1
            right_volume = 1
            left_volume = 1
        corner_check = copy.deepcopy(turns_allowed)
        corner_check[direction] = False
        if level[center_y // yscale][center_x // xscale] == 1:
            level[center_y // yscale][center_x // xscale] = 0
        elif level[center_y // yscale][center_x // xscale] == 2:
            cookie_winner_num = player_num
            level[center_y // yscale][center_x // xscale] = 0
        if sum(corner_check) >= 2 or corner_check == turns_allowed:
            if (
                    level[last_activate_turn_tile[0]][last_activate_turn_tile[1]]
                    != -1 * player_num
                    and time_to_corner > 10
            ):
                channel.play(sound_thud)
                start_player_time = time.time()
                start_mrk_time = pylsl.local_clock()
                channel.set_volume(right_volume, left_volume)
                level[center_y // yscale][center_x // xscale] = -1 * player_num
                last_activate_turn_tile = [center_y // yscale, center_x // xscale]
                player_speed = 0
        elif (
                level[last_activate_turn_tile[0]][last_activate_turn_tile[1]]
                == -1 * player_num
                and calibration_moving_flag
        ):
            channel.play(sound_go)
            channel.set_volume(right_volume, left_volume)
            level[last_activate_turn_tile[0]][last_activate_turn_tile[1]] = 0
            player_speed = original_speed
            time_to_corner = 0
        return (
            start_mrk_time,
            last_activate_turn_tile,
            player_speed,
            time_to_corner,
            level,
            cookie_winner_num,
            start_player_time,
        )

    def draw_player(
            direction: int,
            last_direction: int,
            player_x: float,
            player_y: float,
            player_images,
            moving_flag: bool = True,
    ):
        # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
        if moving_flag:
            for direction_idx in range(0, 4):
                if direction_idx == direction:
                    last_direction = direction
                    screen.blit(
                        player_images[direction],
                        (player_x - xscale / 2, player_y - yscale / 2),
                    )
        else:
            screen.blit(
                player_images[last_direction],
                (player_x - xscale / 2, player_y - yscale / 2),
            )
        return last_direction

    def check_position(
            direction: int, center_x: int, center_y: int, level: list
    ) -> list[bool]:
        turns = [False, False, False, False]  # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
        half_scale = xscale // 2 + 5
        if direction == 2 or direction == 3:
            if xscale // 3 <= center_x % xscale <= xscale:
                if level[(center_y + half_scale) // yscale][center_x // xscale] < 3:
                    # if dev_mode: pygame.draw.circle(screen, 'pink', (center_x, (center_y + half_scale)), 20, 1)
                    turns[3] = True
                if (
                        level[(center_y - half_scale - 10) // yscale][center_x // xscale]
                        < 3
                ):
                    # if dev_mode: pygame.draw.circle(screen, 'pink', (center_x, (center_y - half_scale - 10)), 20, 1)
                    turns[2] = True
            if yscale // 3 <= center_y % yscale <= yscale:
                if level[center_y // yscale][(center_x - xscale) // xscale] < 3:
                    # if dev_mode: pygame.draw.circle(screen, 'pink', (center_x - xscale, center_y), 20, 1)
                    turns[1] = True
                if level[center_y // yscale][(center_x + xscale) // xscale] < 3:
                    # if dev_mode: pygame.draw.circle(screen, 'pink', (center_x + xscale, center_y), 20, 1)
                    turns[0] = True
        elif direction == 0 or direction == 1:
            if xscale // 3 <= center_x % xscale <= xscale:
                if level[(center_y + yscale) // yscale][center_x // xscale] < 3:
                    # if dev_mode: pygame.draw.circle(screen, 'pink', (center_x, center_y + yscale), 20, 1)
                    turns[3] = True
                if level[(center_y - yscale) // yscale][center_x // xscale] < 3:
                    # if dev_mode: pygame.draw.circle(screen, 'pink', (center_x, center_y - yscale), 20, 1)
                    turns[2] = True
            if yscale // 3 <= center_y % yscale <= yscale:
                if level[center_y // yscale][(center_x - half_scale - 8) // xscale] < 3:
                    # if dev_mode: pygame.draw.circle(screen, 'pink', ((center_x - half_scale - 8), center_y), 20, 1)
                    turns[1] = True
                if level[center_y // yscale][(center_x + half_scale) // xscale] < 3:
                    # if dev_mode: pygame.draw.circle(screen, 'pink', ((center_x + half_scale), center_y), 20, 1)
                    turns[0] = True
        return turns

    def move_player(
            direction: int, turns_allowed, play_x: int, play_y: int, player_speed: int
    ):
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

    def draw_board(
            level: list,
            color: str,
            corner_color: str,
            player1_center_x: int,
            player1_center_y: int,
            player2_center_x: int = 0,
            player2_center_y: int = 0,
    ):
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
                        [
                            (j * xscale + (xscale * 0.5)),
                            (i * yscale + (0.5 * yscale)),
                            xscale,
                            yscale,
                        ],
                        PI / 2,
                        PI,
                        3,
                    )
                if level[i][j] == 7:
                    pygame.draw.arc(
                        screen,
                        color,
                        [
                            (j * xscale + (xscale * 0.5)),
                            (i * yscale - (0.4 * yscale)),
                            xscale,
                            yscale,
                        ],
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
                        [player1_center_x - xscale, player1_center_y - yscale, 60, 60],
                        border_radius=10,
                    )
                if level[i][j] == -2:
                    pygame.draw.rect(
                        screen,
                        corner_color,
                        [player2_center_x - xscale, player2_center_y - yscale, 60, 60],
                        border_radius=10,
                    )
        return level

    def eeg_cleaning_to_prediction(
            eeg,
            processing_function: ProcessingMethods,
            player_turns_allowed,
            subject_id: int,
    ) -> int:
        probs_array = BrainCommand_test(eeg, subject_id, processing_function)[0]
        valid_array = [
            0 if not flag else x for x, flag in zip(probs_array, player_turns_allowed)
        ]
        print(valid_array)
        return np.argmax(
            valid_array
        )  # this one just chooses the highest value from available, if you want to add a difference threshold between the highest and the second highest, you have to do it before this,

    def decision_maker(start_mrk_time, eeg_in, player_total_game_turns, desired_direction: int, player_speed: int,
                       start_time_eeg: float, player_turns_allowed: list[bool], player_eeg_data: dict,
                       player_direction_command: int, player_level_turns: list[int], processing_function, subject_id,
                       desired_directions_map: list = [], failed_movements: int = 0, level: list = [],
                       last_activate_turn_tile: list = [], time_to_corner: int = 0):
        """
        Section to process the direction by predicting with the EEG, or by random if in debug.
        """
        # try:
        #     next_direction = desired_directions_map[len(player_level_turns) - failed_movements]
        # except IndexError:
        #     next_direction = 99  # end, when there is no other direction pending
        # if game_mode == 'singleplayer' and next_direction in automatic_movement_classes and any(
        #         player_turns_allowed[i] for i in automatic_movement_classes) and time_to_corner > 10:
        #     player_direction_command = next_direction
        #     time_to_corner = 0
        #     player_level_turns.append(player_direction_command)
        #     player_speed = original_speed  # Otherwise speed doesn't return, that it's only for arrow key
        # el
        if time.time() - start_time_eeg > 1.4 and player_speed == 0:
            end_mrk_time = pylsl.local_clock()

            # Redraw to avoid the blue lingering
            screen.fill("black")
            level[last_activate_turn_tile[0]][
                last_activate_turn_tile[1]
            ] = 0
            level = draw_board(level, color, corner_color, player1_center_x, player1_center_y)

            screen.blit(
                player1_images[player1_last_direction],
                (player1_player_x - xscale / 2, player1_player_y - yscale / 2),
            ) # Update the player1 image

            pygame.display.flip()

            eeg, t_eeg = get_samples_from_certain_timestamps(eeg_in, start_mrk_time, end_mrk_time) # todo: check that the <325 sample got fixed too
            if len(eeg)>325: # todo: check that the saving got fix the problem with the first line always lost, if yes, we wouldnt need that condition anymore
                eeg = eeg[-325:]
                print(len(eeg))
                # if len(player_level_turns)!=0: # Uncomment if you want to see the trial per trial
                # check_eeg(eeg, t_eeg, start_mrk_time, end_mrk_time)
                if (
                        game_mode == "calibration2"
                ):  # This is the movement decider. Not by keys and not by EEG
                    remainder_to_record = (
                            minimum_total_trials_per_movement
                            - player_eeg_data["class"].count(desired_direction)
                    )  # From total class, how many of current type do I need?
                    if remainder_to_record < 3:
                        remainder_to_record = 3  # If I already have the max for this class, then move to the next one
                    if random.randint(0, remainder_to_record) == 0:
                        prediction_movement = desired_direction
                    else:
                        prediction_movement = toggle_direction(player_direction_command)
                elif game_mode == "calibration3":
                    prediction_movement = desired_directions_map[
                        len(player_level_turns) - failed_movements
                        ]
                elif (
                        game_mode == "free singleplayer"
                        and process_mode
                        and len(player_level_turns) != 0
                ):  # The first movement never have the data complete, that is because the first blue is just to show position. The start movement is in its place.
                    prediction_movement = eeg_cleaning_to_prediction(
                        eeg,
                        processing_function=processing_function,
                        player_turns_allowed=player_turns_allowed,
                        subject_id=subject_id,
                    )
                elif (
                        process_mode and len(player_level_turns) != 0
                ):  # The first movement never have the data complete, that is because the first blue is just to show position. The start movement is in its place.
                    eeg_prediction_movement = eeg_cleaning_to_prediction(
                        eeg,
                        processing_function=processing_function,
                        player_turns_allowed=player_turns_allowed,
                        subject_id=subject_id,
                    )
                    next_direction = desired_directions_map[
                        len(player_level_turns) - failed_movements
                        ]
                    if (
                            eeg_prediction_movement == next_direction
                            or len(set(player_level_turns[-10:])) == 1
                    ):  # If it fails more than 10 times, let the user move on.
                        prediction_movement = next_direction
                    else:
                        prediction_movement = player_level_turns[-1]
                        failed_movements += 1

                else:
                    allowed_movement_random = [
                        x
                        for x, flag in zip(movement_option, player_turns_allowed)
                        if flag
                    ]
                    prediction_movement = allowed_movement_random[
                        random.randint(0, len(allowed_movement_random) - 1)
                    ]  # exclusive range
                    # time_to_corner = 0
                # todo: I took off the automatic movements bc it kept messing up with the game. FIX!
                player_eeg_data["time"].append(
                    eeg[-325:]
                )  # 325 instead of 350 because sometimes the trial doesn't get the full 1.4, instead we are looking for 1.3s.
                if game_mode == "calibration2":
                    player_eeg_data["class"].append(desired_direction)
                else:
                    player_eeg_data["class"].append(prediction_movement)
                player_eeg_data["game index"].append(len(player_total_game_turns))
                player_eeg_data["movement index"].append(len(player_level_turns))
                print(f"Classifier returned: {prediction_movement}")

                player_direction_command = prediction_movement
                player_level_turns.append(player_direction_command)
                player_speed = original_speed  # Otherwise speed doesn't return, that it's only for arrow key

        return (
            player_speed,
            player_direction_command,
            player_level_turns,
            player_eeg_data,
            failed_movements,
            level,
            time_to_corner,
        )

    def ask_for_input(
            calibration_style,
            eeg_data,
            level_turns,
            moving_flag,
            eeg_in,
            start_time_eeg,
            total_game_turns,
            direction_command,
            corner_color,
    ):
        if calibration_style == "only_blue":  # think and press, no forced waiting time
            if (
                    eeg_data["movement index"][-1] != len(level_turns) or not moving_flag
            ):  # Valid direction or already EEG caption in process
                # Right after the person click the arrow, we get the last 1.4s when they also thought the movement
                eeg, t_eeg = eeg_in.pull_chunk(
                    timeout=0, max_samples=int(1.4 * fs)
                )  # Take the last 1.4 seconds
                if eeg:
                    eeg_data["game index"].append(len(total_game_turns))
                    eeg_data["movement index"].append(len(level_turns))
                    eeg_data["class"].append(
                        direction_command
                    )  # given by the latest arrow key
                    eeg_data["time"].append(eeg)
            else:
                corner_color = "blue"

        elif calibration_style == "green_blue":  # press and think
            if (
                    eeg_data["movement index"][-1] != len(level_turns) or not moving_flag
            ):  # Valid direction or already EEG caption in process
                moving_flag = False
                if (
                        time.time() - start_time_eeg > 0.5
                ):  # always 0.5s after deciding with the arrow key
                    corner_color = "blue"
                    if dev_mode:
                        moving_flag = True
                    elif time.time() - start_time_eeg > (
                            0.5 + 1.4
                    ):  # After giving the user 1.4s for the IS
                        eeg, t_eeg = eeg_in.pull_chunk(
                            timeout=0, max_samples=int(1.4 * fs)
                        )  # Take the last 1.4 seconds
                        if eeg:
                            eeg_data["game index"].append(len(total_game_turns))
                            eeg_data["movement index"].append(len(level_turns))
                            eeg_data["class"].append(
                                direction_command
                            )  # given by the latest arrow key
                            eeg_data["time"].append(eeg)
                            moving_flag = True
            else:
                corner_color = "green"
                start_time_eeg = time.time()
        return corner_color, start_time_eeg, moving_flag

    def pregame_countdown(startup_counter, game_over: bool, game_won: bool, dev_mode: bool, start_time):
        if startup_counter < 200 and not game_over and not game_won and not dev_mode:
            moving = False
            startup_counter += 1
            if startup_counter < 60:
                draw_text("3")
            elif startup_counter < 120:
                draw_text("2")
            elif startup_counter < 180:
                draw_text("1")
            else:
                draw_text("GO!")
                start_time = time.time()
        else:
            moving = True
        return moving, startup_counter, start_time

    run = True
    while run:
        timer.tick(fps)
        screen.fill("black")
        moving, startup_counter, start_time = pregame_countdown(startup_counter, game_over, game_won, dev_mode, start_time)

        player1_center_x: int = player1_player_x + xscale // 2
        player1_center_y: int = player1_player_y + yscale // 2
        if game_mode == "multiplayer" or game_mode == "calibration2":
            player2_center_x: int = player2_player_x + xscale // 2
            player2_center_y: int = player2_player_y + yscale // 2
            level = draw_board(level, color, corner_color, player1_center_x, player1_center_y, player2_center_x, player2_center_y)
        else:  # game_mode == 'singleplayer' or game_mode == 'calibration3':
            level = draw_board(level, color, corner_color, player1_center_x, player1_center_y)

        player1_last_direction = draw_player(
            player1_direction,
            player1_last_direction,
            player1_player_x,
            player1_player_y,
            player1_images,
            player1_moving_flag,
        )

        if game_mode == "multiplayer" or game_mode == "calibration2":
            player2_last_direction = draw_player(
                player2_direction,
                player2_last_direction,
                player2_player_x,
                player2_player_y,
                player2_images,
            )

        draw_misc(cookie_winner[-1:], game_mode, misc_color)

        if moving and not game_won and not game_over:
            player1_turns_allowed = check_position(
                player1_direction, player1_center_x, player1_center_y, level
            )
            if game_mode == "multiplayer" or game_mode == "calibration2":
                player2_turns_allowed = check_position(
                    player2_direction, player2_center_x, player2_center_y, level
                )
            if not dev_mode and game_mode == "calibration1":
                corner_color, player1_start_time_eeg, player1_moving_flag = ask_for_input(
                    calibration_style,
                    player1_eeg_data,
                    player1_level_turns,
                    player1_moving_flag,
                    player1_eeg_in,
                    player1_start_time_eeg,
                    player1_total_game_turns,
                    player1_direction_command,
                    corner_color,
                )

            if not dev_mode and game_mode!='calibration1':
                player1_speed, player1_direction_command, player1_level_turns, player1_eeg_data, failed_movements, level, player1_time_to_corner = decision_maker(
                    player1_start_mrk_time, player1_eeg_in, player1_total_game_turns, player1_desired_direction_player, player1_speed, player1_start_time_eeg, player1_turns_allowed, player1_eeg_data,
                    player1_direction_command, player1_level_turns, player1_processing_function, player1_subject_id, desired_directions_map, failed_movements, level, player1_last_activate_turn_tile, player1_time_to_corner)
                if game_mode == 'multiplayer' or game_mode == 'calibration2':
                    player2_speed, player2_direction_command, player2_level_turns, player2_eeg_data, _, _, _ = decision_maker(player2_start_mrk_time,
                                                                                                                              player2_eeg_in, player2_total_game_turns, player2_desired_direction_player, player2_speed, player2_start_time_eeg, player2_turns_allowed, player2_eeg_data,
                                                                                                                              player2_direction_command, player2_level_turns, player2_processing_function, player2_subject_id)

            if player1_moving_flag:
                player1_player_x, player1_player_y = move_player(
                    player1_direction,
                    player1_turns_allowed,
                    player1_player_x,
                    player1_player_y,
                    player1_speed,
                )
            if game_mode == "multiplayer" or game_mode == "calibration2":
                player2_player_x, player2_player_y = move_player(
                    player2_direction,
                    player2_turns_allowed,
                    player2_player_x,
                    player2_player_y,
                    player2_speed,
                )

            (
                player1_start_mrk_time,
                player1_last_activate_turn_tile,
                player1_speed,
                player1_time_to_corner,
                level,
                player1_cookie_winner_num,
                player1_start_time_eeg,
            ) = check_collisions(
                player1_start_mrk_time,
                player1_last_activate_turn_tile,
                player1_speed,
                player1_time_to_corner,
                player1_turns_allowed,
                player1_direction,
                player1_center_x,
                player1_center_y,
                level,
                1,
                player1_start_time_eeg,
                player1_moving_flag,
            )
            if game_mode == "multiplayer" or game_mode == "calibration2":
                (
                    player2_start_mrk_time,
                    player2_last_activate_turn_tile,
                    player2_speed,
                    player2_time_to_corner,
                    level,
                    player2_cookie_winner_num,
                    player2_start_time_eeg,
                ) = check_collisions(
                    player2_start_mrk_time,
                    player2_last_activate_turn_tile,
                    player2_speed,
                    player2_time_to_corner,
                    player2_turns_allowed,
                    player2_direction,
                    player2_center_x,
                    player2_center_y,
                    level,
                    2,
                    player2_start_time_eeg,
                )

            ## Section to decide if the game is finished.
            # Suggestion: Put it in a Function and call it instead
            game_won = False

            if cookies_at_the_beginning != flat_a_list(level).count(2):
                if play_won_flag:
                    if player1_cookie_winner_num:
                        cookie_winner.append(player1_cookie_winner_num)
                    elif player2_cookie_winner_num and (
                        game_mode == "multiplayer" or game_mode == "calibration2"
                    ):
                        cookie_winner.append(player2_cookie_winner_num)
                    sound_win.play()
                    total_game_time.append("{:.2f}".format(time.time() - start_time))
                    player1_total_game_turns.append(player1_level_turns)
                    if game_mode == "multiplayer" or game_mode == "calibration2":
                        player2_total_game_turns.append(player2_level_turns)
                    play_won_flag = False
                if len(player1_start_execution_positions) == current_level + 1:
                    game_over = True
                game_won = True

            player1_time_to_corner += 1
            if game_mode == "multiplayer" or game_mode == "calibration2":
                player2_time_to_corner += 1

            for direction_index in range(0, 4):
                if (
                    player1_direction_command == direction_index
                    and player1_turns_allowed[direction_index]
                ):
                    player1_direction = direction_index
                if game_mode == "multiplayer" or game_mode == "calibration2":
                    if (
                        player2_direction_command == direction_index
                        and player2_turns_allowed[direction_index]
                    ):
                        player2_direction = direction_index

        if (
            game_won and game_mode == "calibration2" and current_level % 4 != 0
        ):  # In calibration, the first one and every 4 maps it gives a break to ask if they want to continue
            misc_color = "aquamarine4"  # if automatic is this color
            if (
                startup_counter < 300
            ):  # Always has 200 from last time, so its actually waiting only for 100 more
                startup_counter += 1  # Automatic reset
            else:
                (
                    current_level,
                    player2_direction,
                    player2_player_x,
                    player2_player_y,
                    player2_direction_command,
                    player1_direction,
                    player1_player_x,
                    player1_player_y,
                    player1_direction_command,
                    game_won,
                    play_won_flag,
                    startup_counter,
                    player1_speed,
                    player1_level_turns,
                    player2_speed,
                    level,
                    player2_level_turns,
                    cookies_at_the_beginning,
                    player1_desired_direction_player,
                    player2_desired_direction_player,
                    desired_directions_map,
                    failed_movements,
                ) = reset_game(
                    current_level,
                    player2_direction,
                    player2_player_x,
                    player2_player_y,
                    player2_direction_command,
                    player1_direction,
                    player1_player_x,
                    player1_player_y,
                    player1_direction_command,
                    player2_speed,
                    level,
                    player2_level_turns,
                    cookies_at_the_beginning,
                    player1_desired_direction_player,
                    player2_desired_direction_player,
                )
        else:
            misc_color = "lightpink4"  # if not-automatic then another color

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if (
                    (player1_speed == 0 or player2_speed == 0) and dev_mode
                ) or game_mode == "calibration1":
                    if event.key == pygame.K_RIGHT and player1_turns_allowed[0]:
                        player1_direction_command = 0
                        player1_level_turns.append(player1_direction_command)
                        player1_speed = original_speed
                    elif event.key == pygame.K_LEFT and player1_turns_allowed[1]:
                        player1_direction_command = 1
                        player1_level_turns.append(player1_direction_command)
                        player1_speed = original_speed
                    elif event.key == pygame.K_UP and player1_turns_allowed[2]:
                        player1_direction_command = 2
                        player1_level_turns.append(player1_direction_command)
                        player1_speed = original_speed
                    elif event.key == pygame.K_DOWN and player1_turns_allowed[3]:
                        player1_direction_command = 3
                        player1_level_turns.append(player1_direction_command)
                        player1_speed = original_speed

                    if game_mode == "multiplayer" or game_mode == "calibration2":
                        if event.key == pygame.K_d and player2_turns_allowed[0]:
                            player2_direction_command = 0
                            player2_level_turns.append(player2_direction_command)
                            player2_speed = original_speed
                        elif event.key == pygame.K_a and player2_turns_allowed[1]:
                            player2_direction_command = 1
                            player2_level_turns.append(player2_direction_command)
                            player2_speed = original_speed
                        elif event.key == pygame.K_w and player2_turns_allowed[2]:
                            player2_direction_command = 2
                            player2_level_turns.append(player2_direction_command)
                            player2_speed = original_speed
                        elif event.key == pygame.K_s and player2_turns_allowed[3]:
                            player2_direction_command = 3
                            player2_level_turns.append(player2_direction_command)
                            player2_speed = original_speed
                if event.key == pygame.K_SPACE and game_over:
                    # I think I forgot to delete this before:
                    # if game_mode == 'multiplayer' or game_mode == 'calibration2': player2_total_game_turns.append(player2_level_turns[1:])
                    run = False
                elif game_won and event.key == pygame.K_SPACE:
                    (
                        current_level,
                        player2_direction,
                        player2_player_x,
                        player2_player_y,
                        player2_direction_command,
                        player1_direction,
                        player1_player_x,
                        player1_player_y,
                        player1_direction_command,
                        game_won,
                        play_won_flag,
                        startup_counter,
                        player1_speed,
                        player1_level_turns,
                        player2_speed,
                        level,
                        player2_level_turns,
                        cookies_at_the_beginning,
                        player1_desired_direction_player,
                        player2_desired_direction_player,
                        desired_directions_map,
                        failed_movements,
                    ) = reset_game(
                        current_level,
                        player2_direction,
                        player2_player_x,
                        player2_player_y,
                        player2_direction_command,
                        player1_direction,
                        player1_player_x,
                        player1_player_y,
                        player1_direction_command,
                        player2_speed,
                        level,
                        player2_level_turns,
                        cookies_at_the_beginning,
                        player1_desired_direction_player,
                        player2_desired_direction_player,
                    )
            if dev_mode or game_mode == "calibration1":
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_RIGHT and player1_direction_command == 0:
                        player1_direction_command = player1_direction
                    elif event.key == pygame.K_LEFT and player1_direction_command == 1:
                        player1_direction_command = player1_direction
                    elif event.key == pygame.K_UP and player1_direction_command == 2:
                        player1_direction_command = player1_direction
                    elif event.key == pygame.K_DOWN and player1_direction_command == 3:
                        player1_direction_command = player1_direction

                    if game_mode == "multiplayer" or game_mode == "calibration2":
                        if event.key == pygame.K_d and player2_direction_command == 0:
                            player2_direction_command = player2_direction
                        elif (
                            event.key == pygame.K_a and player2_direction_command == 1
                        ):
                            player2_direction_command = player2_direction
                        elif (
                            event.key == pygame.K_w and player2_direction_command == 2
                        ):
                            player2_direction_command = player2_direction
                        elif (
                            event.key == pygame.K_s and player2_direction_command == 3
                        ):
                            player2_direction_command = player2_direction

        pygame.display.flip()

    if not dev_mode:
        player1_eeg_data["movement index"] = player1_eeg_data["movement index"][
            1:
        ]  # Avoid the init, consequence is that calibration1 the first one is 1 instead of 0. No better solution found so far
        pd.DataFrame(player1_eeg_data).to_csv(
            f"assets/game_saved_files/eeg_data_{game_mode}_sub{player1_subject_id:02d}.csv"
        )
        output_name_txt = f"assets/game_saved_files/time_and_movement_{game_mode}_sub{player1_subject_id:02d}.txt"
        if game_mode == "multiplayer" or game_mode == "calibration2":
            player2_eeg_data["movement index"] = player2_eeg_data["movement index"][
                1:
            ]  # Avoid the init
            pd.DataFrame(player2_eeg_data).to_csv(
                f"assets/game_saved_files/eeg_data_{game_mode}_sub{player2_subject_id:02d}.csv"
            )
            output_name_txt = f"assets/game_saved_files/time_and_movement_{game_mode}_sub{player1_subject_id:02d}_and_sub_{player2_subject_id:02d}.txt"

        file = open(output_name_txt, "w")
        file.write(f"player1_ID, {player1_subject_id}\n")
        file.write(f"player2_ID, {player2_subject_id}\n")
        file.write(f"game_mode, {game_mode}\n")
        file.write(f"total_game_time, {total_game_time}\n")
        file.write(f"cookie_winner, {cookie_winner}\n")
        file.write(
            f"player1_turns, {player1_total_game_turns}\n"
        )  # Represents the movements of the character
        file.write(f"player2_turns, {player2_total_game_turns}\n")
        file.close()

        if (
            process_mode
        ):  # If a calibration mode was run, then train the classifier for the next round
            if "calibration" in game_mode:
                BrainCommand_train(
                    game_mode, player1_subject_id, selected_classes=movement_option
                )
            # if game_mode == 'calibration2':
            #    BrainCommand_train(game_mode, player2_subject_id, selected_classes=movement_option)

    print("Congrats! Game finished :D")
