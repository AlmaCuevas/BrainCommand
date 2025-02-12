import copy
from board_tutorial import boards_tutorial, start_positions_tutorial, commands_list_tutorial
import pygame
import math
import time

def tutorial():
    # GAME
    pygame.init()
    current_level = 0  # Inicialmente, el nivel 0 está en juego

    # Dimensions
    display_info = pygame.display.Info() # Get the monitor's display info
    WIDTH = int(display_info.current_h)
    HEIGHT = int(display_info.current_h)

    level = copy.deepcopy(boards_tutorial[current_level])
    div_width = len(level[0])
    div_height = len(level)
    yscale: int = HEIGHT // div_height
    xscale: int = WIDTH // div_width

    commands_list = commands_list_tutorial.pop(0)
     # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN, 4-STOP

    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    timer = pygame.time.Clock()
    fps = 60  # This decides how fast the game goes. Including pacman and ghosts.
    font = pygame.font.Font("assets/RetroFont.ttf", 30)
    color = "white"
    PI = math.pi


    ## Images import
    player_images = []
    player_images = [pygame.transform.scale(pygame.image.load(f'assets/extras_images/right_1.png'), (xscale, yscale)),
                     pygame.transform.scale(pygame.image.load(f'assets/extras_images/left_1.png'), (xscale, yscale)),
                     pygame.transform.scale(pygame.image.load(f'assets/extras_images/forward_1.png'), (xscale, yscale)),
                     pygame.transform.scale(pygame.image.load(f'assets/extras_images/back_1.png'), (xscale, yscale)),
                     pygame.transform.scale(pygame.image.load(f'assets/extras_images/back_1.png'), (xscale, yscale))]
                    # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN, 4-Bigger Image
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
    cookie_big = pygame.transform.scale(pygame.image.load(f'assets/extras_images/cookie.png'), (xscale * 10, yscale * 10))
    player_image_big = pygame.transform.scale(pygame.image.load(f'assets/extras_images/right_1.png'), (xscale * 10, yscale * 10))


    ## Positions
    start = start_positions_tutorial.pop(0)
    player_x = int(start[0] * xscale)
    player_y = int(start[1] * yscale)
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

    def print_direction_instructions(direction_word: str, complementary_text: str, height_mod: int = 0, color_mod: str = "white"):
        tutorial_text = font.render(direction_word, True, color_mod)
        screen.blit(tutorial_text,
                    (WIDTH / 2 - tutorial_text.get_width() / 2, HEIGHT / 2 - tutorial_text.get_height() / 2 - 300 - height_mod))
        tutorial_text = font.render(complementary_text, True, color_mod)
        screen.blit(tutorial_text,
                    (WIDTH / 2 - tutorial_text.get_width() / 2, HEIGHT / 2 - tutorial_text.get_height() / 2 + 100 + height_mod))

    def draw_misc():
        if current_level == 0 or current_level == 4:
            print_direction_instructions(direction_word="ARRIBA", complementary_text="Moverá el personaje hacia arriba")
        elif current_level == 1 or current_level == 5:
            print_direction_instructions(direction_word="DERECHA", complementary_text="Moverá al personaje hacia su derecha")
        elif current_level == 2 or current_level == 6:
            print_direction_instructions(direction_word="ABAJO",
                                         complementary_text="Moverá al personaje hacia abajo")
        elif current_level == 3 or current_level == 7:
            print_direction_instructions(direction_word="IZQUIERDA",
                                         complementary_text="Moverá al personaje hacia su izquierda")
        if game_won:
            pygame.draw.rect(screen, "lightgrey", [WIDTH*.05, HEIGHT*.1, WIDTH*.9, HEIGHT*.8], 0, 10)
            pygame.draw.rect(screen, "lightpink4", [WIDTH*.1, HEIGHT*.2, WIDTH*.8, HEIGHT*.6], 0, 10)
            level_done = font.render("¡Nivel Completado!", True, "lightgrey")
            prepare_for_next_level = font.render("¡Prepárate para el siguiente nivel!", True, "lightgrey")
            screen.blit(level_done,
                        (WIDTH / 2 - level_done.get_width() / 2, HEIGHT / 2 - level_done.get_height() / 2-100))
            screen.blit(prepare_for_next_level,
                        (WIDTH / 2 - prepare_for_next_level.get_width() / 2,
                         HEIGHT / 2 - prepare_for_next_level.get_height() / 2 + 100))

    def command_leader(current_command, player_y, player_x):
        goal_x=player_x
        goal_y=player_y
        if current_command == 'right':  # Right
            goal_x = player_x + xscale * 3
        elif current_command == 'left':  # Left
            goal_x = player_x - xscale * 3
        elif current_command == 'up':  # Up
            goal_y = player_y - yscale * 3
        elif current_command == 'down':  # Down
            goal_y = player_y + yscale * 3
        return goal_x, goal_y

    def check_collisions(last_activate_turn_tile):
        level[last_activate_turn_tile[0]][last_activate_turn_tile[1]] = 0
        if 0 < player_x < 870:
            if level[center_y // yscale][center_x // xscale] == 1:
                level[center_y // yscale][center_x // xscale] = 0
            if level[center_y // yscale][center_x // xscale] == 2:
                level[center_y // yscale][center_x // xscale] = 0
        return last_activate_turn_tile


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
                            (j * xscale - (xscale * 0.4)) - 2,
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
                            (j * xscale - (xscale * 0.4)) - 2,
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
                if level[i][j] < 0:
                    number_text = font.render(str(abs(level[i][j])), True, "white")
                    cell_x = j * xscale + (0.5 * xscale) - 10
                    cell_y = i * yscale + (0.5 * yscale) - 10
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

    def change_colors() -> None:

        if len(commands_list)>= 0:

            if current_level < 4 or current_level == 8 or current_level ==9:
                # Green (Imagined Speech)
                draw_misc()

                pygame.draw.rect(
                    screen,
                    "green",
                    [center_x - xscale, center_y - yscale, 60, 60],
                    border_radius=10,
                )

                draw_player(last_direction)

                if first_movement==True:
                    movement_command = current_command
                    if current_command == 'right':  # Right
                        screen.blit(arrow_images[0], (player_x + xscale, player_y))
                    elif current_command == 'left':  # Left
                        screen.blit(arrow_images[1], (player_x - xscale, player_y))
                    elif current_command == 'up':  # Up
                        screen.blit(arrow_images[2], (player_x, player_y - yscale))
                    elif current_command == 'down':  # Down
                        screen.blit(arrow_images[3], (player_x, player_y + yscale))
                else:
                    movement_command = commands_list[0]
                    if commands_list[0] == 'right':  # Right
                        screen.blit(arrow_images[0], (player_x + xscale, player_y))
                    elif commands_list[0] == 'left':  # Left
                        screen.blit(arrow_images[1], (player_x - xscale, player_y))
                    elif commands_list[0] == 'up':  # Up
                        screen.blit(arrow_images[2], (player_x, player_y - yscale))
                    elif commands_list[0] == 'down':  # Down
                        screen.blit(arrow_images[3], (player_x, player_y + yscale))

                pygame.display.flip()
                time.sleep(1.4)

            # Blue (Auditory Speech)
            screen.fill("black")
            draw_board("white")
            draw_misc()
            pygame.draw.rect(
                screen,
                "blue",
                [center_x - xscale, center_y - yscale, 60, 60],
                border_radius=10,
            )
            draw_player(last_direction)
            pygame.display.flip()
            time.sleep(1.4)

            draw_board("white")


    # Commands
    current_command = commands_list.pop(0)
    goal_x, goal_y = command_leader(current_command, player_y, player_x)

    run = True
    first_movement = True
    move_counter = 0
    while run:
        timer.tick(fps)
        if startup_counter < fps*3 and not game_won:
            moving = False
            startup_counter += 1
        else:
            moving = True

        if moving and first_movement:
           change_colors()
           first_movement = False

        if move_counter == 0:
            # Bienvenida
            screen.fill("black")
            tutorial_text = font.render("¡Bienvenido al tutorial!", True, "white")
            screen.blit(tutorial_text, (WIDTH / 2 - tutorial_text.get_width() / 2, HEIGHT / 2 - tutorial_text.get_height() / 2))
            pygame.display.flip()
            time.sleep(2)
            # Objetivo
            screen.fill("black")
            tutorial_text = font.render("¡Comanda a tus tropas a un lugar seguro!", True, "white")
            screen.blit(tutorial_text, (WIDTH / 2 - tutorial_text.get_width() / 2, HEIGHT / 2 - tutorial_text.get_height() / 2 - 300))
            screen.blit(player_image_big, (WIDTH * 1 / 6, HEIGHT / 2-100))
            screen.blit(cookie_big, (WIDTH * 1 / 2, HEIGHT / 2-100))
            pygame.display.flip()
            time.sleep(2)
            # Prompts
            screen.fill("black")
            tutorial_text = font.render("Puedes utilizar 4 palabras:", True, "white")
            screen.blit(tutorial_text, (WIDTH / 2 - tutorial_text.get_width() / 2, HEIGHT / 2 - tutorial_text.get_height() / 2 - 300))
            tutorial_text = font.render("Arriba", True, "white")
            screen.blit(cookie, (WIDTH * 1 / 2 - 100, HEIGHT / 2-195))
            screen.blit(tutorial_text, (WIDTH * 1 / 2, HEIGHT / 2-200))
            tutorial_text = font.render("Abajo", True, "white")
            screen.blit(cookie, (WIDTH * 1 / 2 - 100, HEIGHT / 2-95))
            screen.blit(tutorial_text, (WIDTH * 1 / 2, HEIGHT / 2-100))
            tutorial_text = font.render("Izquierda", True, "white")
            screen.blit(cookie, (WIDTH * 1 / 2 - 100, HEIGHT / 2 + 5))
            screen.blit(tutorial_text, (WIDTH * 1 / 2, HEIGHT / 2))
            tutorial_text = font.render("Derecha", True, "white")
            screen.blit(cookie, (WIDTH * 1 / 2 - 100, HEIGHT / 2+105))
            screen.blit(tutorial_text, (WIDTH * 1 / 2, HEIGHT / 2+100))
            pygame.display.flip()
            time.sleep(5)
            # Colores
            screen.fill("black")
            tutorial_text = font.render("¡Demuestra que eres el mejor comandante!", True, "green")
            screen.blit(tutorial_text, (WIDTH / 2 - tutorial_text.get_width() / 2, HEIGHT / 2 - tutorial_text.get_height() / 2-300))
            tutorial_text = font.render("Esto es una guerra de mentes", True, "green")
            screen.blit(tutorial_text, (WIDTH / 2 - tutorial_text.get_width() / 2, HEIGHT / 2 - tutorial_text.get_height() / 2+100))
            tutorial_text_2 = font.render("Comanda solo con HABLA IMAGINADA", True, "green")
            screen.blit(tutorial_text_2,
                        (WIDTH / 2 - tutorial_text_2.get_width() / 2,
                         HEIGHT / 2 - tutorial_text_2.get_height() / 2 + 150))
            draw_board("white")
            pygame.display.flip()
            time.sleep(6)
            screen.fill("black")
            tutorial_text = font.render("En AZUL", True, "blue")
            screen.blit(tutorial_text, (WIDTH / 2 - tutorial_text.get_width() / 2, HEIGHT / 2 - tutorial_text.get_height() / 2-300))
            tutorial_text = font.render("Realiza el HABLA IMAGINADA", True, "blue")
            screen.blit(tutorial_text_2,
                        (WIDTH / 2 - tutorial_text_2.get_width() / 2, HEIGHT / 2 - tutorial_text_2.get_height() / 2 + 150))
            draw_board("white")
            pygame.display.flip()
            time.sleep(6)
            move_counter +=1

        screen.fill("black")
        draw_board("white")
        center_x = int(player_x + xscale // 2)
        center_y = int(player_y + yscale // 2)

        # Tutorial
        if current_level==2 and moving:
            direction = 3
            last_direction = draw_player(last_direction)
        elif current_level==3 and moving:
            direction = 1
            last_direction = draw_player(last_direction)
        elif current_level==4 and moving:
            direction = 2
            last_direction = draw_player(last_direction)
        elif current_level==5 and moving:
            direction = 0
            last_direction = draw_player(last_direction)
        elif current_level==6 and moving:
            direction = 3
            last_direction = draw_player(last_direction)
        elif current_level==7 and moving:
            direction = 1
            last_direction = draw_player(last_direction)
        # Examples
        if current_level==8 and moving and move_counter==1:
            direction = 0
            last_direction = draw_player(last_direction)
            move_counter +=1
        elif current_level==9 and moving and move_counter==2:
            direction = 2
            last_direction = draw_player(last_direction)
            move_counter +=1
        elif current_level==10 and moving and move_counter==3:
            direction = 1
            last_direction = draw_player(last_direction)
            move_counter +=1
        elif current_level==11 and moving and move_counter==4:
            direction = 0
            last_direction = draw_player(last_direction)
            move_counter +=1

        last_direction = draw_player(last_direction)
        draw_misc()
        turns_allowed = [True,True,True,True]

        if moving:
            player_x, player_y = move_player(player_x, player_y)
        last_activate_turn_tile = check_collisions(last_activate_turn_tile)

        if math.isclose(goal_x, player_x, abs_tol = 0) and math.isclose(goal_y, player_y, abs_tol = 0):
            if len(commands_list) != 0:
                change_colors()
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
            draw_misc()
            pygame.display.flip()
            time.sleep(3)
            startup_counter = 0
            if start_positions_tutorial:
                start = start_positions_tutorial.pop(0)
            else:
                run = False
            player_x = int(start[0] * xscale)
            player_y = int(start[1] * yscale)
            direction = start[2]
            direction_command = start[2]
            current_level += 1
            if current_level < len(boards_tutorial):
                level = copy.deepcopy(boards_tutorial[current_level])
                commands_list = commands_list_tutorial.pop(0)
                game_won = False

                current_command = commands_list.pop(0)
                goal_x, goal_y = command_leader(current_command, player_y, player_x)
                first_movement = True
            else:
                run = False


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        pygame.display.flip()
    #pygame.quit() # We don't quit so it returns to the menu