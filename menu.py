import pygame
import execution
import tutorial_explanation

class Menu():
    def __init__(self, game, dev_mode: bool = False, player1_ID: int = 0, player2_ID: int = 0):
        self.game = game
        self.dev_mode = dev_mode
        self.mid_w, self.mid_h = self.game.DISPLAY_W / 2, self.game.DISPLAY_H / 2
        self.run_display = True
        self.cursor_rect = pygame.Rect(0, 0, 20, 20)
        self.offset = - 350
        self.player1_ID = player1_ID
        self.player2_ID = player2_ID

    def draw_cursor(self):
        self.game.draw_text('*', 40, self.cursor_rect.x, self.cursor_rect.y)

    def blit_screen(self):
        self.game.window.blit(self.game.display, (0, 0))
        pygame.display.update()
        self.game.reset_keys()

class MainMenu(Menu):
    def __init__(self, game, dev_mode: bool = False, player1_ID: int = 0, player2_ID: int = 0):
        Menu.__init__(self, game, dev_mode, player1_ID, player2_ID)
        self.state = "Tutorial"
        self.tutorial_1_x, self.tutorial_1_y = self.mid_w, self.mid_h - 150
        self.calibration_3_x, self.calibration_3_y = self.mid_w, self.mid_h - 90
        self.singleplayerx, self.singleplayery = self.mid_w, self.mid_h - 30
        # self.free_singleplayer_x, self.free_singleplayer_y = self.mid_w, self.mid_h + 60
        # self.multiplayerx, self.multiplayery = self.mid_w, self.mid_h + 120
        self.creditsx, self.creditsy = self.mid_w, self.mid_h + 150
        self.cursor_rect.midtop = (self.tutorial_1_x + self.offset, self.tutorial_1_y)

    def display_menu(self):
        self.run_display = True
        while self.run_display:
            self.game.check_events()
            self.check_input()
            self.game.display.fill(self.game.BLACK)
            self.game.draw_text('Brain Command', 70, self.game.DISPLAY_W / 2, self.game.DISPLAY_H / 2 - 300)
            self.game.draw_text("Tutorial", 40, self.tutorial_1_x, self.tutorial_1_y)
            self.game.draw_text("Calibración", 40, self.calibration_3_x, self.calibration_3_y)
            self.game.draw_text("Solo", 40, self.singleplayerx, self.singleplayery)
            # self.game.draw_text("Solo libre", 40, self.free_singleplayer_x, self.free_singleplayer_y)
            # self.game.draw_text("Competitivo", 40, self.multiplayerx, self.multiplayery)
            self.game.draw_text("Créditos", 40, self.creditsx, self.creditsy)
            self.draw_cursor()
            self.blit_screen()


    def move_cursor(self):
        if self.game.DOWN_KEY:
            if self.state == 'Tutorial':
                self.cursor_rect.midtop = (self.calibration_3_x + self.offset, self.calibration_3_y)
                self.state = 'Calibration'
            elif self.state == 'Calibration':
                self.cursor_rect.midtop = (self.singleplayerx + self.offset, self.singleplayery)
                self.state = 'Singleplayer'
            elif self.state == 'Singleplayer':
                self.cursor_rect.midtop = (self.creditsx + self.offset, self.creditsy)
                self.state = 'Credits'
            # elif self.state == 'Free Singleplayer':
            #     self.cursor_rect.midtop = (self.creditsx + self.offset, self.creditsy)
            #     self.state = 'Credits'
            # elif self.state == 'Multiplayer':
            #     self.cursor_rect.midtop = (self.creditsx + self.offset, self.creditsy)
            #     self.state = 'Credits'
            elif self.state == 'Credits':
                self.cursor_rect.midtop = (self.tutorial_1_x + self.offset, self.tutorial_1_y)
                self.state = 'Tutorial'
        elif self.game.UP_KEY:
            if self.state == 'Tutorial':
                self.cursor_rect.midtop = (self.creditsx + self.offset, self.creditsy)
                self.state = 'Credits'
            elif self.state == 'Credits':
                self.cursor_rect.midtop = (self.singleplayerx + self.offset, self.singleplayery)
                self.state = 'Singleplayer'
            # elif self.state == 'Free Singleplayer':
            #     self.cursor_rect.midtop = (self.singleplayerx + self.offset, self.singleplayery)
            #     self.state = 'Singleplayer'
            elif self.state == 'Singleplayer':
                self.cursor_rect.midtop = (self.calibration_3_x + self.offset, self.calibration_3_y)
                self.state = 'Calibration'
            elif self.state == 'Calibration':
                self.cursor_rect.midtop = (self.tutorial_1_x + self.offset, self.tutorial_1_y)
                self.state = 'Tutorial'
            # elif self.state == 'Credits':
            #     self.cursor_rect.midtop = (self.multiplayerx + self.offset, self.multiplayery)
            #     self.state = 'Multiplayer'
            # elif self.state == 'Multiplayer':
            #     self.cursor_rect.midtop = (self.free_singleplayer_x + self.offset, self.free_singleplayer_y)
            #     self.state = 'Free Singleplayer'


    def check_input(self):
        self.move_cursor()
        if self.game.START_KEY:
            sound_start = pygame.mixer.Sound('assets/sounds/start_sound.mp3')
            sound_start.set_volume(0.5)
            sound_start.play()
            if self.state == 'Tutorial':
                self.game.playing = True
                tutorial_explanation.tutorial()  # No interactive video
            elif self.state == 'Calibration':
                self.game.playing = True
                execution.play_game(game_mode='calibration3', player1_subject_id=self.player1_ID, player2_subject_id=self.player2_ID,
                                    dev_mode=self.dev_mode)  # With defined toggle
            elif self.state == 'Singleplayer':
                self.game.playing = True
                execution.play_game(game_mode='singleplayer', player1_subject_id=self.player1_ID, player2_subject_id=0,
                                    dev_mode=self.dev_mode)
            # elif self.state == 'Free Singleplayer':
            #     self.game.playing = True
            #     execution.play_game(game_mode='free singleplayer', player1_subject_id=self.player1_ID, player2_subject_id=0,
            #                         dev_mode=self.dev_mode)
            # elif self.state == 'Multiplayer':
            #     self.game.playing = True
            #     execution.play_game(game_mode='multiplayer', player1_subject_id=self.player1_ID,
            #                         player2_subject_id=self.player2_ID, dev_mode=self.dev_mode)
            elif self.state == 'Credits':
                self.game.curr_menu = self.game.credits
            self.run_display = False

class CreditsMenu(Menu):
    def display_menu(self):
        self.run_display = True
        text_y_position = self.game.DISPLAY_H / 1  
        text_y_offset = 20
    
        while self.run_display:
            self.game.check_events()
            if self.game.START_KEY or self.game.BACK_KEY or (text_y_position < -500):
                self.game.curr_menu = self.game.main_menu
                self.run_display = False
            self.game.display.fill(self.game.BLACK)

            self.game.draw_text('Brain Command', 70, self.game.DISPLAY_W / 2, text_y_position)

            self.game.draw_text('Hecho por:', 45, self.game.DISPLAY_W / 2, self.game.DISPLAY_H / 1 + 50 + text_y_offset)
            self.game.draw_text('Alma Cuevas', 40, self.game.DISPLAY_W / 2, self.game.DISPLAY_H / 1 + 90 + text_y_offset)
            self.game.draw_text('Edgar Aguilera', 40, self.game.DISPLAY_W / 2, self.game.DISPLAY_H / 1 + 130 + text_y_offset)
            self.game.draw_text('Santiago Mendoza', 40, self.game.DISPLAY_W / 2, self.game.DISPLAY_H / 1 + 170 + text_y_offset)
            self.game.draw_text('Eduardo Rivera', 40, self.game.DISPLAY_W / 2, self.game.DISPLAY_H / 1 + 210 + text_y_offset)
            self.game.draw_text('David Villanueva', 40, self.game.DISPLAY_W / 2, self.game.DISPLAY_H / 1 + 250 + text_y_offset)
            self.game.draw_text('Rebecca De Stefano', 40, self.game.DISPLAY_W / 2, self.game.DISPLAY_H / 1 + 290 + text_y_offset)
            self.game.draw_text('Supervisión:', 45, self.game.DISPLAY_W / 2, self.game.DISPLAY_H / 1 + 380 + text_y_offset)
            self.game.draw_text('Dra. Luz María Alonso Valerdi', 40, self.game.DISPLAY_W / 2, self.game.DISPLAY_H / 1 + 420 + text_y_offset)
            self.game.draw_text('Dr. Alejandro Antonio Torres García', 40, self.game.DISPLAY_W / 2, self.game.DISPLAY_H / 1 + 460 + text_y_offset)
            self.game.draw_text('Dr. Luis Alberto Muñoz Ubando', 40, self.game.DISPLAY_W / 2, self.game.DISPLAY_H / 1 + 500 + text_y_offset)

            text_y_position += -1.2  
            text_y_offset += -1.2

            self.blit_screen()
