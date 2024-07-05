from game import Game
import argparse
from BrainCommand_train import ClfSwitcher # without it, the model doesn't load in the game

parser = argparse.ArgumentParser(description="BrainCommand game. What ID do your players have?")

parser.add_argument("--dev_mode", type=bool, help="Are you developing? True/False", required=False, default=False)
parser.add_argument("--player1", help="Player 1 ID", required=False, default=99)
parser.add_argument("--player2", type=int, help="Player 2 ID", required=False, default=98)

args = parser.parse_args()

g = Game(args.dev_mode, args.player1, args.player2)

while g.running:
    g.curr_menu.display_menu()
    g.game_loop()
