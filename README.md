# Brain Command
## Table of Contents
1. Introduction
2. Requirements
3. Files in Repository 
4. How to run
5. License
## 1. Introduction
Brain-Computer Interfaces (BCI) are a groundbreaking technology that have become a subject of public interest, leading to rapid growth in their development and research. A BCI uses a person's brain activity, taken from an electroencephalograph (EEG) signal, and classifies it, turning their thoughts into a real-life activity. Brain Command is a BCI designed to classify EEG signals in real time, turning brain activity into actions in a video game.
## 2. Requirements
* In order to run Brain Command the user must  have a python compiler and execuction environment, and install the libraries found in *requirements.txt*. This file is found in the repository. 
* To send instructions into the game we need a signal input. This signal can be either a virtually simulated signal, or a  real EEg signal collected in real time.
##  3. Files in Repository
* board_calibration.py; board_execution.py; board_execution_closed.py: These files contain the layout of all the maps that can be played in the game, including the tutorials and the calibration, and their configuration.
* BrainCommand_train.py: The signal classifiers for the input signals are created 
* calibration.py: Using the boards in *board_calibration.py*, the input signals are used to train the classifiers established in *BrainCommand_train.py* 
* execution.py: Configures the LSL communication, and turns the classified signals into commands for the game, and creates the actions made by these commands.
* game.py: Runs the game
