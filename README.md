# Brain Command
## Table of Contents
1. [Introduction](https://github.com/AlmaCuevas/BrainCommand/blob/main/README.md#1-introduction)
2. [Requirements](https://github.com/AlmaCuevas/BrainCommand#2-requirements)
3. [Files in Repository](https://github.com/AlmaCuevas/BrainCommand#3-files-in-repository)
4. [Game Maps](https://github.com/AlmaCuevas/BrainCommand#4-game-maps)
5. [How to run](https://github.com/AlmaCuevas/BrainCommand#5-how-to-run)
6. [Credits](https://github.com/AlmaCuevas/BrainCommand#6-credits)
7. [References](https://github.com/AlmaCuevas/BrainCommand#7-references)
8. [License](https://github.com/AlmaCuevas/BrainCommand#8-license)

## 1. Introduction
This game focuses on PC as a platform for a serious game research-oriented in the action genre. The two-dimensional game features a character that moves in four directions: right, left, up, and down. The main character, an animated soldier, is guided by the human participant acting as the Commander. The goal is to reach the refuge, represented by a green tent at the end of the path. The paradigm includes a single blue square cue behind the main character to indicate the moment for imagined speech. Participants must make decisions and perform imagined speech within 1.3 seconds.

## 2. Requirements
* In order to run Brain Command the user must  have a python compiler and execuction environment, and install the libraries found in *requirements.txt*. This file is found in the repository. 
* To send instructions into the game we need a signal input. This signal can be either a virtually simulated signal, or a  real EEg signal collected in real time.
##  3. Files in Repository
* **board_tutorial.py**: Layout of the maps presented in the tutorial.
* **BrainCommand_train.py**: Functions to load and classify the signal.
* **board_execution_short_maps.py**: Contains the maps that the user play on. 
* **execution.py**: Configures the LSL communication, turns the classified signals into commands for the game, and creates the actions made by these commands.
* **game.py**: Runs the game, generating the selected gameboard, and creates game environment.
* **menu.py**: Configuration of menu which works as the first user interface allowing the user to select which action wants to perform.
* **main.py**: Execution which prompts the menu, allowing for user interaction.
* **assets**: Folder containing additional visual and audio resources to execute the game.
* **requirements.txt**: File specifying the libraries that need to be installed to be able to run the game.
* The other files are extra for development.

## 4. Game Maps
The five original maps function as mazes with a unique solution—one true direction leading from the start to the finish. Level variations are achieved by placing the subject at either the designated starting or ending location within these maps. Calibration has 10 levels (57 movements per direction) and singleplayer 4 by taking only the levels 1, 3, 5 and 7 (22 movements per direction).

![image](https://github.com/user-attachments/assets/d1cb1817-b8d6-4586-8905-b4950b236c74)

![all_maps_horizontal](https://github.com/user-attachments/assets/08cdbb70-9b47-405c-a5b7-fe0b7c83d1bf)


## 5. How to run
1. Install the additional libraries specified in *requirements.txt*. To install pygame you need to use Windows.
```
pip install -r requirements.txt
```
2. Run *main.py*. You can specify the ID of the player with argument flags.
Example:
```
python main.py --player1 23
```
To visualize the signal in real time you can use Brain Streaming Layer:
```
bsl_stream_viewer -s player1
```
If you don't have an EEG lsl device available you can use the synthetic data from OpenBCI or you can use the dev mode of the game and use the keyboard arrows.
```
python main.py --dev_mode True
```
3. Watch the tutorial for a detailed guide to perform the imagined speech for the game.
4. Select "Calibración" to perform a maze calibration. Follow the steps shown in the calibration by completing all maps.
6. For execution, select "Solo". This will only run if the calibration section had run first.
## 6. Credits
Credits
If you use this code in your project use the citation below:
```
@misc{Cuevas2024BrainCommand,
    title={Brain Command},
    author={Alma Cuevas},
    year={2024},
    url={https://github.com/AlmaCuevas/BrainCommand}
}
```
Check the game credits to know all the people behind the proyect!

## 7. References
This code was inspired from
* https://github.com/plemaster01/PythonPacman
  * For a tutorial you can check: https://www.youtube.com/watch?v=9H27CimgPsQ&t=684s
* https://github.com/AlmaCuevas/Gamified_Imagined_Speech_Paradigm
## 8. License
[MIT License](https://github.com/AlmaCuevas/BrainCommand/blob/main/LICENSE)

## 9. Thanks
For supervising the project:
* PhD. Luz Maria Alonso Valerdi
* PhD. Alejandro Antonio Torres García
* PhD. Luis Alberto Muñoz Ubando

For supporting the project, making it possible, thanks to:
* ITESM
* CONAHCYT
* Neurotechs

Thanks to all the collaborators on Neurotechs of the Brain-Computer Research Project.
