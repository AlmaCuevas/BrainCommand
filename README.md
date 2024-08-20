# Brain Command
## Table of Contents
1. [Introduction](https://github.com/AlmaCuevas/BrainCommand/blob/main/README.md#1-introduction)
2. [Requirements](https://github.com/AlmaCuevas/BrainCommand#2-requirements)
3. [Files in Repository](https://github.com/AlmaCuevas/BrainCommand#3-files-in-repository)
4. [How to run](https://github.com/AlmaCuevas/BrainCommand#4-how-to-run)
5. [Credits](https://github.com/AlmaCuevas/BrainCommand#5-credits)
6. [References](https://github.com/AlmaCuevas/BrainCommand#6-references)
7. [License](https://github.com/AlmaCuevas/BrainCommand#7-license)
## 1. Introduction
Brain-Computer Interfaces (BCI) are a groundbreaking technology that have become a subject of public interest, leading to rapid growth in their development and research. A BCI uses a person's brain activity, taken from an electroencephalograph (EEG) signal, and classifies it, turning their thoughts into a real-life activity. **Brain Command** is a BCI designed to classify EEG signals in real time, turning brain activity into actions in a video game.
## 2. Requirements
* In order to run Brain Command the user must  have a python compiler and execuction environment, and install the libraries found in *requirements.txt*. This file is found in the repository. 
* To send instructions into the game we need a signal input. This signal can be either a virtually simulated signal, or a  real EEg signal collected in real time.
##  3. Files in Repository
* **board_calibration.py; board_execution.py; board_execution_closed.py**: These files contain the layout of all the maps that can be played in the game, including the tutorials and the calibration, and their configuration.
* **BrainCommand_train.py**: The signal classifiers for the input signals are created 
* **calibration.py**: Using the boards in *board_calibration.py*, the input signals are used to train the classifiers established in *BrainCommand_train.py* 
* **execution.py**: Configures the LSL communication, and turns the classified signals into commands for the game, and creates the actions made by these commands.
* **game.py**: Runs the game, generating the selected gameboard, and creates game environment.
* **menu.py**: Configuration of menu which works as the first user interface allowing the user to select which action wants to perform.
* **main.py**: Execution which prompts the menu, allowing for user interaction.
* **tutorial_calibration2.py**: Tutorial shown during calibration to make the classifier model.
* **tutorial_green_blue.py**: Tutorial to guide the user for the imagined speech to controll the game.
* **tutorial_blue_only**: Tutorial to practice the real rythm of the imagined speech, without guide.
* **assets**: Folder containing additional visual and audio resources to execute the game.
* **requirements.txt**: File specifying the libraries that need to be installed to be able to run the game.
## 4. How to run
1. Install the additional libraries specified in *requirements.txt*. To install pygame you need to use Windows.
```
pip install -r requirements.txt
```
2. Run *main.py*. You can specify the ID of the player with argument flags.
Example:
```
python main.py --player1 23 --player2 20
```
To visualize the signal in real time you can use Brain Streaming Layer:
```
bsl_stream_viewer -s player1
```
If you don't have an EEG lsl device available you can use the synthetic data from OpenBCI or you can use the dev mode of the game and use the keyboard arrows.
```
python main.py --dev_mode True
```
3. Play the tutorial for a detailed guide to perform the imagined speech for the game.
4. Select "Calibration 1" in the menu if there is only 1 player and you want to perform a maze calibration. Follow the steps shown in the calibration and configure the signal adquisition environment.
5. Select "Calibration 2" in the menu if there are 2 players (either real players or one player plus synthetic data) and you want to perform an square map calibration. Follow the steps shown in the calibration and configure the signal adquisition environment.
6. For execution, select "Solo" or "Competitive" depending on the number of participants of the game.
## 5. Credits
Credits
If you use this code in your project use the citation below:
```
@misc{Cuevas2024BrainCommand,
    title={Brain Command},
    author={Alma Cuevas},
    year={2024},
    url={https://github.com/AlmaCuevas/BrainCommand}}
}
```
Check the game credits to know all the people behind the proyect!

## 6. References
This code was inspired from
* https://github.com/plemaster01/PythonPacman
  * For a tutorial you can check: https://www.youtube.com/watch?v=9H27CimgPsQ&t=684s
* https://github.com/AlmaCuevas/Gamified_Imagined_Speech_Paradigm
## 7. License
[MIT License](https://github.com/AlmaCuevas/BrainCommand/blob/main/LICENSE)

## 8. Thanks
For supervising the project:
* PhD. Luz Maria Alonso Valerdi
* PhD. Alejandro Antonio Torres García
* PhD. Luis Alberto Muñoz Ubando

For supporting the project, making it possible, thanks to:
* ITESM
* CONAHCYT
* Neurotechs

Thanks to all the collaborators on Neurotechs of the Brain-Computer Research Project.