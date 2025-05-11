import mne
from mne import EpochsArray
from processing_eeg_methods.data_dataclass import ProcessingMethods
from processing_eeg_methods.data_utils import convert_into_independent_channels, data_normalization, balance_samples
import numpy as np
import os
import torch
from processing_eeg_methods.share import GLOBAL_SEED
from scipy import signal
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from eegnet_implementation import EvalModel, EEGNetModel, TrainModel

device = "cpu"

dataset_info = {  # BrainCommand
    "dataset_name": "braincommand",
    "#_class": 4,
    "target_names": ["Derecha", "Izquierda", "Arriba", "Abajo"],
    "#_channels": 8,
    "samples": 325,  # 250*1.3
    "sample_rate": 250,
    "montage": "standard_1020",
    "channels_names": ["F3", "C3", "F5", "FC5", "C5", "F7", "FT7", "T7"],
    "subjects": 1,  # PENDING
    "total_trials": 228,
    "event_dict": {"Derecha": 0, "Izquierda": 1, "Arriba": 2, "Abajo": 3},
}

def braincommand_dataset_loader(
    filepath: str, subject_id: int, game_mode: str = "calibration3"
):
    complete_information = pd.read_csv(
        f"{filepath}/eeg_data_{game_mode}_sub{subject_id:02d}.csv"
    )
    x_list = list(complete_information["time"].apply(eval))
    label = list(complete_information["class"])

    label_0 = label.count(0)
    print(f"label 0 is {label_0}")

    label_1 = label.count(1)
    print(f"label 1 is {label_1}")

    label_2 = label.count(2)
    print(f"label 2 is {label_2}")

    label_3 = label.count(3)
    print(f"label 3 is {label_3}")

    x_array = np.array(x_list)  # trials, time, channels
    x_array = x_array[
        :, :, :-9
    ]  # The last channels are accelerometer (x3), gyroscope (x3), validity, battery and counter
    x_array = np.transpose(x_array, (0, 2, 1))
    x_array = signal.detrend(x_array)

    frequency_bandwidth = [0.5, 40]
    iir_params = dict(order=8, ftype="butter")
    filt = mne.filter.create_filter(
        x_array,
        250,
        l_freq=frequency_bandwidth[0],
        h_freq=frequency_bandwidth[1],
        method="iir",
        iir_params=iir_params,
        verbose=False,
    )
    filtered = signal.sosfiltfilt(filt["sos"], x_array)
    filtered = filtered.astype("float64")
    return filtered, label

def BrainCommand_train(game_mode: str, subject_id: int, selected_classes: list[int], independent_channels: bool = False) -> None:
    filepath: str = 'assets/game_saved_files'

    dataset_info["#_class"] = len(selected_classes)

    data, label = braincommand_dataset_loader(
        filepath, subject_id, game_mode=game_mode
    )
    event_dict = {"Derecha": 0, "Izquierda": 1, "Arriba": 2, "Abajo": 3}

    if independent_channels:
        data, label = convert_into_independent_channels(
            data, label
        )
        dataset_info["channels_names"] = ["Fz"]
        data= np.transpose(np.array([data]), (1, 0, 2))

    events = np.column_stack(
        (
            np.arange(
                0,
                dataset_info["sample_rate"] * data.shape[0],
                dataset_info["sample_rate"],
            ),
            np.zeros(len(label), dtype=int),
            np.array(label),
        )
    )
    epochs = EpochsArray(
        data,
        info=mne.create_info(
            sfreq=dataset_info["sample_rate"],
            ch_types="eeg",
            ch_names=dataset_info["channels_names"],
        ),
        events=events,
        event_id=event_dict,
        baseline=(None, None),
    )

    label = epochs.events[:, 2].astype(np.int64)  # To always keep the right format
    data = epochs.get_data()

    X_train = torch.Tensor(data).unsqueeze(1)
    y_train = torch.LongTensor(label)

    train_dataset = TensorDataset(X_train, y_train)

    input_size = (1, data.shape[1], data.shape[2])
    eegnet_model = EEGNetModel(chans=input_size[1], classes=len(selected_classes), time_points=input_size[2]).to(device)

    # Training Hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    trainer = TrainModel()
    trained_eegnet_model = trainer.train_model(eegnet_model, train_dataset, learning_rate=LEARNING_RATE,
                                       batch_size=BATCH_SIZE, epochs=EPOCHS)

    saving_folder: str = f'./assets/classifier_data/classifier_{game_mode}_sub{subject_id:02d}'
    os.makedirs(saving_folder, exist_ok=True)
    file_name: str = saving_folder + f'/eegnet_model_subject_{subject_id}.pth'

    torch.save(trained_eegnet_model.state_dict(), file_name)

    print(f"Classifier saved! {game_mode}: Subject {subject_id:02d}")

def BrainCommand_online_train_test(eeg_train, label_train, eeg_test, trained_eegnet_model,  fs: int, independent_channels: bool = False):
    if eeg_train:
        print("Pre-process training data")
        x_array_train = np.array(eeg_train)  # trials, time, channels
        x_array_train = x_array_train[
            :, :, :-9
        ]  # The last channels are accelerometer (x3), gyroscope (x3), validity, battery and counter
        x_array_train = np.transpose(x_array_train, (0, 2, 1))
        x_array_train = signal.detrend(x_array_train)
        print(x_array_train.shape)

        frequency_bandwidth = [0.5, 40]
        iir_params = dict(order=8, ftype="butter")
        filt = mne.filter.create_filter(
            x_array_train,
            250,
            l_freq=frequency_bandwidth[0],
            h_freq=frequency_bandwidth[1],
            method="iir",
            iir_params=iir_params,
            verbose=False,
        )
        filtered = signal.sosfiltfilt(filt["sos"], x_array_train)
        filtered_train = filtered.astype("float64")

        X_train = torch.Tensor(filtered_train).unsqueeze(1)
        y_train = torch.LongTensor(label_train)

        train_dataset = TensorDataset(X_train, y_train)

        # Training Hyperparameters
        print(f"Starts training...")
        EPOCHS = 100
        BATCH_SIZE = 64
        LEARNING_RATE = 0.001
        trained_eegnet_model = EEGNetModel(chans=8, classes=4,
                                                   time_points=325)
        trainer = TrainModel()


        trained_eegnet_model = trainer.train_model(trained_eegnet_model, train_dataset, learning_rate=LEARNING_RATE,
                                                   batch_size=BATCH_SIZE, epochs=EPOCHS)

    print(f"Starts testing...")
    # Preprocess
    x_array = np.array([eeg_test])
    x_array = x_array[
              :, :, :-9
              ]  # The last channels are accelerometer (x3), gyroscope (x3), validity, battery and counter
    x_array = np.transpose(x_array, (0, 2, 1))
    x_array = signal.detrend(x_array)
    data = data_normalization(x_array)

    if independent_channels:
        data, _ = convert_into_independent_channels(
            data, [0]
        )
        data = np.transpose(np.array([data]), (1, 0, 2))

    frequency_bandwidth = [0.5, 40]
    iir_params = dict(order=8, ftype="butter")
    filt = mne.filter.create_filter(
        data,
        fs,
        l_freq=frequency_bandwidth[0],
        h_freq=frequency_bandwidth[1],
        method="iir",
        iir_params=iir_params,
        verbose=False,
    )
    filtered = signal.sosfiltfilt(filt["sos"], data)
    data_test = filtered.astype("float64")

    data_test = torch.Tensor(data_test).unsqueeze(1)
    y_test = torch.LongTensor([0]) # just to fill the space

    test_dataset = TensorDataset(data_test, y_test)
    eval_model = EvalModel(trained_eegnet_model)

    return eval_model.test_model(test_dataset) # returns probabilities


def BrainCommand_test(eeg, trained_eegnet_model,  fs: int, independent_channels: bool = False):

    # Preprocess
    x_array = np.array([eeg])
    x_array = x_array[
              :, :, :-9
              ]  # The last channels are accelerometer (x3), gyroscope (x3), validity, battery and counter
    x_array = np.transpose(x_array, (0, 2, 1))
    x_array = signal.detrend(x_array)
    data = data_normalization(x_array)

    if independent_channels:
        data, _ = convert_into_independent_channels(
            data, [0]
        )
        data = np.transpose(np.array([data]), (1, 0, 2))

    frequency_bandwidth = [0.5, 40]
    iir_params = dict(order=8, ftype="butter")
    filt = mne.filter.create_filter(
        data,
        fs,
        l_freq=frequency_bandwidth[0],
        h_freq=frequency_bandwidth[1],
        method="iir",
        iir_params=iir_params,
        verbose=False,
    )
    filtered = signal.sosfiltfilt(filt["sos"], data)
    data_test = filtered.astype("float64")

    data_test = torch.Tensor(data_test).unsqueeze(1)
    y_test = torch.LongTensor([0]) # just to fill the space

    test_dataset = TensorDataset(data_test, y_test)
    eval_model = EvalModel(trained_eegnet_model)

    return eval_model.test_model(test_dataset) # returns probabilities

if __name__ == "__main__":
    import time

    subject_id = 80
    game_mode = 'calibration3'
    selected_classes = [0, 1, 2, 3]

    start = time.time()
    BrainCommand_train(game_mode, subject_id, selected_classes)
    end = (time.time() - start)/60

    print(f"In total it took {end:.2f} minutes")
