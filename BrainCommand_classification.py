import mne
from processing_eeg_methods.data_dataclass import ProcessingMethods
from processing_eeg_methods.data_utils import convert_into_independent_channels, data_normalization
import numpy as np
import os
from scipy import signal
import pandas as pd

dataset_info = {  # BrainCommand
    "dataset_name": "braincommand",
    "#_class": 4,
    "target_names": ["Derecha", "Izquierda", "Arriba", "Abajo"],
    "#_channels": 16,
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

def BrainCommand_train(game_mode: str, subject_id: int, selected_classes: list[int], independent_channels: bool = True) -> None:
    filepath: str = 'assets/game_saved_files'

    dataset_info["#_class"] = len(selected_classes)

    data, label = braincommand_dataset_loader(
        filepath, subject_id, game_mode=game_mode
    )
    pm = ProcessingMethods()
    pm.activate_methods(
        spatial_features=False,  # Training is over-fitted. Training accuracy >90
        simplified_spatial_features=False,
        # Simpler than selected_transformers, only one transformer and no frequency bands. No need to activate both at the same time
        ShallowFBCSPNet=True,
        LSTM=False,  # Training is over-fitted. Training accuracy >90
        GRU=False,  # Training is over-fitted. Training accuracy >90
        diffE=False,  # It doesn't work if you only use one channel in the data
        feature_extraction=False,
        number_of_classes=dataset_info["#_class"],
    )

    if independent_channels:
        data, labels = convert_into_independent_channels(
            data, label
        )

    data_t = np.transpose(np.array([data]), (1, 0, 2))

    pm.train(
        subject_id=subject_id,
        data=data_t,
        labels=labels,
        dataset_info=dataset_info,
    )

    saving_folder: str = f'./assets/classifier_data/classifier_{game_mode}_sub{subject_id:02d}'
    os.makedirs(saving_folder, exist_ok=True)

    pm.save_models(
        saving_folder
    )

    print(f"Classifier saved! {game_mode}: Subject {subject_id:02d}")

def BrainCommand_test(eeg, subject_id: int, processing_function: ProcessingMethods, fs: int, independent_channels: bool = True):

    # Preprocess
    x_array = np.array([eeg])
    x_array = x_array[
              :, :, :-9
              ]  # The last channels are accelerometer (x3), gyroscope (x3), validity, battery and counter
    x_array = np.transpose(x_array, (0, 2, 1))
    x_array = signal.detrend(x_array)
    data = data_normalization(x_array)

    if independent_channels:
        data_test, _ = convert_into_independent_channels(
            data, [0]
        )
        data_test = np.transpose(np.array([data_test]), (1, 0, 2))

    frequency_bandwidth = [0.5, 40]
    iir_params = dict(order=8, ftype="butter")
    filt = mne.filter.create_filter(
        data_test,
        fs,
        l_freq=frequency_bandwidth[0],
        h_freq=frequency_bandwidth[1],
        method="iir",
        iir_params=iir_params,
        verbose=False,
    )
    filtered = signal.sosfiltfilt(filt["sos"], data_test)
    data_test = filtered.astype("float64")

    # Process
    probs_by_channels = []
    for pseudo_trial in range(len(data_test)):
        processing_function.test(
            subject_id = subject_id,
            data=np.asarray([data_test[pseudo_trial]]),
            dataset_info=dataset_info,
        )
        probs_by_channels.append(
            processing_function.voting_decision(
            voting_by_mode = False,
            weighted_accuracy= False,
            )
        )

    return np.nanmean(probs_by_channels, axis=0)



if __name__ == "__main__":
    import time

    subject_id = 99
    game_mode = 'calibration3'
    selected_classes = [0, 1, 2, 3]

    start = time.time()
    BrainCommand_train(game_mode, subject_id, selected_classes)
    end = (time.time() - start)/60

    print(f"In total it took {end:.2f} minutes")
    # And to reload for testing
    # processing_function = selected_transformers_function().load(
    #     saving_folder
    # )