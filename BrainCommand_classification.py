from processing_eeg_methods.classifiers_classes import selected_transformers_function, ProcessingMethod
from processing_eeg_methods.data_loaders import braincommand_dataset_loader
from processing_eeg_methods.data_utils import convert_into_independent_channels
import numpy as np

dataset_info = {  # BrainCommand
                    'dataset_name': 'BrainCommand',
                    '#_class': 4,
                    "target_names": ['Derecha', 'Izquierda', 'Arriba', 'Abajo'],
                    '#_channels': 8,
                    'samples': 350 - 50,  # 250*1.4
                    'sample_rate': 250,
                    'channels_names': ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'],  # This is from the original Unicorn cap
                    'subjects': 0,  # PENDING
                    'total_trials': 228,
                }


def BrainCommand_train(game_mode: str, subject_id: int) -> None:
    data, labels = braincommand_dataset_loader(game_mode, subject_id)

    processing_function = selected_transformers_function()

    data_train, labels_train = convert_into_independent_channels(
        data, labels
    )
    data_train = np.transpose(np.array([data_train]), (1, 0, 2))

    processing_function.train(data_train, labels_train, dataset_info)

    processing_function.save(
        f'assets/classifier_data/classifier_{game_mode}_sub{subject_id:02d}'
    )

    print(f"Classifier saved! {game_mode}: Subject {subject_id:02d}")

def BrainCommand_test(data, processing_function: ProcessingMethod):
    data_test, _ = convert_into_independent_channels(
        data, [0]
    )
    data_test = np.transpose(np.array([data_test]), (1, 0, 2))

    probs_by_channels = []
    for pseudo_trial in range(len(data_test)):
        probs_by_channels.append(processing_function.test(
            data=np.asarray([data_test[pseudo_trial]]),
            dataset_info=dataset_info,
        )
        )

    return np.nanmean(probs_by_channels, axis=0)



if __name__ == "__main__":
    subject_id = 1
    game_mode = 'calibration3'

    BrainCommand_train(game_mode, subject_id)

    # And to reload for testing
    # processing_function = selected_transformers_function().load(
    #     saving_folder
    # )