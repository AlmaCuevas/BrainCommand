
import joblib
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from scipy import signal
from mne.preprocessing import Xdawn
from mne import EpochsArray
import mne
from autoreject import AutoReject

# todo: install the voting systema as a package and then import it here, delete the repetitions

# MDM() Always nan at the end
classifiers = [ # The Good, Medium and Bad is decided on Torres dataset. This to avoid most of the processings.
    # KNeighborsClassifier(3), # Good
    SVC(kernel='linear', probability=True), # Good
    # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42), # Good # It doesn't have .coef
    # DecisionTreeClassifier(max_depth=5, random_state=42), # Good # It doesn't have .coef
    # RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1, random_state=42), # Good It doesn't have .coef
    # MLPClassifier(alpha=1, max_iter=1000, random_state=42), # Good # 'MLPClassifier' object has no attribute 'coef_'. Did you mean: 'coefs_'?
    # AdaBoostClassifier(algorithm="SAMME", random_state=42), # Medium
    # GaussianNB(), # Medium
    # QuadraticDiscriminantAnalysis(), # Bad
    # LinearDiscriminantAnalysis(), # Bad
    #LogisticRegression(), # Good
]

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

class ClfSwitcher(BaseEstimator):
    # https://stackoverflow.com/questions/48507651/multiple-classification-models-in-a-scikit-pipeline-python

    def __init__(
        self,
        estimator = SGDClassifier(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """

        self.estimator = estimator


    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self


    def predict(self, X, y=None):
        return self.estimator.predict(X)


    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


    def score(self, X, y):
        return self.estimator.score(X, y)

    def coef_(self):
        return self.estimator.coef_

def get_best_classificator_and_test_accuracy(data, labels, estimators):
    param_grid = []
    for classificator in classifiers:
        param_grid.append({'clf__estimator': [classificator]})

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    clf = GridSearchCV(estimator=estimators, param_grid=param_grid, cv=cv) # https://stackoverflow.com/questions/52580023/how-to-get-the-best-estimator-parameters-out-from-pipelined-gridsearch-and-cro

    clf.fit(data, labels)

    acc = clf.best_score_ # Best Test Score
    print("Best Test Score: \n{}\n".format(acc))

    if acc <= 0.25:
        acc = np.nan
    return clf.best_estimator_, acc

def data_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) + 0.00000001 / (max_val - min_val)
    return scaled_data

from collections import defaultdict

def indexes(l, chosen_key):
    _indices = defaultdict(list)
    for index, item in enumerate(l):
        _indices[item].append(index)

    for key, value in _indices.items():
        if key == chosen_key:
            return value


def braincommand_dataset_loader(game_mode: str, subject_id: int):
    complete_information = pd.read_csv(f'assets/game_saved_files/eeg_data_{game_mode}_sub{subject_id:02d}.csv')
    x_list = list(complete_information['time'].apply(eval))
    label = list(complete_information['class'][1:])# TODO: I'm removing the first one because the EEG data is incomplete. Real time seems to have this problem too. So the first one will always be lost

    label_0 = label.count(0)
    print(f"label 0 is {label_0}")

    label_1 = label.count(1)
    print(f"label 1 is {label_1}")

    label_2 = label.count(2)
    print(f"label 2 is {label_2}")

    label_3 = label.count(3)
    print(f"label 3 is {label_3}")

    x_array = np.array(x_list[1:]) # trials, time, channels
    x_array = x_array[:, :, :-9] # The last channels are accelerometer (x3), gyroscope (x3), validity, battery and counter
    x_array = np.transpose(x_array, (0, 2, 1))
    x_array = signal.detrend(x_array)
    #x_array, label = convert_to_epochs(x_array, label)
    x_array = data_normalization(x_array)
    return x_array, label

def class_selection(dataX, dataY, event_dict: dict, selected_classes: list[int]):
    dataX_selected: list = []
    dataY_selected: list = []
    for dataX_idx, dataY_idx in zip(dataX, dataY):
        if dataY_idx in selected_classes:
            dataX_selected.append(dataX_idx)
            dataY_selected.append(dataY_idx)
    dataX_selected_np = np.asarray(dataX_selected)
    dataY_selected_df = pd.Series(dataY_selected)

    label_remap = {dataY_original: dataY_remap_idx for dataY_remap_idx, dataY_original in enumerate(selected_classes)}

    event_dict = {key: label_remap[value] for key, value in event_dict.items()
                  if value in selected_classes}

    return dataX_selected_np, np.asarray(dataY_selected_df.replace(label_remap)), event_dict

def convert_to_epochs(data, labels):
    events = np.column_stack((
        np.arange(0, dataset_info['sample_rate'] * data.shape[0], dataset_info['sample_rate']),
        np.zeros(len(labels), dtype=int),
        np.array(labels),
    ))

    event_dict = {'Arriba': 2, 'Abajo': 3, 'Derecha': 0, 'Izquierda': 1}

    epochs = EpochsArray(data, info=mne.create_info(
        sfreq=dataset_info['sample_rate'], ch_types='eeg', ch_names=dataset_info['channels_names']), events=events,
                         event_id=event_dict)
    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage)

    ar = AutoReject()
    epochs = ar.fit_transform(epochs)
    data = epochs.get_data()
    label = epochs.events[:, -1]

    #data, label, event_dict = class_selection(data, label, event_dict, selected_classes=[0,1,2])
    return data, label

def simple_train(data, labels):
    clf = Pipeline([("Cova", Covariances()), ("ts", TangentSpace()), ('clf', ClfSwitcher())]) # This is probably the best one, at least for Torres
    classifier, acc = get_best_classificator_and_test_accuracy(data, labels, clf)
    return classifier, acc

def BrainCommand_train(game_mode: str, subject_id: int, simple_flag: bool = True) -> None:
    data, labels = braincommand_dataset_loader(game_mode, subject_id)
    if simple_flag:
        classifier, acc = simple_train(data, labels)
    else:
        """
        models_outputs, processing_name, columns_list, transform_methods = group_methods_train(
            "braincommand",
            subject_id,
            methods,
            models_outputs,
            data,
            _,
            labels,
            dataset_info,
        )
        """

    joblib.dump(classifier, f'assets/classifier_data/classifier_{game_mode}_sub{subject_id:02d}.joblib')

    print(f"Classifier saved! {game_mode}: Subject {subject_id:02d}")

if __name__ == "__main__":
    subject_id = 1
    game_mode = 'calibration2'

    BrainCommand_train(game_mode, subject_id)