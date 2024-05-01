
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
    print("Best Test Score: \n{}\n".format(clf.best_score_))

    if acc <= 0.25:
        acc = np.nan
    return clf.best_estimator_, acc

def data_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

from collections import defaultdict

def indexes(l, chosen_key):
    _indices = defaultdict(list)
    for index, item in enumerate(l):
        _indices[item].append(index)

    for key, value in _indices.items():
        if key == chosen_key:
            return value

def remove_too_different_trials_v2(data: np.array, label) -> np.array:
    data_cleaned = []
    label_cleaned = []
    for word_index in range(len(dataset_info["target_names"])):
        indexes_from_word = indexes(label, word_index)
        for i_index in indexes_from_word:
            trial = data[i_index,:,:]
            trial_label = label[i_index]
            if np.mean(np.abs(trial)) < np.mean(np.abs(data[indexes_from_word,:,:]))*1.5 :
                data_cleaned.append(trial)
                label_cleaned.append(trial_label)
    return np.array(data_cleaned), label_cleaned

def braincommand_dataset_loader(game_mode: str, subject_id: int):
    complete_information = pd.read_csv(f'assets/game_saved_files/eeg_data_{game_mode}_sub{subject_id:02d}.csv')
    x_list = list(complete_information['time'].apply(eval))
    label = list(complete_information['class'][1:])# TODO: I'm removing the first one because the EEG data is incomplete. Real time seems to have this problem too. So the first one will always be lost
    x_array = np.array(x_list[1:]) # trials, time, channels
    x_array = x_array[:, 50:, :-9] # The last channels are accelerometer (x3), gyroscope (x3), validity, battery and counter
    x_array = np.transpose(x_array, (0, 2, 1))
    x_array = signal.detrend(x_array)
    x_array, label = remove_too_different_trials_v2(x_array, label)
    x_array = data_normalization(x_array)
    return x_array, label

def simple_train(data, labels):
    clf = Pipeline([("Cova", Covariances()), ("ts", TangentSpace()), ('clf', ClfSwitcher())]) # This is probably the best one, at least for Torres
    classifier, acc = get_best_classificator_and_test_accuracy(data, labels, clf)
    return classifier, acc

def BrainCommand_train(game_mode: str, subject_id: int) -> None:
    data, labels = braincommand_dataset_loader(game_mode, subject_id)
    classifier, acc = simple_train(data, labels)
    joblib.dump(classifier, f'assets/classifier_data/classifier_{game_mode}_sub{subject_id:02d}.joblib')

    print(f"Classifier saved! {game_mode}: Subject {subject_id:02d}")

if __name__ == "__main__":
    dataset_info = {  # BrainCommand
                    'dataset_name': 'BrainCommand',
                    '#_class': 4,
                    "target_names": ['Derecha', 'Izquierda', 'Arriba', 'Abajo'],
                    '#_channels': 8,
                    'samples': 350 - 50,  # 250*1.4
                    'sample_rate': 250,
                    'channels_names': ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'],  # This is from the original Unicorn cap
                    'subjects': 0,  # PENDING
                    'total_trials': 0,  # VARIABLE SINCE THE SUBJECT HAS TOTAL CONTROL OF HOW MANY MOVEMENTS THEY WANT TO DO
                }

    subject_id = 0
    game_mode = 'calibration1'

    BrainCommand_train(game_mode, subject_id)