
from data_utils import get_best_classificator_and_test_accuracy, ClfSwitcher
import joblib
from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import numpy as np

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

def load_data(dataset_info: dict, subject_id: int):
    # todo: get a way to save the eeg data with events and everything, matlab? from that you can create the dataloader
    return data, labels

def simple_train(data, labels):
    clf = Pipeline([("Cova", Covariances()), ("ts", TangentSpace()), ('clf', ClfSwitcher())]) # This is probably the best one, at least for Torres
    classifier, acc = get_best_classificator_and_test_accuracy(data, labels, clf)
    return classifier, acc



if __name__ == "__main__":
    dataset_info = { # BrainCommand
                    'dataset_name': 'BrainCommand',
                    '#_class': 4,
                    "target_names": ['Derecha', 'Izquierda', 'Arriba', 'Abajo'],
                    '#_channels': 8, # PENDING
                    'samples': 700, # PENDING
                    'sample_rate': 500, # PENDING
                    'channels_names': ['PENDING'], # PENDING
                    'subjects': 0, # PENDING
                    'total_trials':0, # PENDING
                }

    subject_id = 0
    game_mode = 'calibration1'
    print(subject_id)
    data, labels = load_data(dataset_info, subject_id)
    classifier, acc = simple_train(data, labels)
    joblib.dump(classifier, f'assets/classifier_data/classifier_{game_mode}_sub{subject_id:02d}.pkl')

