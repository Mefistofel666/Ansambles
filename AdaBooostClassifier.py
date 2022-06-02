from copy import copy
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import List


class AdaBoostClassifier():

    def __init__(
        self,
        *,
        n_estimators: int = 5,
        base_estimator: object = None
        ) -> None:

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

        self._estimators: List[object] = []
        self._coeffs: List[float] = []
        self._weights: List[float] = []


    def _get_estimator(self, data_train: List[List[float]], y_train: List[float]):
        estimator = copy(self.base_estimator)
        estimator.fit(data_train, y_train, sample_weight=self._weights)
        return estimator


    def fit(self, data_train: List[List[float]], y_train : List[int]):
        self._estimators = []
        self._coeffs = []
        self._weights = np.array([1 / data_train.shape[0] for _ in range(data_train.shape[0])])

        for _ in range(self.n_estimators):

            estimator = self._get_estimator(data_train, y_train)
            self._estimators.append(estimator)
            prediction = estimator.predict(data_train)

            N = np.sum([self._weights[j] if prediction[j] != y_train[j] else 0 for j in range(data_train.shape[0])])
            coeff = (np.log((1-N)/N)) / 2.0
            self._coeffs.append(coeff)

            self._weights = np.array([self._weights[j] * np.exp(-coeff * y_train[j] * prediction[j]) for j in range(data_train.shape[0])])
            w0 = np.sum(self._weights)
            self._weights = self._weights / w0


    def predict(self, data_test: List[List[float]]):
        pred = []
        for estimator in self._estimators:
            pred.append(estimator.predict(data_test))

        pred = np.array(pred).T
        
        return [1 if np.dot(pred[i], self._coeffs) > 0 else -1 for i in range(pred.shape[0])]