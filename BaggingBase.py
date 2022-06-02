from abc import ABC, abstractmethod
from copy import copy
from typing import List, Tuple, Union

import numpy as np


class Bagging(ABC):
    """
        TODO: ADD DOCSTRING
    """

    def __init__(
        self,
        *,
        base_estimator: object,
        n_estimators: int = 20,

    ) -> None:

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

        self._subsample_size : int = None
        self._estimators: List[object] = []

    
    def _get_bootstrap_samples(
        self, 
        data_train: List[List[float]],
        y_train: Union[List[float], float]
        ) -> Tuple[List[List[float]], List[float]]:

        indices = np.random.randint(0, len(data_train), (self.n_estimators, self._subsample_size))
        samples_train = data_train[indices]
        samples_y = y_train[indices]

        return samples_train, samples_y

        
    def _get_estimator(self, data_train: List[List[float]], y_train: List[float]) -> object:

        estimator = copy(self.base_estimator)
        estimator.fit(data_train, y_train)
        return estimator

    @abstractmethod
    def fit(
        self,
        data_train: List[List[float]],
        y_train: Union[List[float], float]
        ) -> None:

        self._estimators = []
        self._subsample_size = int(data_train.shape[0])
        samples_train, samples_y = self._get_bootstrap_samples(data_train, y_train)

        for i in range(self.n_estimators):
            estimator = self._get_estimator(samples_train[i], samples_y[i].reshape(-1))
            self._estimators.append(estimator)

    @abstractmethod
    def predict(self, test_data: List[List[float]]) -> List[float]:

        raise NotImplementedError
        
    