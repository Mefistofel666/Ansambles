import numpy as np
from typing import List
from copy import copy
from abc import ABC, abstractmethod

from sklearn.tree import DecisionTreeRegressor


class GradienBoostingBase(ABC):

    def __init__(
        self,
        *,
        base_estimator: object,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        loss_function_name: str
    ) -> None:

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss_function_name = loss_function_name
        
        self._estimators: List[object] = []
    

    @abstractmethod
    def _get_residual(self, current_approx: List[float], y_train: List[float]) -> List[float]:
        raise NotImplementedError


    def _get_estimator(self, data_train: List[List[float]], y_train: List[float]) -> object:
        estimator = copy(self.base_estimator)
        estimator.fit(data_train, y_train)
        return estimator


    @abstractmethod
    def fit(self, data_train: List[List[float]], y_train: List[float]) -> None:
        raise NotImplementedError


    @abstractmethod
    def predict(self, data_test: List[List[float]]) -> List[float]:
        raise NotImplementedError


