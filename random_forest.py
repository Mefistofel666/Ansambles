from typing import List, Union
from sklearn.tree import DecisionTreeRegressor
import numpy as np


class RandomForestRegressor():
    """
        TODO: ADD DOCSTRING!!!
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_leaf: int = 1,
        ) -> None:

        self._subsample_size = None

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.models = []


    def _make_estimator(self):
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )


    def _get_bootstrap_samples(
                                self, 
                                data_train: List[List[float]],
                                y_train: Union[List[float], float]
                            ):
        indices = np.random.randint(0, len(data_train), (self.n_estimators, self._subsample_size))
        samples_train = data_train[indices]
        samples_y = y_train[indices]
        return samples_train, samples_y


    def fit(
            self,
            data_train: List[List[float]],
            y_train: Union[List[float], float]
            ) -> None:

        self._subsample_size = int(data_train.shape[0])
        samples_train, samples_y = self._get_bootstrap_samples(data_train, y_train)

        for i in range(self.n_estimators):
            estimator = self._make_estimator()
            estimator.fit(samples_train[i], samples_y[i].reshape(-1))
            self.models.append(estimator)


    def predict(self, test_data: List[List[float]]) -> List[float]:
        n_samples = test_data.shape[0]
        predictions = []

        for i in range(self.n_estimators):
            model_predict = self.models[i].predict(test_data)
            predictions.append(model_predict)
        
        predictions = np.array(predictions).T

        return np.array([np.mean(predictions[i]) for i in range(n_samples)])


if __name__ == "__main__":
    pass