from BaggingBase import Bagging
from typing import List, Union
import numpy as np
from sklearn.tree import DecisionTreeClassifier



class RandomForestClassifier(Bagging):

    def __init__(
        self,
        *,
        max_depth: int = None,
        min_samples_leaf: int = 1,
        n_estimators: int = 20
        ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        self.base_estimator = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf
        )

        super().__init__(
            base_estimator=self.base_estimator,
            n_estimators=self.n_estimators
        )

    
    def fit(self, data_train: List[List[float]], y_train: List[float]) -> None:
        return super().fit(data_train, y_train)
    

    @staticmethod
    def mode(lst: List[int]) -> int:
        lst = lst.tolist()
        return max((lst.count(item), item) for item in set(lst))[1]
        

    def predict(self, test_data: List[List[float]]) -> List[float]:
        n_samples = test_data.shape[0]
        predictions = []

        for i in range(self.n_estimators):
            model_predict = self._estimators[i].predict(test_data)
            predictions.append(model_predict)
        


        predictions = np.array(predictions).T
        return np.array([self.mode(predictions[i]) for i in range(n_samples)])
    
