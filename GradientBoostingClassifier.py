from typing import List
import numpy as np

from GradientBoostingBase import GradienBoostingBase



class GradientBoostingClassifier(GradienBoostingBase):
    """
        TODO: ADD DOCSTRING
    """
    
    @staticmethod
    def mode(lst: List[int]) -> int:
        lst = lst.tolist()
        return max((lst.count(item), item) for item in set(lst))[1]

    
    def _get_residual(self, current_approx: List[float], y_train: List[float]) -> List[float]:

        if self.loss_function_name == 'exponential':
            print(-y_train * np.exp(-current_approx * y_train))
            return -y_train * np.exp(-current_approx * y_train)
        
        
    def fit(self, data_train: List[List[float]], y_train: List[float]) -> None:

        self.y = y_train
        current_approx = np.array([np.mean(y_train)] * data_train.shape[0])
        self._estimators = []

        for i in range(self.n_estimators):

            if i == 0:
                residual = y_train
            else:
                residual = self._get_residual(current_approx, y_train)

            estimator = self._get_estimator(data_train, residual)
            prediction = estimator.predict(data_train)
        
            self._estimators.append(estimator)

            current_approx += self.learning_rate * prediction


    def predict(self, data_test: List[List[float]]) -> List[float]:

        pred = np.ones([data_test.shape[0]]) 


        for t in range(self.n_estimators):
            pred += self.learning_rate * self._estimators[t].predict(data_test).reshape([data_test.shape[0]])
            
        return [1 if pred[i] > 0 else -1 for i in range(pred.shape[0])]