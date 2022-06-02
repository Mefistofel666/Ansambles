from BaggingBase import Bagging
from typing import List, Union
import numpy as np



class RandomForestRegressor(Bagging):
    
    def fit(self, data_train: List[List[float]], y_train: List[float]) -> None:
        return super().fit(data_train, y_train)
    
    
    def predict(self, test_data: List[List[float]]) -> List[float]:
        n_samples = test_data.shape[0]
        predictions = []

        for i in range(self.n_estimators):
            model_predict = self._estimators[i].predict(test_data)
            predictions.append(model_predict)
        
        predictions = np.array(predictions).T
        return np.array([round(np.mean(predictions[i])) for i in range(n_samples)])
    
