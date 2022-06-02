from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from validation import get_metrics
from GradientBoostingBase import GradienBoostingBase



class GradientBoostingRegressor(GradienBoostingBase):
    """
        TODO: ADD DOCSTRING
    """
    
    def _get_residual(self, current_approx: List[float], y_train: List[float]) -> List[float]:

        if self.loss_function_name == 'squared_error':
            return  (y_train - current_approx)
        
        
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

        pred = np.ones([data_test.shape[0]]) * np.mean(self.y)

        for t in range(self.n_estimators):
            pred += self.learning_rate * self._estimators[t].predict(data_test).reshape([data_test.shape[0]])
            
        return pred


if __name__ == "__main__":

    df = pd.read_csv('TSLA.csv')
    df.drop(columns=["Date","Adj Close"], inplace=True)

    X = np.array(df.drop(columns='Volume'))
    y = np.array(df['Volume'])

    dt_regressor = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=40, 
        min_samples_leaf=0.0077,
        )

    gb_regressor = GradientBoostingRegressor(
        base_estimator=dt_regressor,
        n_estimators=50,
        learning_rate=0.1,
        loss_function_name='squared_error',
    )

    

    models = [dt_regressor, gb_regressor]

    metrics = [
        get_metrics(
            X,
            y,
            n_folds=3, 
            model=model, 
            metric=r2_score,
            ) for model in models
        ]
    
    fig7, ax7 = plt.subplots()
    ax7.set_title('Comparison Algorithms')
    colors = ['#78C850', '#F08030']
    error_df = pd.DataFrame({
        'Decision Tree': metrics[0],

        'GBRegressor': metrics[1],
        })
    sns.boxplot(
        x="variable", 
        y="value",
        palette=colors, 
        data=pd.melt(error_df))
    plt.show()