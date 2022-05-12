from typing import List, Union
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from decision_tree import MyDecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import copy


class GradienBoosting:
    """
        TODO: ADD DOCSTRING!!!
    """

    def __init__(
        self,
        *,
        base_estimator: object,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        loss_function: str = 'squared_error'
    ) -> None:

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.__coeffs = []
        self.__estimators = []
    

    def __get_residual(self, current_approx: List[float], y_train: List[float]) -> List[float]:

        if self.loss_function == 'squared_error':
            return (y_train - current_approx)


    def __get_estimator(self, data_train: List[List[float]], y_train: List[float]):

        estimator = copy.copy(self.base_estimator)
        estimator.fit(data_train, y_train)
        return estimator
    

    def fit(self, data_train: List[List[float]], y_train: List[float]) -> None:
        
        current_approx = np.array([0] * data_train.shape[0])

        for _ in range(self.n_estimators):

            residual = self.__get_residual(current_approx, y_train)
            estimator = self.__get_estimator(data_train, residual)
            prediction = estimator.predict(data_train)

            error = (y_train - current_approx)

            numerator = sum([error[i] * prediction[i] for i in range(len(prediction))])
            denominator = sum(prediction**2)
            coeff =  numerator / denominator

            self.__estimators.append(estimator)
            self.__coeffs.append(coeff)

            current_approx = current_approx + coeff * prediction


    def predict(self, data_test) -> List[float]:

        result = []

        for i in range(self.n_estimators):
            estimator_pred = self.__estimators[i].predict(data_test)
            result.append(estimator_pred)
        
        result = np.array(result).T
        res = []
        for row in result:
            pred = sum([self.__coeffs[i] * row[i] for i in range(len(row))])
            res.append(pred)
        return np.array(res)



def get_metrics(X, y, n_folds=2, model=None, metric=mean_squared_error):

    kf = KFold(n_splits=n_folds, shuffle=True)
    kf.get_n_splits(X)

    er_list = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        er_list.append(metric(y_test, predict))
    
    return er_list


if __name__ == '__main__':

    my_dt = MyDecisionTreeRegressor(min_samples_leaf=2, max_depth=4)
    sklearn_dt = DecisionTreeRegressor(min_samples_leaf=2, max_depth=4)
    boosting = GradienBoosting(
        base_estimator=sklearn_dt,
        n_estimators=5,
        learning_rate=0.1,
        )

    df = pd.read_csv('TSLA.csv')
    df.drop(columns=["Date","Adj Close"], inplace=True)
    X = np.array(df.drop(columns='Volume'))
    y = np.array(df['Volume'])

    er_sklearn_dt = get_metrics(X, y, 5, sklearn_dt, r2_score)
    er_my_dt = get_metrics(X, y, 5, my_dt, r2_score)
    er_boosting = get_metrics(X, y, 5, boosting, r2_score)

    data = [er_sklearn_dt, er_my_dt,er_boosting ]
    fig7, ax7 = plt.subplots()
    ax7.set_title('')
    ax7.boxplot(data, labels=['Sklearn DT','My DT', 'Boosting'])
    plt.grid()
    plt.show()
