import numpy as np
from typing import List, Tuple
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd


def mse(y_true: List[float], y_pred: List[float]) -> float:
    return np.mean((y_true - y_pred)**2)


def mae(y_true: List[float], y_pred: List[float]) -> float:
    return np.mean(abs(y_true - y_pred))


class MyDecisionTreeRegressorGain():
    """
        TODO: ADD DOCSTRING!!!
    """


    def __init__(
        self,
        max_depth: int = None,
        min_samples_leaf: int = 2,
        criterion: str = 'squared_error',
        lmd: float = 1.0,
        gamma: float = 0.1, 
        ) -> None:

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.value = None
        self.feature_idx = -1
        self.feature_threshold = 0
        self.lmd = lmd
        self.gamma = gamma
        self.left = None
        self.right = None


    def __get_optimal_value(self, data: List[float]) -> float:

        if self.criterion == 'squared_error':
            return np.mean(data)

        elif self.criterion == 'mae':
            return np.median(data)

        else:
            raise ValueError(f'In {self.__class__.__name__}(...) criterion may be "mse" or "mae",'
                            f'not "{self.criterion}"')


    def __get_error(self, y_true: List[float], y_pred: List[float]) -> float:
        
        if self.criterion == 'squared_error':
            return mse(y_true, y_pred)

        elif self.criterion == 'mae':
            return mae(y_true, y_pred)
        
        else:
            raise ValueError(f'In {self.__class__.__name__}(...) criterion may be "mse" or "mae",'
                            f'not "{self.criterion}"')
    

    def __get_error_first_der(self, y_true: List[float], y_pred: List[float]) -> float:

        if self.criterion == 'squared_error':
            return y_pred - y_true

        elif self.criterion == 'mae':
            return np.array(list(map(
                lambda diff : 0 if diff == 0 else 1 if diff > 0 else -1,
                list(y_true - y_pred)
                )))
        
        else:
            raise ValueError(f'In {self.__class__.__name__}(...) criterion may be "mse" or "mae",'
                            f'not "{self.criterion}"')
    

    def __get_best_partition(self, data_train: List[List[float]], y_train: List[float]) -> Tuple[int, float]:

        head_inp_size = data_train.shape[0]
        feature_idx, feature_threshold = -1, 0
        head_error = self.__get_error(y_train, np.array([self.value]*head_inp_size)) * head_inp_size
        best_gain = -self.gamma

        for feature in range(data_train.shape[1]):

            idxs = np.argsort(data_train[:, feature])
            threshold = 1
            left_size, right_size = head_inp_size, 0

            gain_left, gain_right = sum(y_train), 0.0


            while threshold < head_inp_size - 1:

                left_size -= 1
                right_size += 1

                gain_left -= y_train[idxs[threshold]]
                gain_right += y_train[idxs[threshold]]

                gain = gain_left**2 / (left_size + self.lmd) + gain_right**2 / (right_size + self.lmd)
                gain -= (gain_left+gain_right)**2 / (left_size+right_size + self.lmd) + self.gamma  

                if gain > best_gain:
                    if min(left_size, right_size) > self.min_samples_leaf:
                        best_gain = gain
                        feature_idx = feature
                        feature_threshold = data_train[idxs[threshold], feature]

                threshold += 1
        return feature_idx, feature_threshold, best_gain
        

    def __create_childs(self,) -> None:

        self.left = MyDecisionTreeRegressorGain(
            max_depth=self.max_depth-1,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            lmd=self.lmd,
            gamma=self.gamma,
            )

        self.right = MyDecisionTreeRegressorGain(
            max_depth=self.max_depth-1,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            lmd=self.lmd,
            gamma=self.gamma,
            )
    

    def __fit_childs(self, data_train: List[List[float]], y_train: List[float]) -> None:

        idxs_l = (data_train[:, self.feature_idx] > self.feature_threshold)
        idxs_r = (data_train[:, self.feature_idx] <= self.feature_threshold)
        
        self.left.fit(data_train[idxs_l, :], y_train[idxs_l])
        self.right.fit(data_train[idxs_r, :], y_train[idxs_r])


    def fit(self, data_train: List[List[float]], y_train: List[float]) -> None:
        
        self.value = self.__get_optimal_value(y_train)

        if self.max_depth and self.max_depth <= 1:
            return

        self.feature_idx, self.feature_threshold, self.gain = self.__get_best_partition(data_train, y_train)

        if self.feature_idx == -1:
            return
        
        self.__create_childs()
        self.__fit_childs(data_train, y_train)

        if (self.left.left == None or self.right.left == None):
            if self.gain < 0.0:
                self.left = None
                self.right = None
                self.feature_idx = -1
    

    def __predict(self, test_point: List[float]) -> float:

        if self.feature_idx == -1:
            return self.value

        if test_point[self.feature_idx] > self.feature_threshold:
            return self.left.__predict(test_point)

        return self.right.__predict(test_point)


    def predict(self, data_test: List[List[float]]) -> List[float]:

        y = np.zeros(data_test.shape[0])

        for i in range(data_test.shape[0]):
            y[i] = self.__predict(data_test[i])

        return y


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

    my_dt = MyDecisionTreeRegressorGain(min_samples_leaf=2, max_depth=4)
    sklearn_dt = DecisionTreeRegressor(min_samples_leaf=2, max_depth=4)

    df = pd.read_csv('TSLA.csv')
    df.drop(columns=["Date","Adj Close"], inplace=True)
    X = np.array(df.drop(columns='Volume'))
    y = np.array(df['Volume'])

    er_sklearn_dt = get_metrics(X, y, 10, sklearn_dt, r2_score)
    er_my_dt = get_metrics(X, y, 10, my_dt, r2_score)

    data = [er_sklearn_dt, er_my_dt, ]
    fig7, ax7 = plt.subplots()
    ax7.set_title('')
    ax7.boxplot(data, labels=['Sklearn DT','My DT'])
    plt.grid()
    plt.show()
