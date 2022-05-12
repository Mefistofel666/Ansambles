from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
                            

from decision_tree import MyDecisionTreeRegressor
from decision_tree_gain import MyDecisionTreeRegressorGain
from random_forest import RandomForestRegressor
from boosting import GradienBoosting

from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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



if __name__ == "__main__":

    df = pd.read_csv('TSLA.csv')
    df.drop(columns=["Date","Adj Close"], inplace=True)
    X = np.array(df.drop(columns='Volume'))
    y = np.array(df['Volume'])


    MAX_DEPTH = 10
    MIN_SAMPLES_LEAF = 5

    my_decision_tree = MyDecisionTreeRegressor(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    )

    my_decision_tree_gain = MyDecisionTreeRegressorGain(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        lmd=0.5,
        gamma=0.1,
    )

    my_random_forest = RandomForestRegressor(
        n_estimators=20,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF
    )

    my_boosting_gain = GradienBoosting(
        base_estimator = my_decision_tree_gain,
        n_estimators=20,
        learning_rate=0.1,
        loss_function='squared_error',
    )

    my_boosting = GradienBoosting(
        base_estimator = my_decision_tree,
        n_estimators=20,
        learning_rate=0.1,
        loss_function='squared_error',
    )

    algorithms = [
        my_decision_tree,
        my_decision_tree_gain,
        my_random_forest, 
        my_boosting_gain, 
        my_boosting,
        ]

    alg_metrics = [
        get_metrics(
            X,
            y,
            n_folds=5, 
            model=algo, 
            metric=r2_score
            ) for algo in algorithms
        ]
    fig7, ax7 = plt.subplots()
    ax7.set_title('')
    ax7.boxplot(
        alg_metrics,
        labels=[
            'My DT','My DT Gain',
            'My Random Forest',
            'My Boosting Gain',
            'My General Boosting'
            ]
        )
    plt.grid()
    plt.show()