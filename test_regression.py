import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from GradientBoostingRegressor import GradientBoostingRegressor
from RandomForestRegressor import RandomForestRegressor
from validation import get_metrics


if __name__ == "__main__":

    df = pd.read_csv('TSLA.csv')
    df.drop(columns=["Date","Adj Close"], inplace=True)

    X = np.array(df.drop(columns='Volume'))
    y = np.array(df['Volume'])

    dt_regressor = DecisionTreeRegressor(
        max_depth=40, 
        min_samples_leaf=0.0077,
        )

    gb_regressor = GradientBoostingRegressor(
        base_estimator=dt_regressor,
        n_estimators=40,
        learning_rate=0.1,
        loss_function_name='squared_error',
    )

    rf_regressor = RandomForestRegressor(
        base_estimator=dt_regressor,
        n_estimators=40
    )

   

    models = [dt_regressor, gb_regressor, rf_regressor]

    metrics = [
        get_metrics(
            X,
            y,
            n_folds=10, 
            model=model, 
            metric=r2_score
            ) for model in models
        ]
    fig7, ax7 = plt.subplots()
    ax7.set_title('Comparison of Algorithms')
    colors = ['#78C850', '#F08030', '#6890F0', ]
    error_df = pd.DataFrame({
        'Sklearn DT': metrics[0],
        'GBRegressor': metrics[1],
        'RFRegressor': metrics[2],
        })
    sns.boxplot(
        x="variable", 
        y="value",
        palette=colors, 
        data=pd.melt(error_df))
    plt.show()
