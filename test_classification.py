import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from AdaBooostClassifier import AdaBoostClassifier
from RandomForestClassifier import RandomForestClassifier
from validation import get_metrics
from GradientBoostingClassifier import GradientBoostingClassifier

if __name__ == "__main__":

    df = pd.read_csv('ionosphere_data_kaggle.csv')

    df['label'].replace('b', -1, inplace = True)
    df['label'].replace('g', 1, inplace = True) 

    X = np.array(df.drop(columns=['label']))
    y = np.array(df['label'])
    
    dt_regressor = DecisionTreeRegressor(
        max_depth=20,
        min_samples_leaf=5
    )

    dt_classifier = DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=5,
    )

    adaboost = AdaBoostClassifier(
        n_estimators=20,
        base_estimator=dt_classifier,
    )

    rf_classifier = RandomForestClassifier(
        n_estimators=10,
        max_depth=None, 
        min_samples_leaf=5
    )


    models = [dt_classifier, adaboost, rf_classifier]

    metrics = [
        get_metrics(
            X,
            y,
            n_folds=10, 
            model=model, 
            metric=accuracy_score
            ) for model in models
        ]
    fig7, ax7 = plt.subplots()
    ax7.set_title('Comparison of Algorithms')
    colors = ['#78C850', '#F08030', '#6890F0']
    error_df = pd.DataFrame({
        'DTClassifier': metrics[0],
        'AdaBoostClassifier': metrics[1],
        'RFClassifier': metrics[2],
        })
    sns.boxplot(
        x="variable", 
        y="value",
        palette=colors, 
        data=pd.melt(error_df))
    plt.show()
