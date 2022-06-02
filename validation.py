from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

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
