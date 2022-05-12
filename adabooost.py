import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoost():

    def __init__(self, T=5, max_depth=10, min_size=9):
        self.T = T # number of basics algorithms
        self.max_depth = max_depth
        self.min_size = min_size
        self.models = [DecisionTreeClassifier(
                            max_depth=self.max_depth,
                            min_samples_leaf=self.min_size
                            ) for i in range(T)]


    def fit(self, X, y):
        self.alphas = []
        self.weights = np.array([1 / X.shape[0] for _ in range(X.shape[0])])
        for i in range(self.T):
            self.models[i].fit(X, y, sample_weight=self.weights)
            pred = self.models[i].predict(X)
            N = np.sum([self.weights[j] if pred[j] != y[j] else 0 for j in range(X.shape[0])])
            alpha_i = (np.log((1-N)/N)) / 2.0
            self.alphas.append(alpha_i)
            self.weights = np.array([self.weights[j] * np.exp(-alpha_i * y[j] * pred[j]) for j in range(X.shape[0])])
            w0 = np.sum(self.weights)
            self.weights = self.weights / w0


    def predict(self, X):
        pred = []
        for idx, model in enumerate(self.models):
            pred.append(model.predict(X))
        pred = np.array(pred).T
        return [1 if np.dot(pred[i], self.alphas) > 0 else -1 for i in range(pred.shape[0])]