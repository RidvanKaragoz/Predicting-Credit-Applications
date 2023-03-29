import pandas as pd
import numpy as np


class Baseline:
    def __init__(self, feature="bernoulli_p"):
        self.feature = feature

    def fit(self, X, y):
        pass

    def predict(self, X):
        y_pred = X[self.feature].values

        return y_pred > 0.5

    def predict_proba(self, X):
        y_pred = pd.concat([1 - X[self.feature], X[self.feature]], axis=1).values

        return np.clip(y_pred, 0, 1)
