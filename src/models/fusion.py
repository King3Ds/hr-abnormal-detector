import numpy as np
from sklearn.linear_model import LogisticRegression

class LateFusion:
    def __init__(self):
        self.meta = LogisticRegression(max_iter=200, multi_class='auto')

    def fit(self, P_svm, P_cnn, P_rule, y):
        Z = np.hstack([P_svm, P_cnn, P_rule])
        self.meta.fit(Z, y)

    def predict_proba(self, P_svm, P_cnn, P_rule):
        Z = np.hstack([P_svm, P_cnn, P_rule])
        return self.meta.predict_proba(Z)