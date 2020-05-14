import sklearn
import numpy as np

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

def train(X, y):
    model = SVC(C=0.1, random_state=1, decision_function_shape='ovr')
    model.fit(X, y)
    return model

def train_transform(X, y):
    model = train(X, y)
    y_pred = model.predict(X)
    return y_pred

class Model():

    def __init__(self):
        pass

    def train(self):
        pass

    def output(self):
        pass

class Baseline(Model):

    def __init__(self, C):
        self.svm = SVC(C=C, random_state=1, decision_function_shape='ovr')

    def train(self, X, y):
        self.svm.fit(X, y)

    def output(self, X):
        return self.svm.predict(X)
