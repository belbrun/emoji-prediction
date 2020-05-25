import sklearn
import numpy as np

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

class Model():

    def __init__(self):
        pass

    def train(self):
        pass

    def output(self):
        pass

class Baseline(Model):

    def __init__(self, C):
        self.svm = LinearSVC(C=C, random_state=1, multi_class='ovr', class_weight='balanced')

    def train(self, X, y):
        self.svm.fit(X, y)

    def output(self, X):
        return self.svm.predict(X)
