import sklearn
import numpy as np

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

def train(X, y):
    model = OneVsRestClassifier(SVC(C = 0.1, random_state = 1))
    model.fit(X, y)
    return model

def train_transform(X, y):
    model = train(X, y)
    y_pred = model.predict(X)
    return y_pred