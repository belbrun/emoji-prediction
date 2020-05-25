import sklearn
import numpy as np
import torch.nn as nn

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


class RNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss()
        self.lstm = nn.LSTM(input_size=200, hidden_size=300, num_layers=2,
                            dropout=0, bidirectional=True)
        self.fc1 = nn.Linear(305, 100)
        self.fc2 = nn.Linear(100, 20)

    def forward(self, x, data):
        #preprocess to time first, random initi h0 and c0?
        _, (h, _) = self.lstm(x)
        x = torch.cat((x, data), dim=1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
