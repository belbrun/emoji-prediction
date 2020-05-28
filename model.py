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

    def __init__(self, input_size, hidden_size, num_layers, dropout, f_size):
        super().__init__(input_size, hidden_size, num_layers, dropout, f_size)
        self.activation = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2 + f_size, 100)
        self.fc2 = nn.Linear(100, 20)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x, f):
        # preprocess to time first, random initi h0 and c0?
        x = x.permute(1, 0, 2)
        _, (h, _) = self.lstm(x)
        x = torch.cat((h[-2], h[-1], f), dim=1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def train(self, batch):
        self.train()
        self.zero_grad()
        x, y, f = batch
        logits = self(x, f)
        loss = self.criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        return loss.detach().item()

    def evaluate(self, x, f):
        self.eval()
        with torch.no_grad():
            return self(x, f)
