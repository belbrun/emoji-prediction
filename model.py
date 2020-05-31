import sklearn
import numpy as np
import torch.nn as nn
import torch
import data

from sklearn.linear_model import LogisticRegression


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model():

    def __init__(self):
        pass

    def train(self):
        pass

    def output(self):
        pass

class Baseline(Model):

    def __init__(self, C):
        self.lr = LogisticRegression(C=C, random_state=1, multi_class='ovr', class_weight='balanced', n_jobs=-1, max_iter=10000)

    def train(self, X, y):
        self.lr.fit(X, y)

    def output(self, X):
        return self.lr.predict(X)


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, f_size):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.f_size = f_size
        self.activation = nn.ReLU()
        self.eval_criterion =  nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(
            data.class_weights()).to(device))
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2 + f_size, 200)
        self.fc2 = nn.Linear(200, 20)
        #self.fc3 = nn.Linear(100, 20)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x, f):
        # preprocess to time first, random initi h0 and c0?
        batch_size = x.size()[0]
        x = x.permute(1, 0, 2)

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).requires_grad_().to(device)

        _, (h, _) = self.lstm(x, (h0, c0))

        hidden = h.view(self.num_layers, 2, batch_size, self.hidden_size)
        last_hidden = hidden[-1]
        if self.f_size == 0:
            x = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        else:
            x = torch.cat((last_hidden[0], last_hidden[1], f.float()), dim=1)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        #x = self.activation(x)
        #x = self.fc3(x)

        return x

    def train_model(self, batch):
        #if the method is called train, the self.train() does not work
        self.train()
        self.zero_grad()

        x, y, f = batch
        logits = self(x.to(device), f.to(device))
        #print(logits[2], torch.argmax(logits, dim=1)[2], y)
        loss = self.criterion(logits, y.to(device))
        loss.backward()

    #    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.05)
        self.optimizer.step()

        return loss.to(device='cpu').detach().item()

    def evaluate(self, x, f):
        self.eval()
        with torch.no_grad():
            return self(x.to(device), f.to(device)).to(device='cpu')
