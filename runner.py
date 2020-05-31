import data
import pipeline
import sklearn
import numpy as np
import time
import joblib
import torch

from sklearn.metrics import classification_report, confusion_matrix
from pipeline import BaselinePipeline, RNNPipeline

def select_n(X_train, y_train, n):
    N = len(X_train)
    p = n/N

    select = np.random.choice([0, 1], size=N, p=[1-p, p])
    X_train = X_train[select==1]
    y_train = y_train[select==1]

    return X_train, y_train

def train_baseline():
    k = 100000

    X_train, y_train = data.load_data('train')
    X_test, y_test = data.load_data('test')

    baseline_pipeline = BaselinePipeline(C=0.1, k=k)
    baseline_pipeline.train(X_train, y_train)

    try:
        baseline_pipeline.save('model.sav', 'preprocess.sav')
    except:
        print('Error while trying to save the model...')

    y_pred = baseline_pipeline.run(X_test)
    print(classification_report(y_test, y_pred))

params = {
    'embedding_dim': 100,
    'hidden_size': 200,
    'num_layers': 3,
    'dropout': 0,
    'f_size': 8 # set to 0 to use model without additional features
}

batch_size = 100
n_epochs = 20

def train_rnn():
    log = []
    (train_data, valid_data, test_data), text_field = \
                data.get_iterators(batch_size, params['embedding_dim'])
    pipe = RNNPipeline(params, text_field)
    for epoch in range(n_epochs):
        loss = pipe.train(train_data)
        y_p, y = pipe.evaluate(valid_data)
        log.append('Epoch {}\nLoss: {}\n{}\n{}\n'.\
                   format(epoch, loss, classification_report(y, y_p),
                          confusion_matrix(y, y_p)))
        print(log[-1])
        y_p, y = pipe.evaluate(test_data)
        log.append('Test {}\nLoss: {}\n{}\n{}\n'.\
                   format(epoch, loss, classification_report(y, y_p),
                          confusion_matrix(y, y_p)))
        print(log[-1])
    y_p, y = pipe.evaluate(test_data)
    log.append('Test\n{}\n{}\n{}\n'.\
               format(epoch, loss, classification_report(y, y_p),
                      confusion_matrix(y, y_p)))
    print(log[-1])
    data.write_log(log)

seed = 12345
if __name__ == '__main__':
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_rnn()
