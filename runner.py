import data
import pipeline
import sklearn
import numpy as np
import time

from sklearn.metrics import classification_report
from pipeline import BaselinePipeline, RNNPipeline

def select_n(X_train, y_train, n):
    N = len(X_train)
    p = n/N

    select = np.random.choice([0, 1], size=N, p=[1-p, p])
    X_train = X_train[select==1]
    y_train = y_train[select==1]

    return X_train, y_trai

def train_baseline():
    k = 100000

    X_train, y_train = data.load_data('train')

    start_time = time.time()
    baseline_pipeline = BaselinePipeline(C=0.1, k=k)
    baseline_pipeline.train(X_train, y_train)

    try:
        baseline_pipeline.save('model.sav', 'preprocess.sav')
    except:
        print('Error while trying to save the model...')

    X_test, y_test = data.load_data('test')
    y_pred = baseline_pipeline.run(X_test)

    end_time = time.time()
    print('Elapsed time:', end_time - start_time)

    print(classification_report(y_test, y_pred))

params = {
    'input_size': 100,
    'hidden_size': 200,
    'num_layers': 3,
    'dropout': 0.1,
    'f_size': 0
}

batch_size = 10
n_epochs = 10

def train_rnn():
    train_data, valid_data, test_data = data.get_iterators(batch_size)
    pipe = RNNPipeline(params)
    for epoch in range(n_epochs):
        loss = pipe.train(train_data)
        y_p, y = pipe.evaluate(valid_data)
        print('Epoch {}\n{}\n'.format(epoch, classification_report(y, y_p)))
    y_p, y = pipe.evaluate(test_data)
    print('Test\n{}\n'.format(classification_report(y, y_p)))


if __name__ == '__main__':
    train_rnn()
