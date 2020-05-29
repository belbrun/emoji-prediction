import data
import pipeline
import sklearn
import numpy as np
import time
import joblib

from sklearn.metrics import classification_report
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

def run_rnn():
    train_data = data.load_data('test')
    train_data = data.add_features(train_data)

    text_field = data.get_text_field(train_data.text)
    label_field = data.get_label_field()

    train_dataset = data.get_dataset(train_data, text_field, label_field)
    train_iter = data.get_iterator(train_dataset, 10)

    model_pipeline = RNNPipeline(150, 1, 0, 7, text_field, 100)
    model_pipeline.train(train_iter)
    #model_pipeline.load_model()

    test_data = data.load_data('trial')
    test_data = data.add_features(test_data)

    test_dataset = data.get_dataset(test_data, text_field, label_field)
    test_iter = data.get_iterator(test_dataset, 32)

    model_pipeline.evaluate(test_iter)



params = {
    'embedding_dim': 100,
    'hidden_size': 200,
    'num_layers': 3,
    'dropout': 0.1,
    'f_size': 0 # set to 0 to use model without additional features
}

batch_size = 100
n_epochs = 10

def train_rnn():
    log = []
    (train_data, valid_data, test_data), text_field = data.get_iterators(batch_size)
    pipe = RNNPipeline(params, text_field)
    for epoch in range(n_epochs):
        loss = pipe.train(train_data)
        y_p, y = pipe.evaluate(valid_data)
        log.append('Epoch {}\nLoss: {}\n{}\n'.format(epoch, loss, classification_report(y, y_p)))
        print(log[-1])
    y_p, y = pipe.evaluate(test_data)
    log.append('Test\n{}\n'.format(classification_report(y, y_p)))
    print(log[-1])
    data.write_log(log)

if __name__ == '__main__':
    #run_rnn()
    train_rnn()
