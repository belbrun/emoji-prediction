import data
import pipeline
import sklearn
import numpy as np
import time
import joblib
import torch

from scipy.stats import t
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from pipeline import BaselinePipeline, RNNPipeline
from sklearn.model_selection import train_test_split

def select_n(X_train, y_train, n):
    N = len(X_train)
    p = n/N

    select = np.random.choice([0, 1], size=N, p=[1-p, p])
    X_train = X_train[select==1]
    y_train = y_train[select==1]

    return X_train, y_train

def train_baseline():
    X_train, y_train = data.load_raw_data('train', True)
    X_test, y_test = data.load_raw_data('test', False)
    X_trial, y_trial = data.load_raw_data('trial', False)

    print(len(X_train))
    print(len(X_test))
    print(len(X_trial))
    
    baseline_pipeline = BaselinePipeline(C=10)
    baseline_pipeline.train(X_train, y_train, max_n_gram=1, add_special=False, use_tf_idf=True, do_preprocess=True)
    #baseline_pipeline.load_preprocess('lr+.sav')
    #try:
    #    baseline_pipeline.save('lr_temp.sav', 'lr_prep_temp.sav')
    #except:
    #    print('Error while trying to save the model...')

    y_pred = baseline_pipeline.run(X_test)
    print(classification_report(y_test, y_pred))

    y_pred = baseline_pipeline.run(X_trial)
    print(classification_report(y_trial, y_pred))

params = {
    'embedding_dim': 100,
    'hidden_size': 150,
    'num_layers': 3,
    'dropout': 0.1,
    'f_size': 0 # set to 0 to use model without additional features
}

batch_size = 32
n_epochs = 25

def tune_baseline(X_train, X_test, X_valid, y_train, y_valid, add_special):
    best_c, best_f = 0, 0
    for c in [0.1, 1, 10]:
        baseline_pipeline = BaselinePipeline(C=c)
        baseline_pipeline.train(X_train, y_train, max_n_gram=1, add_special=add_special, use_tf_idf=False, do_preprocess=True)
        y_pred = baseline_pipeline.run(X_valid)
        
        f = f1_score(y_valid, y_pred, average='macro')
        if f > best_f:
            best_c = c
            best_f = f
    
    baseline_pipeline = BaselinePipeline(C=best_c)
    baseline_pipeline.train(X_train, y_train, max_n_gram=1, add_special=False, use_tf_idf=False, do_preprocess=True)
    
    y_pred = baseline_pipeline.run(X_test)
    return y_pred

def signifficance_test_baseline():

    X_train, y_train = data.load_raw_data('train', True)
    X_test, y_test = data.load_raw_data('test', False)
    X_valid, y_valid = data.load_raw_data('trial', False)

    X = np.concatenate((X_train, X_valid))
    y = np.concatenate((y_train, y_valid))

    base = []
    base_plus = []
    for random_state in [42, 193, 546]:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=random_state, shuffle=True)

        y_pred = tune_baseline(X_train, X_test, X_valid, y_train, y_valid, False)
        base.append(f1_score(y_test, y_pred, average='macro'))

        y_pred = tune_baseline(X_train, X_test, X_valid, y_train, y_valid, True)
        base_plus.append(f1_score(y_test, y_pred, average='macro'))
        
        print(base)
        print(base_plus)

    base = np.array(base)
    base_plus = np.array(base_plus)

    diff = [y - x for y, x in zip(base, base_plus)]

    d_bar = np.mean(diff)
    sigma2 = np.var(diff)
    n1 = int(len(y_train)*0.8)
    n2 = len(y_test)
    n = len(y)
    sigma2_mod = sigma2 * (1/n + n2/n1)
    t_statistic =  d_bar / np.sqrt(sigma2_mod)

    p_value = ((1 - t.cdf(t_statistic, n-1))*200)
    print('p-value:', p_value)
    
    return base, base_plus, p_value

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
    log.append('Test\n{}\n{}\n{}\n'.\
               format(epoch, loss, classification_report(y, y_p),
                      confusion_matrix(y, y_p)))
    print(log[-1])
    data.write_log(log)

seed = 12345
if __name__ == '__main__':
    np.random.seed(seed)
    torch.manual_seed(seed)
    #train_rnn()
    train_baseline()
    #signifficance_test_baseline()
