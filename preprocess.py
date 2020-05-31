import sklearn
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from feature_extraction import make_feature_matrix_baseline
from scipy import sparse

def preprocess_tweet(text):
    new_text = []
    for word in text:
        if word == '@user':
            new_word = '<user>'
        else:
            new_word = re.sub('[^A-Za-z0-9]+', '', word)
        new_word = new_word.strip()
        if len(new_word) > 0:
            new_text.append(new_word.lower())
    return new_text

def preprocess_baseline(text):
    new_text = []
    text = text.split(' ')
    for word in text:
        if word == '@user':
            new_word = ''
        else:
            new_word = re.sub('\b([^A-Za-z0-9]+)\b', '', word)
        new_word = new_word.strip()
        if len(new_word) > 0:
            new_text.append(new_word.lower())
    return new_text

class Preprocess():

    def train(self, X_train, y_train, max_n_gram, add_special, use_tf_idf, do_preprocess):
        self.add_special = add_special
        self.use_tf_idf = use_tf_idf
        self.do_preprocess = do_preprocess

        if do_preprocess:
            X = [' '.join(preprocess_baseline(x)) for x in X_train]
        else:
            X = X_train

        self.counter = CountVectorizer(ngram_range=(1, max_n_gram), lowercase=True).fit(X)
        X = self.counter.transform(X)
        
        if add_special:
            special_features = make_feature_matrix_baseline(X_train)
            X = sparse.hstack([X, sparse.csr_matrix(special_features)])

        if use_tf_idf:
            self.scaler = TfidfTransformer().fit(X)
            X = self.scaler.transform(X)
        print(X.shape)
        return X
    
    def run(self, X_test):
        if self.do_preprocess:
            X = [' '.join(preprocess_baseline(x)) for x in X_test]
        else:
            X = X_test
        
        X = self.counter.transform(X)

        if self.add_special:
            special_features = make_feature_matrix_baseline(X_test)
            X = sparse.hstack([X, sparse.csr_matrix(special_features)])
        if self.use_tf_idf:
            X = self.scaler.transform(X)
        print(X.shape)
        return X