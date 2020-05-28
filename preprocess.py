import sklearn
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2

def preprocess_tweet(text):
    new_text = []
    for word in text:
        new_word = re.sub('[#@\.\!\?]+', '', word)
        new_word = re.sub(r'([a-z])\1{3,}', r'\1', new_word) 
        new_word = new_word.strip()
        if len(new_word) > 0:
            new_text.append(new_word.lower())
    return new_text

class Preprocess():

    def __init__(self, k):
        self.k = k

    def train(self, X_train, y_train, max_n_gram):
        self.counter = CountVectorizer(ngram_range=(1, max_n_gram),
                              lowercase=True,
                              min_df=2,
                              stop_words='english').fit(X_train)

        ngrams_train = self.counter.transform(X_train)
        self.scaler = TfidfTransformer().fit(ngrams_train)

        scaled_ngrams_train = self.scaler.transform(ngrams_train)
        self.select = SelectKBest(score_func=chi2, k=self.k).fit(scaled_ngrams_train, y_train)

        return self.select.transform(scaled_ngrams_train)

    def run(self, X):
        ngrams = self.counter.transform(X)
        ngrams = self.scaler.transform(ngrams)
        ngrams = self.select.transform(ngrams)

        return ngrams
