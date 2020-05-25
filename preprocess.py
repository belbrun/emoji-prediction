import sklearn
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2

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
       

        


