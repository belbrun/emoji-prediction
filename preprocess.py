import sklearn
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

def generate_ngrams(X, max_n_gram = 4):
    counter = CountVectorizer(ngram_range = (1, max_n_gram), lowercase = True, min_df = 2)
    ngrams = counter.fit_transform(X)
    return ngrams

def tf_idf_scaling(ngrams):
    transformer = TfidfTransformer()
    tf_idf_ngrams = transformer.fit_transform(ngrams)
    return tf_idf_ngrams
