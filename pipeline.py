import data
import preprocess
import model
import joblib

from preprocess import Preprocess
from model import Baseline

class Pipeline():

    def __init__(self):
        self.steps = []

    def add(self, step):
        self.steps.append(step)

    def train(self):
        pass

    def run(self):
        pass


class BaselinePipeline(Pipeline):

    def __init__(self, C, k):
        super().__init__()
        self.preprocess = Preprocess(k=k)
        self.model = Baseline(C=C)

    def train(self, X_train, y_train, max_n_gram=4):
        X = self.preprocess.train(X_train, y_train, max_n_gram)
        self.model.train(X, y_train) 

    def run(self, X_test):
        ngrams = self.preprocess.run(X_test)
        return self.model.output(ngrams)

    def save(self, model_file, preprocess_file):
        joblib.dump(self.model, model_file)
        joblib.dump(self.preprocess, preprocess_file)

    def load_model(self, filename):
        self.model = joblib.load(filename)
    
    def load_preprocess(self, filename):
        self.preprocess = joblib.load(filename)
