import data
import preprocess
import model
import joblib

from preprocess import Preprocess
from model import Baseline, RNN

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


class RNNPipeline(Pipeline):

    def __init__(self):
        super().__init__()
        self.embedd = None #embeddings part should handle the padding for the batch
        self.text_features = None
        self.word_features = None
        self.model = RNN()

    def train(data):
        avg_loss = 0
        for batch_num, batch in enumerate(data):
            words, labels = batch
            x = self.embedd(words)
            x = torch.cat((x, self.word_features(words)), dim=1)
            f = self.word_features(words)
            loss = self.model.train(x, f)
        return avg_loss/len(data)

    def evaluate(data):
        y_ps, y_s = [], []
        for batch_num, batch in enumerate(data):
            words, labels = batch
            y_s.append(labels)
            x = self.embedd(words)
            x = torch.cat((x, self.word_features(words)), dim=1)
            f = self.text_features(words)
            y_ps.append(self.model.evaluate(x, f))
        y_p = torch.cat(y_ps, dim=0)
        y = torch.cat(ys, dim=0)
        return (y_p, y)
