import data
import preprocess
import model
import joblib
import torch.nn as nn
import feature_extraction as fe

from preprocess import Preprocess, preprocess_tweet
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

    def __init__(self, hidden_size, num_layers, dropout, f_size, text_field, embedding_dim):
        super().__init__()
        self.preprocess = None
        self.embedding_dim = embedding_dim
        pre_trained_emb = torch.FloatTensor(text_field.vocab.vectors)
        self.embedding = nn.Embedding.from_pretrained(pre_trained_emb)
        
        self.text_field = text_field
        self.model = RNN(embedding_dim, hidden_size, num_layers, dropout, f_size)

    def embedd(self, batch_text):
        final_embedding = []
        for text in batch_text:
            text_embedding = []
            for word in text:
                e = self.embedding(word)
                text_embedding.append(e.tolist())
            
            final_embedding.append(text_embedding)
            
        return torch.Tensor(final_embedding)

    def train(self, data):
        avg_loss = 0
        for _, batch in enumerate(data):
            x = batch.text.T
            y = batch.label

            #text features
            f = batch.text_f
            x = self.embedd(x)
            
            avg_loss += self.model.train_model((x, y, f))
        
        return avg_loss/len(data)

    def evaluate(self, data):
        y_ps, y_s = [], []
        for _, batch in enumerate(data):
            x = batch.text
            y = batch.label

            f = batch.text_f
            x = self.embedd(x)
            
            y_ps.append(self.model.evaluate(x, f))
            y_s.append(y)
        
        y_p = torch.cat(y_ps, dim=0)
        y = torch.cat(y_s, dim=0)
        return (y_p, y)
