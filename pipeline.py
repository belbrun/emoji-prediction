import data
import preprocess
import model
import joblib
import torch.nn as nn
import torch
import feature_extraction as fe

from preprocess import Preprocess, preprocess_tweet
from model import Baseline, RNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    def __init__(self, args, text_field):
        super().__init__()
        self.preprocess = preprocess_tweet
        self.embedding_dim = args['embedding_dim']
        pre_trained_emb = torch.FloatTensor(text_field.vocab.vectors)
        self.embedding = nn.Embedding.from_pretrained(pre_trained_emb)

        self.text_field = text_field
        self.model = RNN(args['embedding_dim'], args['hidden_size'],
                         args['num_layers'], args['dropout'], args['f_size'])
        self.model.to(device)

    def train(self, data):
        avg_loss = 0
        for num_batch, batch in enumerate(data):
            x = batch.text.T
            y = batch.label

            #text features
            f = batch.text_f
            x = self.embedding(self.preprocess(x))

            avg_loss += self.model.train_model((x, y, f))
            if num_batch%100 == 0:
                print('Loss:', avg_loss/(num_batch + 1))

        try:
            torch.save(self.model.state_dict(), 'torch_model.pt')
        except:
            print("error saving the model..")
        return avg_loss/len(data)

    def load_model(self):
        self.mode = torch.load('torch_model.pt')

    def evaluate(self, data):
        y_ps, y_s = [], []
        avg_loss = 0
        for num_batch, batch in enumerate(data):
            x = batch.text.T
            y = batch.label

            f = batch.text_f
            x = self.embedding(self.preprocess(x))

            logits = self.model.evaluate(x, f)
            with torch.no_grad():
                loss = self.model.eval_criterion(logits, y).item()
                avg_loss += loss
                if num_batch % 100 == 0:
                    print('Loss:', avg_loss/(num_batch + 1))

            y_ps.append(torch.argmax(logits, dim=1))
            y_s.append(y)

        y_p = torch.cat(y_ps, dim=0)
        y = torch.cat(y_s, dim=0)
        return (y_p, y)
