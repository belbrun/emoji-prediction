import data
import preprocess
import model

def baseline_pipeline(X, y):
    #preprocess
    ngrams = preprocess.generate_ngrams(X)
    print('{}'.format(ngrams.shape))

    scaled_ngrams = preprocess.tf_idf_scaling(ngrams)

    #train
    y_pred = model.train_transform(scaled_ngrams, y)
    return y_pred

class Pipeline():

    def __init__(self):
        self.steps = []

    def add(self, step):
        self.steps.append()

    def train(self):
        pass

    def run(self):
        pass


class BaselinePipeline(Pipeline):

    def __init__(self):
        super().__init__()
        self.add(preprocess.generate_ngrams())
        self.add(preprocess.tf_idf_scaling())
        self.add(model.Baseline(C=0.1))

    def train(self, X, y):
        for step in self.steps:
            if not issubclass(step.__class__, model.Model) :
                X = step(X)
            else :
                X = step.train(X, y)

    def run(self, X):
        for step in self.steps:
            if not issubclass(step.__class__, model.Model) :
                X = step(X)
            else :
                X = step.output(X, y)
