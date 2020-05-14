import data
import preprocess
import baseline

def baseline_pipeline(X, y):
    #preprocess
    ngrams = preprocess.generate_ngrams(X)
    scaled_ngrams = preprocess.tf_idf_scaling(ngrams)
    
    #train
    y_pred = baseline.train_transform(scaled_ngrams, y)
    return y_pred