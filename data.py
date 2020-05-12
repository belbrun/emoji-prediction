import os
import pandas as pd
import csv

dataset_path = 'dataset/Semeval2018-Task2-EmojiPrediction'
embeddings_path = 'embeddings'

def read_file(path):
    with open(path, 'r+') as file:
        return file.read().split('\n')

def load_data(set='test'):
    X = read_file(os.path.join(dataset_path, set, 'us_' + set + '.text'))
    y = read_file(os.path.join(dataset_path, set, 'us_' + set + '.labels'))
    return (X, y)

def load_embeddings(dim=100):
    file = os.path.join(embeddings_path,
                        'glove.twitter.27B.{}d.txt'.format(str(dim)))
    return pd.read_table(file,
                           sep=" ",
                           index_col=0,
                           header=None,
                           quoting=csv.QUOTE_NONE)


if __name__ == '__main__':
    X, y = load_data()
    print('{}\n{}'.format(X[0], y[0]))
    words = load_embeddings()
    print(words.loc['example'])
