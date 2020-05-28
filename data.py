import os
import numpy as np
from  torch.utils.data import Dataset, DataLoader
from nltk import word_tokenize
import pandas as pd
import csv

dataset_path = 'dataset/Semeval2018-Task2-EmojiPrediction'
embeddings_path = 'embeddings'

def read_file(path):
    with open(path, 'r+', encoding = "utf-8") as file:
        return np.array(file.read().split('\n'))

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

def get_word_embedding(embeddings, embedding_dim, word):
    try:
        return embeddings.loc[word].values
    except:
        return np.zeros(embedding_dim)


def embedd_sentences(embeddings, embedding_dim, sentences):
    embedded_sentences = []
    max_sentence_length = 0

    for sentence in sentences:
        sentence_embeddings = []
        words = word_tokenize(sentence)
        #track longest sentence for padding
        if len(words) > max_sentence_length:
            max_sentence_length = len(words)

        for word in words:
            sentence_embeddings.append(get_word_embedding(embeddings, embedding_dim, word))

        embedded_sentences.append(sentence_embeddings)

    # pad all sentences to length of longest sentence
    for sentence in embedded_sentences:
        while len(sentence) < max_sentence_length:
            sentence.append(np.zeros(embedding_dim))

    return torch.Tensor(embedded_sentences)

def get_dataloader(set, batch_size, shuffle):
    dataset = SemevalDataset(set)
    return DataLoader(dataset=dataset, batch_size=batch_size,
                                      shuffle=shuffle)

class SemevalDataset(Dataset):

    def __init__(self, set):
        self.X = read_file(os.path.join(dataset_path, set, 'us_' + set + '.text'))
        self.y = read_file(os.path.join(dataset_path, set, 'us_' + set + '.labels'))

    def __getitem__(self, idx):
        return (X[idx], y[idx])

    def __len__(self):
        return self.X.shape[0]





if __name__ == '__main__':
    dl = get_dataloader('test', 10, True)
    X, y = load_data()
    print(X[0].split())
    embeddings = load_embeddings()

    for data in dl:
        tweets = data[0]
        labels = data[1]
        for i in range(10):
            print(tweets[i], labels[i])
        break

    print(embeddings.loc['rt'].values)
