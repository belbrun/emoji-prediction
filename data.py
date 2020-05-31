import os

from torchtext.data import Field
from torchtext.data import Dataset, Example
from torchtext.data import BucketIterator

import pandas as pd
import numpy as np
import feature_extraction as fe
from preprocess import preprocess_tweet

dataset_path = 'dataset/Semeval2018-Task2-EmojiPrediction'

def read_file(path):
    with open(path, 'r+', encoding = "utf-8") as file:
        return np.array(file.read().split('\n'))

def write_log(log):
    with open('log.txt', 'w+') as file:
        for line in log:
            file.write(str(line)+'\n')

def load_raw_data(set='test', is_subset=True):
    if is_subset and set=='train':
        X = np.array(read_file(os.path.join(dataset_path, 'train', 'us_train_s.text')))
        y = np.array(read_file(os.path.join(dataset_path, 'train', 'us_train_s.labels')))
    else:
        X = np.array(read_file(os.path.join(dataset_path, set, 'us_' + set + '.text')))
        y = np.array(read_file(os.path.join(dataset_path, set, 'us_' + set + '.labels')))

    mask = y != ''
    X = X[mask]
    y = y[mask]

    return X, y.astype(np.int64)


def load_data(set='test'):
    X = read_file(os.path.join(dataset_path, set, 'us_' + set + '.text'))
    y = read_file(os.path.join(dataset_path, set, 'us_' + set + '.labels'))
    print(len(X), set)
    d = {'text':X, 'label':y}
    df = pd.DataFrame(data=d)
    df = df.drop(df[df.label == ''].index)

    return df

def class_weights():
    df = load_data()
    counts = np.zeros(20)
    for yi in df['label']:
        counts[int(yi)] += 1
    return np.power(counts,-1)*np.max(counts)

def get_text_field(text, embedding_dim):
    field = Field(
        preprocessing=preprocess_tweet,
        lower=True
    )

    preprocessed_text = text.apply(
        lambda x: field.preprocess(x)
    )

    field.build_vocab(
            preprocessed_text,
            vectors='glove.twitter.27B.{}d'.format(embedding_dim)
    )

    return field

def get_label_field():
    field = Field(sequential=False, use_vocab=False)
    return field

class SemEvalDataset(Dataset):
    def __init__(self, data, fields):
        e = [
                Example.fromlist(list(r), fields)
                for i, r in data.iterrows()
            ]
        super(SemEvalDataset, self).__init__(e, fields)

    def __len__(self):
        return len(self.examples)

def get_dataset(data, text_field, label_field):
    fields = [('text',text_field), ('label',label_field), ('text_f', label_field)]
    dataset = SemEvalDataset(data, fields)
    return dataset

def get_iterator(dataset, batch_size, train):
    iterator = BucketIterator(
        dataset=dataset, batch_size=batch_size,
        sort_key=lambda x: len(x.text), shuffle=True, train=train
    )
    return iterator

def add_features(data):
    features = fe.make_feature_matrix(data.text)
    data['text_f'] = features

    return data

def get_iterators(batch_size, embedding_dim, join=None):
    iters = []
    for set in ['train', 'trial', 'test']:
        data = load_data(set)
        data = add_features(data)
        if set == 'train':
            text_field = get_text_field(data['text'], embedding_dim)
        label_field = get_label_field()
        dataset = get_dataset(data, text_field, label_field)
        iters.append(get_iterator(dataset, batch_size, set == 'train'))
    return (iters, text_field)
