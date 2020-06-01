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


def load_data(set='test'):
    X = read_file(os.path.join(dataset_path, set, 'us_' + set + '.text'))[:1000]
    y = read_file(os.path.join(dataset_path, set, 'us_' + set + '.labels'))[:1000]
    #print(len(X), set)
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
        tokenize='basic_english',
        lower=True
    )

    #preprocessed_text = text.apply(
    #    lambda x: field.preprocess(x)
    #)
    preprocessed_text = preprocess_tweet(text)
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


def get_demo_iterator(data, batch_size, embedding_dim):
    data = add_features(data)
    text_field = get_text_field(data['text'], embedding_dim)
    label_field = get_label_field()
    dataset = get_dataset(data, text_field, label_field)
    return (get_iterator(dataset, batch_size, set == 'train'), text_field)

def get_iterators(batch_size, embedding_dim, vocab_only=False):
    iters = []
    sets = ['train'] if vocab_only else ['train', 'trial', 'test']
    for set in sets:
        data = load_data(set)
        data = add_features(data)
        if set == 'train':
            text_field = get_text_field(data['text'], embedding_dim)
        label_field = get_label_field()
        dataset = get_dataset(data, text_field, label_field)
        iters.append(get_iterator(dataset, batch_size, set == 'train'))
    return (iters, text_field)

if __name__ == "__main__":
    train_data = load_data('test')
    train_data = add_features(train_data)

    text_field = get_text_field(train_data.text)
    label_field = get_label_field()

    train_dataset = get_dataset(train_data, text_field, label_field)
    train_iter = get_iterator(train_dataset, 10)

    for i in range(10):
        batch = next(iter(train_iter))
        print(batch.text)
        print(batch.label)
        print(batch.text_f)
