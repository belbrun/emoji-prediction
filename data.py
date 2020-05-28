import os

from torchtext.data import Field, ReversibleField
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

def load_data(set='test'):
    X = read_file(os.path.join(dataset_path, set, 'us_' + set + '.text'))
    y = read_file(os.path.join(dataset_path, set, 'us_' + set + '.labels'))
    d = {'text':X, 'label':y}
    return pd.DataFrame(data=d)

def get_text_field(text):
    field = Field(
        preprocessing=preprocess_tweet,
        tokenize='basic_english',
        lower=True
    )

    preprocessed_text = text.apply(
        lambda x: field.preprocess(x)
    )
    
    field.build_vocab(
            preprocessed_text, 
            vectors='glove.twitter.27B.100d'
    )
    
    return field

def get_label_field():
    field = Field(sequential=False, use_vocab=False)
    return field

class SemEvalDataset(Dataset):
    def __init__(self, data, fields):
        self.fields = fields
        e = [
                Example.fromlist(list(r), fields) 
                for i, r in data.iterrows()
            ]
        super(SemEvalDataset, self).__init__(e, fields)

def get_dataset(data, text_field, label_field):
    fields = [('text',text_field), ('label',label_field), ('text_f', label_field)]
    dataset = SemEvalDataset(data, fields)
    return dataset

def get_iterator(dataset, batch_size):
    iterator = BucketIterator(
        dataset=dataset, batch_size=batch_size,
        sort_key=lambda x: len(x.text),
    )
    return iterator

def add_features(data):
    features = fe.make_feature_matrix(data.text)
    data['text_f'] = features

    return data

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
        print(batch.text_f, '\n')