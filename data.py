import os

from torchtext.data import Field 
from torchtext.data import Dataset, Example
from torchtext.data import BucketIterator

import pandas as pd
import numpy as np
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
    field =  Field(
        preprocessing=preprocess_tweet,
        tokenize='basic_english', 
        lower=True
    )

    field.build_vocab(
            text, 
            vectors='glove.twitter.27B.100d'
    )
    
    return field

def get_label_filed():
    field = Field(sequential=False, use_vocab=False)
    return field

class SemEvalDataset(Dataset):
    def __init__(self, data, fields):
        super(SemEvalDataset, self).__init__(
            [
                Example.fromlist(list(r), fields) 
                for i, r in data.iterrows()
            ], 
            fields
        )

def get_dataset(data, text_field, label_field):
    dataset = SemEvalDataset(data, [('text',text_field), ('label',label_field)])
    return dataset

def get_iterator(dataset, batch_size):
    iterator = BucketIterator(
        dataset=dataset, batch_size=batch_size,
        sort_key=lambda x: len(x.text)
    )
    return iterator

if __name__ == "__main__":
    train_data = load_data('train')
    text_field = get_text_field(train_data['text'])
    label_field = get_label_filed()

    train_dataset = get_dataset(train_data, text_field, label_field)
    train_iter = get_iterator(train_dataset, 10)