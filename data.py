import os
import numpy as np

dataset_path = 'dataset/Semeval2018-Task2-EmojiPrediction'

def read_file(path):
    with open(path, 'r+', encoding = "utf-8") as file:
        return np.array(file.read().split('\n'))

def load_data(set='test'):
    X = read_file(os.path.join(dataset_path, set, 'us_' + set + '.text'))
    y = read_file(os.path.join(dataset_path, set, 'us_' + set + '.labels'))
    return (X, y)
