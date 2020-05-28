import os
import numpy as np


dataset_path = 'dataset/Semeval2018-Task2-EmojiPrediction'

def read_file(path):
    with open(path, 'r+', encoding = "utf-8") as file:
        return np.array(file.read().split('\n'))

def save_file(path, data):
    with open(path, 'w+', encoding = "utf-8") as file:
        for information in data:
            file.write(str(information)+'\n')

X = read_file(os.path.join(dataset_path, 'train', 'us_train.text'))
y = read_file(os.path.join(dataset_path, 'train', 'us_train.labels'))

X = X[:len(X)//2]
y = y[:len(y)//2]

save_file(os.path.join(dataset_path, 'train', 'us_train_s.text'), X)
save_file(os.path.join(dataset_path, 'train', 'us_train_s.labels'), y)
