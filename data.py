import os
import numpy as np
from  torch.utils.data import Dataset, DataLoader


dataset_path = 'dataset/Semeval2018-Task2-EmojiPrediction'

def read_file(path):
    with open(path, 'r+', encoding = "utf-8") as file:
        return np.array(file.read().split('\n'))

def load_data(set='test'):
    X = read_file(os.path.join(dataset_path, set, 'us_' + set + '.text'))
    y = read_file(os.path.join(dataset_path, set, 'us_' + set + '.labels'))
    return (X, y)

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
    
    for data in dl:
        tweets = data[0]
        labels = data[1]
        for i in range(10):
            print(tweets[i], labels[i])
        break
