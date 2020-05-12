import os

def read_file(path):
    with open(path, 'r+') as file:
        return file.read().split('')

def load_data(path, set):
    X = read_file(os.join(path, set, 'us_' + set + '.text'))
    y = read_file(os.join(path, set, 'us_' + set + '.labels'))
