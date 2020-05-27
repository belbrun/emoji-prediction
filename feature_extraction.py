import re
import torch
from data import load_data
import pandas as pd
from nltk import word_tokenize
import csv
import numpy as np

with open('acronym_dict.csv') as csv_file:
    reader = csv.reader(csv_file)
    acronyms = dict(reader)

def has_ellipsis(text):
    matches = re.findall(r'[.][.][.]', text)
    return len(matches)

def has_char_repetition(text):
    matches = re.findall(r'([a-zA-Z])(\1{3,})', text)
    return len(matches)

def has_exclamation_repetition(text):
    matches = re.findall(r'[!]{2,}', text)
    return len(matches)

def has_user(text):
    matches = re.findall(r'@user', text)
    return len(matches)

def has_location(text):
    matches = re.findall(r'@(?!\W+cat\b)', text)
    return len(matches)

def has_hashtag(text):
    matches = re.findall(r'([#])', text)
    return len(matches)

def is_caps_lock(word):
    return int(word.isupper())

def has_acronym(text, acronyms):
    counts = []
    for acronym in acronyms.keys():
        pattern = r'\b(' + acronym + r')\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        counts.append(len(matches))
    return counts
    
def word_features(sentences):
    final_tensor = []
    
    for sentence in sentences:
        sentence_features = []
        words = word_tokenize(sentence)    
        for word in words:
            features = []
            features.append(has_char_repetition(word))
            features.append(is_caps_lock(word))
            sentence_features.append(features)
        
        final_tensor.append(sentence_features)
    return torch.Tensor(final_tensor)

def text_features(sentences):
    all_features = []
    for x in sentences:
        features = []
        features.append(has_ellipsis(x))
        features.append(has_exclamation_repetition(x))
        has_user_value = has_user(x)
        features.append(has_user_value)
        features.append(has_location(x) - has_user_value)
        features.append(has_hashtag(x))
        features.extend(has_acronym(x, acronyms))
        all_features.append(features)
    
    return torch.Tensor(all_features)

