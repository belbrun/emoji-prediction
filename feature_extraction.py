import re
import torch
import pandas as pd
import numpy as np

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

def has_caps_lock_words(text):
    matches = re.findall(r'\b[A-Z]{3,}\b', text)
    return len(matches)

def make_feature_vector(x):
    features = []
    features.append(has_ellipsis(x))
    features.append(has_char_repetition(x))
    features.append(has_exclamation_repetition(x))
    has_user_value = has_user(x)
    features.append(has_user_value)
    features.append(has_location(x) - has_user_value)
    features.append(has_hashtag(x))
    features.append(has_caps_lock_words(x))

    return features

def make_feature_matrix(X):
    feature_col = []
    for x in X:
        f_vector = make_feature_vector(x)
        feature_col.append(f_vector)

    return pd.Series(feature_col)