import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def standardize(X):
    return StandardScaler().fit_transform(X)

def encode_labels(y):
    return LabelEncoder().fit_transform(y)
