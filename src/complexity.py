import math
import numpy as np

def tabular_complexity(n_samples, n_features, avg_corr):
    return math.log1p(n_features) * (1-avg_corr) * math.sqrt(n_samples)

def image_complexity(avg_w, avg_h, class_count, imbalance):
    return (avg_w*avg_h) * math.log1p(class_count) / (1+imbalance)

def text_complexity(entropy, vocab_size):
    return entropy * math.sqrt(vocab_size)
