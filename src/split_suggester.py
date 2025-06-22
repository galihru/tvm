import math

def suggest_split(n_samples: int):
    r = 1 - 1/math.sqrt(max(n_samples, 2))
    r = max(0.6, min(0.9, r))
    return r, 1 - r
