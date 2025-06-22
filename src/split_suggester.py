import math

def suggest_split(n_samples: int):
    # r_train = 1 - 1/√n, dibatasi [0.6,0.9]
    r = 1 - 1/math.sqrt(max(n_samples,2))
    return max(0.6,min(0.9,r)), 1 - max(0.6,min(0.9,r))
