import math, numpy as np

def suggest_lr(complexity: float, epochs: int=50, grad_var: float=None):
    if grad_var:
        lr = math.sqrt(2/(epochs*grad_var))
    else:
        lr = 1/math.sqrt(max(complexity,1))
    return float(np.clip(lr, 1e-5, 1e-1))
