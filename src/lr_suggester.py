import numpy as np

class LRSuggester:
    """
    Suggest learning rate based on gradient variance estimate and Fisher info.
    lr ~ sqrt(2 / (epochs * var_grad)). If var not known, use lr ~ 1/sqrt(n_features).
    """
    @staticmethod
    def suggest(n_features: int, epochs: int = 50, grad_var: float = None):
        if grad_var:
            lr = np.sqrt(2.0 / (epochs * grad_var))
        else:
            lr = 1.0 / math.sqrt(max(n_features,1))
        # clamp
        return float(np.clip(lr, 1e-5, 1e-1))
