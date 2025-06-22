import math

class SplitSuggester:
    """
    Suggest dynamic train/eval split based on dataset size and complexity.
    From Q1 literature: r_train = 1 - 1/sqrt(n_samples).
    """
    @staticmethod
    def suggest(n_samples: int, n_features: int = None):
        # formula: train_ratio = 1 - 1/sqrt(n)
        r = 1 - 1/math.sqrt(max(n_samples,2))
        # clamp between 0.6 and 0.9
        r = max(0.6, min(0.9, r))
        return r, 1-r
