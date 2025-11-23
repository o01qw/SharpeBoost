from .trainer import SharpePruneTrainer

class SharpeBoostRegressor:
    """
    Scikit-learn compatible wrapper for SharpeBoost.
    """
    def __init__(self, **kwargs):
        self.trainer = SharpePruneTrainer(**kwargs)

    def fit(self, X_train, y_train, X_val, y_val, returns_val):
        self.trainer.fit_one_split(X_train, y_train, X_val, y_val, returns_val)
        return self

    def predict(self, X):
        return self.trainer.predict(X)
