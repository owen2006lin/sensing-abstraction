from abc import ABC, abstractmethod
import joblib

class Component(ABC):
    @abstractmethod
    def fit(self, X, y, X_val = None, y_val = None, params = None, callbacks = None):
        "Train on features X and labels y. Optional validation sets and parameters"
 
    @abstractmethod
    def predict_proba(self,X):
        "Return class probabilities, eg P(slip = 0, 1, 2) -> [0.5, 0.25, 0.25]"
    
    @abstractmethod
    def sample(self, X, rng):
        "Draw a random outcome using probs X"


    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        obj = joblib.load(path)
        assert isinstance(obj, cls)
        return obj