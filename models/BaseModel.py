import numpy as np
import pickle

class BaseModel:
    classifier = None

    def __init__(self, path=None) -> None:
        super().__init__()
        if path is None:
            self.build_model()
            return

        with open(path, 'rb') as f:
            self.classifier = pickle.load(f)

    def build_model(self):
        pass

    def train(self, data, labels):
        self.classifier.fit(data, labels)

    def predict(self, data):
        return self.classifier.predict(data)

    def test(self, data, labels):
        result = self.predict(data)
        return np.mean(result == labels)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.classifier, f)