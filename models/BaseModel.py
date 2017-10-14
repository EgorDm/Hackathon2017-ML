import numpy as np

class BaseModel:
    classifier = None

    def __init__(self) -> None:
        super().__init__()

    def train(self, data, labels):
        self.classifier.fit(data, labels)

    def predict(self, data):
        return self.classifier.predict(data)

    def test(self, data, labels):
        result = self.predict(data)
        return np.mean(result == labels)

