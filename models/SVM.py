from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from models.BaseModel import BaseModel


class SVM(BaseModel):
    def build_model(self):
        self.classifier = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                      alpha=1e-3, n_iter=5, random_state=42)),
        ])


