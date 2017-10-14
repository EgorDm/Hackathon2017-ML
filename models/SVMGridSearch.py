from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from models.BaseModel import BaseModel


class SVMGridSearch(BaseModel):
    def build_model(self):
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf-svm__alpha': (1e-2, 1e-3),
        }
        self.classifier = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                      alpha=1e-3, n_iter=5, random_state=42)),
        ])
        self.classifier = GridSearchCV(self.classifier, parameters, n_jobs=-1)
