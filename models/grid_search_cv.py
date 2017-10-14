from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

classifier = None


def init():
    global classifier
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }
    classifier = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                  alpha=1e-3, n_iter=5, random_state=42)),
    ])
    classifier = GridSearchCV(classifier, parameters, n_jobs=-1)


def train(data, labels):
    classifier.fit(data, labels)


def predict(data):
    return classifier.predict(data)
