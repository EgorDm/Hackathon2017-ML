from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

classifier = None


def init():
    global classifier
    classifier = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                  alpha=1e-3, n_iter=5, random_state=42)),
    ])


def train(data, labels):
    classifier.fit(data, labels)


def predict(data):
    return classifier.predict(data)
