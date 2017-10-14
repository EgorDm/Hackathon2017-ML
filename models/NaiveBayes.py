from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from models.BaseModel import BaseModel


class NaiveBayes(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
        ])
