from models.NaiveBayes import NaiveBayes
from models.SVM import SVM
from models.SVMGridSearch import SVMGridSearch

model_names = {
    'svm': SVM,
    'naive_bayes': NaiveBayes,
    'svm_grid_search': SVMGridSearch
}


def pick_model(name):
    if name not in model_names: raise Exception('No such model')
    return model_names[name]
