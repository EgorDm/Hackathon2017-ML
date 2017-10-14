import re
from nltk import corpus

from tools.constants import contractions


def clean_text(text):
    text = text.lower()

    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    words = text.split()
    words = clean_list(words)
    words = remove_stopwords(words, 'english')
    words = remove_stopwords(words, 'dutch')
    words = apply_blacklist(words)
    text = ' '.join(words)

    return text


def remove_stopwords(words, lang='english'):
    stops = set(corpus.stopwords.words(lang))
    return [w for w in words if not w in stops]

def apply_blacklist(words):
    with open('data/word_blacklist.txt', 'r') as f:
        blacklist = f.read().splitlines()
    return [w for w in words if not w in blacklist]


def clean_list(l):
    ret = []
    for item in l:
        if item is None or item == '': continue
        ret.append(item)
    return ret
