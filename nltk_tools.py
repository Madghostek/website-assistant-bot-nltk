import nltk,numpy as np
from nltk.stem.porter import PorterStemmer
gStemmer = PorterStemmer()


def tokenizer(s):
    return nltk.word_tokenize(s)

def stemmer(word):
    return gStemmer.stem(word.lower())

#bag of words
def compareWords(tokens,allwords):
    tokens = [stemmer(w) for w in tokens]
    match = np.zeros(len(allwords),dtype=np.float32)
    for index, w in enumerate(allwords):
        if w in tokens:
            match[index] = 1.0
    return match


