
# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy as np

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# random, itertools, matplotlib
import random
import itertools
import matplotlib.pyplot as plt

from LabeledLineSentence import LabeledLineSentence
sources = {
    'data/test-neg.txt':'TEST_NEG',
    'data/test-pos.txt':'TEST_POS', 
    'data/train-neg.txt':'TRAIN_NEG', 
    'data/train-pos.txt':'TRAIN_POS', 
    'data/train-unsup.txt':'TRAIN_UNS'
}

sentences = LabeledLineSentence(sources)
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)

model.build_vocab(sentences.to_array())
model.train(sentences.sentences_perm(), total_examples=model.corpus_count, epochs=10)

model.save('./imdb.d2v')

