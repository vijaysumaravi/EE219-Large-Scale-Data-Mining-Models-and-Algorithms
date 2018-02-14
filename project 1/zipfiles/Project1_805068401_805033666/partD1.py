from sklearn import svm
import sklearn.metrics as smet
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
from pickle import dump
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from string import punctuation
from nltk.stem.snowball import SnowballStemmer
import string
import re
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.pipeline import Pipeline
import itertools
from sklearn.decomposition import NMF

snowball_stemmer = SnowballStemmer("english")
analyzer = CountVectorizer().build_analyzer()
tfidf_transform = TfidfTransformer()




def stemTokenizer (doc) :
	doc = re.sub('[,.-:/()?{}*$#&]', ' ', doc)
	doc = ''.join(ch for ch in doc if ch not in string.punctuation)
	doc = ''.join(ch for ch in doc if ord(ch) < 128)
	doc = doc.lower()
	words = doc.split()
	words = [word for word in words if word not in text.ENGLISH_STOP_WORDS]

	return [snowball_stemmer.stem(word) for word in words]



def fetchNMFRepresentation(pipeline, train, test):
	nmf_matrix_train = pipeline.fit_transform(train.data)
	nmf_matrix_test = pipeline.transform(test.data)
	return nmf_matrix_train, nmf_matrix_test



comp_categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
rec_categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

twenty_train = fetch_20newsgroups(subset='train', categories=comp_categories+rec_categories, shuffle=True, random_state=42,)
twenty_test = fetch_20newsgroups(subset='test', categories=comp_categories+rec_categories, shuffle=True, random_state=42,)


pipeline1 = Pipeline([
    ('vect', CountVectorizer(min_df=5, stop_words=text.ENGLISH_STOP_WORDS)),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', NMF(n_components=50, init='random', random_state=0)),
])

train_nmf, test_nmf = fetchNMFRepresentation(pipeline1, twenty_train,  twenty_test)

print "NMF Reduced Dimension is", train_nmf.shape



