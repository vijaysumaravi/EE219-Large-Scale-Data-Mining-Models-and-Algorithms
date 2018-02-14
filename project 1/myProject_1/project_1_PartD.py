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
from sklearn.pipeline import Pipeline
import itertools


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

def get_svd():
    return TruncatedSVD(n_components=50)


def fetchLSIRepresentation(pipeline, train, test):
	svd_matrix_train = pipeline.fit_transform(train.data)
	svd_matrix_test = pipeline.transform(test.data)
	return svd_matrix_train, svd_matrix_test


comp_categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
rec_categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

twenty_train = fetch_20newsgroups(subset='train', categories=comp_categories+rec_categories, shuffle=True, random_state=42,)
twenty_test = fetch_20newsgroups(subset='test', categories=comp_categories+rec_categories, shuffle=True, random_state=42,)


# count_vect1 = CountVectorizer(min_df=2,max_df=0.99, tokenizer=stemTokenizer)
# X_train_counts1 = count_vect1.fit_transform(twenty_train.data)
# X_train_tfidf = tfidf_transform.fit_transform(X_train_counts1)

# print "Number of terms in TF-IDF representation:",X_train_tfidf.shape, "\n"

# u, s, vt = svds(X_train_tfidf.toarray(), k=50)

# print " Size of reduced dimension using SVD is: ", u.shape

pipeline1 = Pipeline([
    ('vect', CountVectorizer(min_df=5, stop_words=text.ENGLISH_STOP_WORDS)),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', get_svd()),
])

train_lsi, test_lsi = fetchLSIRepresentation(pipeline1, twenty_train,  twenty_test)

print "LSI Reduced Dimension is", train_lsi.shape





