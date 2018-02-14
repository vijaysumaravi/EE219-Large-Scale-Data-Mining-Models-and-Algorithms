#TFxIDF to calculate to get the significant terms of the given data set for the given classes:


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


snowball_stemmer = SnowballStemmer("english")
analyzer = CountVectorizer().build_analyzer()
tfidf_transform = TfidfTransformer()



# def stemTokenizer (doc) :
# 	doc = re.sub('[,.-:/()?{}*$#&]', ' ', doc)
# 	doc = ''.join(ch for ch in doc if ch not in string.punctuation)
# 	doc = ''.join(ch for ch in doc if ord(ch) < 128)
# 	doc = doc.lower()
# 	words = doc.split()
# 	words = [word for word in words if word not in text.ENGLISH_STOP_WORDS]

# 	return [snowball_stemmer.stem(word) for word in words]

computerClass = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
recreationClass = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

# class_1 = fetch_20newsgroups(subset = 'all', categories = computerClass, shuffle=True, random_state=42);
# class_2 = fetch_20newsgroups(subset = 'all', categories = recreationClass, shuffle=True, random_state=42);

total = fetch_20newsgroups(subset = 'all', categories = computerClass+recreationClass, shuffle=True, random_state=42);

		


count_vect1 = CountVectorizer(min_df=3, stop_words='english')


# class_1_Count = count_vect1.fit_transform(class_1.data)
# class_1_tfidf = tfidf_transform.fit_transform(class_1_Count)
# print "Number of terms in class1 data TF-IDF representation:",class_1_tfidf.shape

# class_2_Count = count_vect1.fit_transform(class_2.data)
# class_2_tfidf = tfidf_transform.fit_transform(class_2_Count)
# print "Number of terms in class2 data TF-IDF representation:",class_2_tfidf.shape

totalCount = count_vect1.fit_transform(total.data)
totalData_tfidf = tfidf_transform.fit_transform(totalCount)
print "Number of terms in combined data TFxIDF representation:",totalData_tfidf.shape





