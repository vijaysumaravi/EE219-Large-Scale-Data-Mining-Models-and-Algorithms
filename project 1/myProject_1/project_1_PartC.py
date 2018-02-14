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



def stemTokenizer (doc) :
	doc = re.sub('[,.-:/()?{}*$#&]', ' ', doc)
	doc = ''.join(ch for ch in doc if ch not in string.punctuation)
	doc = ''.join(ch for ch in doc if ord(ch) < 128)
	doc = doc.lower()
	words = doc.split()
	words = [word for word in words if word not in text.ENGLISH_STOP_WORDS]

	return [snowball_stemmer.stem(word) for word in words]

comp_categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
rec_categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

all_categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x',
				'rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','alt.atheism','sci.crypt','sci.electronics',
                'sci.med','sci.space','soc.religion.christian','misc.forsale','talk.politics.guns','talk.politics.mideast',
                'talk.politics.misc','talk.religion.misc']


all_docs_per_category=[]

for cat in all_categories:
    categories=[cat]
    all_data = fetch_20newsgroups(subset='train',categories=categories).data
    temp = ""
    for doc in all_data:
        temp= temp + " "+doc
    all_docs_per_category.append(temp)



twenty_train = fetch_20newsgroups(subset='train', categories=comp_categories+rec_categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=comp_categories+rec_categories, shuffle=True, random_state=42)

counts = []

vectorizer = CountVectorizer(analyzer='word',stop_words=text.ENGLISH_STOP_WORDS,ngram_range=(1, 1), tokenizer=stemTokenizer, lowercase=True,max_df=0.99, min_df=2)


# def calculate_tcicf(freq, maxFreq, categories, categories_per_term):
#     val= ((0.5+(0.5*(freq/float(maxFreq))))*math.log10(categories/float(1+categories_per_term)))
#     return val

vectorized_newsgroups_train = vectorizer.fit_transform(all_docs_per_category)

tficf = tfidf_transform.fit_transform(vectorized_newsgroups_train)
tficf_matrix = tficf.toarray()

for category in [2, 3, 14, 15]:
	
	print("10 most significant terms in the category", all_categories[category], "are:\n")
	
	indices = tficf_matrix[category].argsort()[-10:]
	for index in indices:
		print(vectorizer.get_feature_names()[index])






