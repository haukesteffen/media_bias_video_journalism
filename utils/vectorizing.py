import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD

'''from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
import glob'''

rng_seed = 42

### importing data
df = pd.read_csv('data/samples/sample10.csv')
data = df.groupby(['medium'])['preprocessed'].sum()
data = data.loc[['NachDenkSeiten', 'Spiegel', 'ZDFheute', 'BILD', 'Junge Freiheit']]
X = data.values
y = data.index


### instantiating vectorizers
cv = CountVectorizer(ngram_range=(1,3))
X_cv = cv.fit_transform(X)

tfidf = TfidfVectorizer(ngram_range=(1,3))
X_tfidf = tfidf.fit_transform(X)

nlp = spacy.load('de_core_news_sm')
X_we = [nlp(x).vector for x in X]

hv = HashingVectorizer(ngram_range=(1,3))
X_hv = hv.fit_transform(X)


### pca for dimension reduction
pca_cv = TruncatedSVD(n_components=2)
features_cv = pca_cv.fit_transform(X_cv)
xs_cv = features_cv[:,0]
ys_cv = features_cv[:,1]

pca_tfidf = TruncatedSVD(n_components=2)
features_cv = pca_tfidf.fit_transform(X_tfidf)
xs_tfidf = features_cv[:,0]
ys_tfidf = features_cv[:,1]

pca_we = TruncatedSVD(n_components=2)
features_we = pca_we.fit_transform(X_we)
xs_we = features_we[:,0]
ys_we = features_we[:,1]

pca_hv = TruncatedSVD(n_components=2)
features_hv = pca_hv.fit_transform((X_hv))
xs_hv = features_hv[:,0]
ys_hv = features_hv[:,1]


### plots
sns.set()
sns.set_style('darkgrid')
fig, axs = plt.subplots(nrows=2, ncols=2)
sns.scatterplot(x=xs_cv, y=ys_cv, hue=y, ax=axs[0, 0], palette='RdBu').set(title='Count Vectorizer')
sns.scatterplot(x=xs_tfidf, y=ys_tfidf, hue=y, ax=axs[0, 1], palette='RdBu').set(title='TFIDF Vectorizer')
sns.scatterplot(x=xs_we, y=ys_we, hue=y, ax=axs[1, 0], palette='RdBu').set(title='Spacy Word Embeddings Vectorizer')
sns.scatterplot(x=xs_hv, y=ys_hv, hue=y, ax=axs[1, 1], palette='RdBu').set(title='Hashing Vectorizer')
plt.show()