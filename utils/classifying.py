import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD

### choose which media to evaluate
# OPTIONS: NachDenkSeiten, Spiegel, BILD, Junge Freiheit
first = 'Spiegel'
second = 'Junge Freiheit'

### set visualization parameters
sns.set(style='darkgrid')
sns.set_palette('viridis_r')

### load data
seed = 42 #rng seed
n = 300 #number of data points per medium
k = 10 #number of cross validation folds
df_first = pd.read_csv('data/' + first + '_preprocessed.csv', index_col=0)
df_second = pd.read_csv('data/' + second + '_preprocessed.csv', index_col=0)

### sample and shuffle data
first_sample = df_first.sample(n=n, random_state=seed)
first_sample['label'] = first
second_sample = df_second.sample(n=n, random_state=seed)
second_sample['label'] = second
df = pd.concat([first_sample, second_sample])
df = shuffle(df, random_state=seed).astype(str)
print(df.head())

### split data into train and test parts
X_train, X_test, y_train, y_test = train_test_split(df['preprocessed'], df['label'], test_size=0.2, random_state=seed)

### vectorize data and fit and transform model
vectorizer = TfidfVectorizer(ngram_range=(1,3))
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

### instantiate classifier model and fit to training data
clf = MultinomialNB()
clf.fit(X_train_bow, y_train)
y_pred = clf.predict(X_test_bow)

### calculate classification accuracy with k-fold cross validation
scores = cross_val_score(clf, X_test_bow, y_test, cv=k)
print('cross validation scores:\n' + str(scores))
print('mean cv score: ' + str(scores.mean()))
print('confusion matrix:\n' + str(confusion_matrix(y_test, y_pred)))

### data visualization utilizing pca
pca = TruncatedSVD(n_components=2)
features = pca.fit_transform(X_train_bow)
xs = features[:,0]
ys = features[:,1]
sns.scatterplot(x=xs, y=ys, hue=y_train, alpha=0.5, palette='viridis_r')
plt.show()
