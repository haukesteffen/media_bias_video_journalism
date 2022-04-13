import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


n = 20


def get_top_tf_idf_words(response, top_n=n):
    sorted_nzs = np.argsort(response.data)[: -(top_n + 1) : -1]
    return feature_names[response.indices[sorted_nzs]]


### importing data
df = pd.read_csv("data/samples/sample300.csv")
data = df.groupby(["medium"])["preprocessed"].sum()
data = data.loc[["NachDenkSeiten", "Spiegel", "ZDFheute", "BILD", "Junge Freiheit"]]
features = data.values
tfidf = TfidfVectorizer(ngram_range=(1, 3))
X = tfidf.fit_transform(df["preprocessed"].dropna())
y = data.index
feature_names = np.array(tfidf.get_feature_names_out())

for idx, feature in enumerate(data):
    responses = tfidf.transform([features[idx]])
    print([get_top_tf_idf_words(response, n) for response in responses])
