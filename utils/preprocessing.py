import pandas as pd
import spacy
import glob
from sklearn.utils import shuffle

def preprocess(text):
    '''
    tokenizes and lemmatizes german input text
    :param text: raw input text (german)
    :return: list of lemmatized tokens from input text
    '''
    doc = nlp(str(text))
    lemmas_tmp = [token.lemma_.lower() for token in doc]
    lemmas = [lemma for lemma in lemmas_tmp if lemma.isalpha() and lemma not in filterwords]
    return ' '.join(lemmas)

### initializing spacy with german language
nlp = spacy.load('de_core_news_sm')
filterwords = spacy.lang.de.stop_words.STOP_WORDS
filterwords.update(['musik', 'music', 'applaus', 'applause'
                    'bild',
                    'spiegel', 'tv',
                    'nachdenkseiten',
                    'junge freiheit', 'j ftv' , 'jfv', 'fjt v', 'jftv',
                    'zdf', 'claus kleber'])
with open('german_stopwords_full.txt', encoding='utf-8', errors='ignore') as d:
    filterwords.update(d.readlines()[9:])
    
### looping through input files
path = 'data/raw/*.csv'
for csv in glob.glob(path):
    ### importing data
    df = pd.read_csv(csv, index_col=0)

    ### preprocess transcript data
    df['preprocessed'] = df['transcript'].apply(preprocess)
    df.to_csv(csv.replace('data\\','data\\preprocessed\\').replace('.csv','_preprocessed.csv'))


### load data
n_samples = [10, 50, 100, 300]
rng_seed = 42
data = pd.DataFrame()
path = 'data\\preprocessed/*.csv'
for i, k in enumerate(n_samples):
    for csv in glob.glob(path):
        tmp = pd.read_csv(csv, index_col=0)
        tmp = tmp.sample(n=k, random_state=rng_seed)
        tmp['medium'] = csv.replace('data\\preprocessed\\', '').replace('_preprocessed.csv', '')
        data = pd.concat([data, tmp])
    data = shuffle(data, random_state=rng_seed).astype(str)
    data.to_csv('data\\samples\\sample'+str(k)+'.csv')
