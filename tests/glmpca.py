from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

topic = 'blm_movement'
#topic = 'military_spending'
data = pd.read_csv(f'{topic}_counts.csv', index_col=0)
data.drop(columns=['TOTAL'], inplace=True)

political_dict = {
    'aljazeera':'left leaning',
    'alternet':'left',
    'americanconservative':'right leaning',
    'americanspectator':'right',
    'americanthinker':'right',
    'antiwar':'right',
    'ap':'center',
    'atlantic':'left leaning',
    'axios':'center',
    'bbc':'center',
    'breitbart':'right',
    'businessinsider':'center',
    'buzzfeed':'left',
    'canary':'left',
    'cbs':'left leaning',
    'cnbc':'center',
    'cnn':'left leaning',
    'commondreams':'left',
    'consortium':'center',
    'conversation':'left leaning',
    'counterpunch':'left',
    'dailycaller':'right',
    'dailykos':'left',
    'dailymail':'right',
    'dailywire':'right',
    'economist':'left leaning',
    'federalist':'right',
    'fivethirtyeight':'center',
    'foreignaffairs':'center',
    'fox':'right leaning',
    'freebeacon':'right',
    'globalresearch':'center',
    'grayzone':'left',
    'guardian':'left leaning',
    'huffingtonpost':'left',
    'independent':'left leaning',
    'infowars':'right',
    'intercept':'left',
    'jacobinmag':'left',
    'jpost':'center',
    'lewrockwell':'right',
    'mintpress':'left leaning',
    'motherjones':'left',
    'msnbc':'left',
    'nationalreview':'right',
    'nbc':'left leaning',
    'neweasternoutlook':'right',
    'newyorker':'right',
    'npr':'center',
    'nypost':'right',
    'nytimes':'left',
    'observerny':'center',
    'offguardian':'left',
    'opednews':'left leaning',
    'pbs':'center',
    'peoplesdaily':'left',
    'pjmedia':'right leaning',
    'propublica':'left leaning',
    'psychologytoday':'center',
    'rawstory':'left',
    'redstate':'right',
    'reuters':'center',
    'reveal':'left',
    'rt':'center',
    'shadowproof':'left',
    'slate':'left',
    'snopes':'center',
    'socialistalternative':'left',
    'socialistpress':'left',
    'spectator':'right',
    'strategicculture':'right',
    'techcrunch':'center',
    'time':'left leaning',
    'townhall':'right',
    'truthdig':'left',
    'usatoday':'center',
    'verge':'left leaning',
    'veteranstoday':'right',
    'vice':'left',
    'voltairenetwork':'center',
    'vox':'left',
    'wapo':'left leaning',
    'wsj':'center',
    'electronicfrontierfoundation':'center',
}

establishment_dict = {
    'aljazeera':'con',
    'alternet':'con',
    'americanconservative':'con',
    'americanspectator':'pro',
    'americanthinker':'pro',
    'antiwar':'con',
    'ap':'pro',
    'atlantic':'pro',
    'axios':'pro',
    'bbc':'pro',
    'breitbart':'pro',
    'businessinsider':'pro',
    'buzzfeed':'pro',
    'canary':'con',
    'cbs':'pro',
    'cnbc':'pro',
    'cnn':'pro',
    'commondreams':'con',
    'consortium':'con',
    'conversation':'pro',
    'counterpunch':'con',
    'dailycaller':'pro',
    'dailykos':'con',
    'dailymail':'pro',
    'dailywire':'pro',
    'economist':'pro',
    'federalist':'pro',
    'fivethirtyeight':'pro',
    'foreignaffairs':'pro',
    'fox':'pro',
    'freebeacon':'pro',
    'globalresearch':'con',
    'grayzone':'con',
    'guardian':'pro',
    'huffingtonpost':'pro',
    'independent':'pro',
    'infowars':'con',
    'intercept':'con',
    'jacobinmag':'con',
    'jpost':'con',
    'lewrockwell':'con',
    'mintpress':'con',
    'motherjones':'con',
    'msnbc':'pro',
    'nationalreview':'pro',
    'nbc':'pro',
    'neweasternoutlook':'con',
    'newyorker':'con',
    'npr':'pro',
    'nypost':'pro',
    'nytimes':'pro',
    'observerny':'con',
    'offguardian':'con',
    'opednews':'con',
    'pbs':'pro',
    'peoplesdaily':'con',
    'pjmedia':'pro',
    'propublica':'pro',
    'psychologytoday':'pro',
    'rawstory':'con',
    'redstate':'pro',
    'reuters':'pro',
    'reveal':'con',
    'rt':'con',
    'shadowproof':'con',
    'slate':'pro',
    'snopes':'pro',
    'socialistalternative':'con',
    'socialistpress':'con',
    'spectator':'pro',
    'strategicculture':'con',
    'techcrunch':'con',
    'time':'pro',
    'townhall':'con',
    'truthdig':'con',
    'usatoday':'pro',
    'verge':'pro',
    'veteranstoday':'con',
    'vice':'pro',
    'voltairenetwork':'con',
    'vox':'pro',
    'wapo':'pro',
    'wsj':'pro',
    'electronicfrontierfoundation':'con',
}

'''color_palette = [
    'rgb(90.0, 0.0, 210.0)',
    'rgb(0.0, 0.0, 255.0)',
    'rgb(210.0, 0.0, 90.0)',
    'rgb(255.0, 0.0, 0.0)',
    'rgb(120.0, 120.0, 120.0)',
]'''

data = data[['cnn','dailymail','redstate','fox','breitbart','ap','nbc','usatoday','wapo','counterpunch','rt','guardian','cbs','independent','buzzfeed','aljazeera','npr','nypost','nationalreview','dailywire','huffingtonpost','commondreams','nytimes','rawstory','alternet','motherjones','intercept','vox','bbc','businessinsider','dailykos','time','slate','vice','dailycaller','infowars','federalist','pjmedia','townhall','spectator']]
color_palette = [
    'rgb(90.0, 0.0, 210.0)',
    'rgb(255.0, 0.0, 0.0)',
    'rgb(210.0, 0.0, 90.0)',
    'rgb(120.0, 120.0, 120.0)',
    'rgb(0.0, 0.0, 255.0)',
]


n_components = 3
scale = True 

scaler = StandardScaler()
#model = TruncatedSVD(n_components=n_components)
model = PCA(n_components=n_components)

if scale:
    data_scaled = scaler.fit_transform(data.values)
    data_trunc = model.fit_transform(data_scaled)
else:
    data_trunc = model.fit_transform(data.values)

'''fig = px.scatter(
    x=model.components_[1], 
    y=model.components_[2], 
    color=[political_dict[medium] for medium in data.columns.to_list()],
    color_discrete_sequence=color_palette,
    symbol=[establishment_dict[medium] for medium in data.columns.to_list()],
    hover_data=[data.columns.to_list()], 
    title=f'Topic: BLM, 1. and 2. Principal Component',
    )
fig.show()'''


from glmpca import glmpca
res = glmpca.glmpca(data.values, 3)
factors= res["factors"]
fig = px.scatter(
    x=factors[:,0], 
    y=factors[:,2], 
    color=[political_dict[medium] for medium in data.columns.to_list()],
    color_discrete_sequence=color_palette,
    symbol=[establishment_dict[medium] for medium in data.columns.to_list()],
    hover_data=[data.columns.to_list()], 
    title=f'Topic: BLM, 1. and 2. Principal Component',
    )
fig.show()