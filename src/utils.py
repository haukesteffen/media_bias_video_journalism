from youtube_transcript_api import YouTubeTranscriptApi
from youtubesearchpython import Playlist, Channel, playlist_from_channel_id
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from tqdm import tqdm
import spacy
from pandarallel import pandarallel
import joblib
import numpy as np

topic_dict = {
    0: "Misc1",
    1: "Ampelregierung",
    2: "Innenpolitik",
    3: "Familie",
    4: "Gender",
    5: "Wahlen",
    6: "Justiz",
    7: "Medien",
    8: "Impfung",
    9: "Abspann",
    10: "Live",
    11: "Lokal",
    12: "Befragung",
    13: "Krieg",
    14: "Angela Merkel",
    15: "Fußball",
    16: "Ukrainekonflikt",
    17: "Wirtschaft",
    18: "Schule",
    19: "COVID-19 Maßnahmen",
    20: "Interview",
    21: "USA",
    22: "Misc2",
    23: "International",
    24: "CDU-CSU",
}


def define_print(verbose=True):
    if verbose:
        verboseprint = print
    else:
        verboseprint = lambda *args: None
    return verboseprint


def get_transcript(video_id):
    """
    Downloads Transcript from YouTube video and returns it in a DataFrame.
    :param video_id: String of YouTube video ID
    :return: Pandas DataFrame of Transcript
    """
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["de"])
    transcript = pd.DataFrame(transcript)
    transcript["text"] = transcript["text"] + " "
    return transcript["text"].sum()


def load_filter():
    nlp = spacy.load("de_core_news_sm")
    filterwords = spacy.lang.de.stop_words.STOP_WORDS
    with open("docs/filterwords.txt", encoding="utf-8", errors="ignore") as d:
        filterwords.update(d.read().split())
    with open("docs/german_stopwords_full.txt", encoding="utf-8", errors="ignore") as d:
        filterwords.update(d.read().split()[53:])
    return list(set(filterwords))


def scrape(channel_id, to_csv=True, verbose=True):
    verboseprint = define_print(verbose=verbose)
    assert len(channel_id) == 24, "length of channel id should be 24"

    verboseprint("getting info on youtube channel with id " + channel_id + "...")
    channel = Channel.get(channel_id)
    medium = channel.get("title")

    verboseprint("retrieving videos of youtube channel " + medium + "...")
    playlist = Playlist(playlist_from_channel_id(channel["id"]))
    while playlist.hasMoreVideos:
        playlist.getNextVideos()

    verboseprint(f"getting transcripts for {len(playlist.videos)} videos...")
    transcript_dict = {
        "medium": [],
        "title": [],
        "id": [],
        "duration": [],
        "transcript": [],
    }
    for video in tqdm(playlist.videos):
        except_counter = 0
        try:
            transcript_dict["transcript"].append(get_transcript(video["id"]))
            transcript_dict["medium"].append(medium)
            transcript_dict["title"].append(video.get("title"))
            transcript_dict["id"].append(video.get("id"))
            transcript_dict["duration"].append(video.get("duration"))
        except:
            except_counter += 1

    verboseprint(
        f"fetched transcripts for {len(playlist.videos) - except_counter} out of {len(playlist.videos)} videos."
    )

    df = pd.DataFrame(transcript_dict)
    if to_csv:
        verboseprint("saving to csv file in data/raw/ directory...")
        df.to_csv("data/raw/" + medium + "_raw.csv")

    return df


def lemmatize(text, nlp, filterwords):
    """
    tokenizes and lemmatizes german input text
    :param text: raw input text (german)
    :return: list of lemmatized tokens from input text
    """

    with nlp.select_pipes(enable="lemmatizer"):
        doc = nlp(text)
    lemmas = [token.lemma_.lower() for token in doc]
    lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma not in filterwords]
    return " ".join(lemmas)


def preprocess(df, to_csv=True, verbose=True):
    verboseprint = define_print(verbose=verbose)
    pandarallel.initialize(progress_bar=True)
    filterwords = load_filter()
    nlp = spacy.load("de_core_news_sm")

    verboseprint(f"lemmatizing transcript data of {len(df.index)} videos...")
    df["preprocessed"] = df["transcript"].parallel_apply(
        lemmatize, args=(nlp, filterwords)
    )

    if to_csv:
        df.to_csv("data/preprocessed/" + df.iloc[0]["medium"] + "_preprocessed.csv")
    return df


def create_samples(dfs, n_samples=[10, 50, 100, 300], to_csv=True):
    df_list = []
    for n in tqdm(n_samples):
        data = pd.DataFrame()
        for df in dfs:
            tmp = df.sample(n=n, random_state=42)
            data = pd.concat([data, tmp])
        data = shuffle(data, random_state=42)
        if to_csv:
            data.to_csv("data//samples//sample" + str(n) + ".csv")
        df_list.append(data)
    return df_list


def extract_topics(df, to_csv=True, verbose=True):
    verboseprint = define_print(verbose=verbose)
    df.dropna(inplace=True)

    verboseprint("loading cv model...")
    cv_model = joblib.load("data/cv_model.pkl")

    verboseprint("loading lda model...")
    lda_model = joblib.load("data/lda_model.pkl")

    verboseprint(f"getting topics for {df.shape[0]} videos...")
    lda = cv_model.transform(df["preprocessed"].to_list())
    lda = lda_model.transform(lda)
    lda = pd.DataFrame(lda)
    lda.rename(columns=topic_dict, inplace=True)
    lda["dominant topic"] = [
        topic_dict[topic] for topic in np.argmax(lda.values, axis=1)
    ]
    lda["id"] = df["id"].to_list()

    verboseprint("merging data...")
    df = df.merge(lda, how="outer", on="id")
    if to_csv:
        verboseprint("saving csv file...")
        df.to_csv("data/labeled/" + df["medium"].iloc[0] + "_labeled.csv")
    return df


def sort_topics(dfs, to_csv=True, verbose=True):
    verboseprint = define_print(verbose=verbose)
    dfs_dict = {}

    verboseprint('initializing dataframes...')
    for _, topic in topic_dict.items():
        dfs_dict[topic] = pd.DataFrame()

    verboseprint(f'iterating through {len(dfs)} input dataframes...')
    for df in dfs:
        verboseprint('sorting ' + df['medium'].iloc[0] + 'dataframe by topic...')
        for _, topic in topic_dict.items():
            dfs_dict[topic] = pd.concat([dfs_dict[topic], df[df['dominant topic'] == topic]])
    
    if to_csv:
        verboseprint('saving csv files...')
        for _, topic in topic_dict.items():
            dfs_dict[topic].to_csv('data/sorted/'+topic+'.csv')
    
    return dfs_dict


def get_N_matrix(topic, verbose=True):
    verboseprint = define_print(verbose=verbose)
    MEDIA = ['NachDenkSeiten', 'DER SPIEGEL', 'ZDFheute Nachrichten', 'BILD', 'Junge Freiheit']
    cv = CountVectorizer(max_df=0.9, min_df=10, max_features=10000, ngram_range=(1,3))

    verboseprint('importing dataframe with topic ' + topic + ' and fitting model...')
    df = pd.read_csv('data/sorted/'+topic+'.csv', index_col=0)
    cv.fit(df['preprocessed'])

    verboseprint('restructuring dataframe with ' + str(len(df)) + ' transcripts...')
    df['preprocessed'] = df['preprocessed'] + ' '
    df = df[['medium', 'preprocessed', 'dominant topic']]
    df_grouped = df.groupby(['medium', 'dominant topic']).sum()

    df = pd.DataFrame(index=MEDIA, columns=['preprocessed'])
    for medium in MEDIA:
        df.loc[medium] = df_grouped.loc[medium].loc[topic]['preprocessed']

    verboseprint('counting n-gram occurences...')
    N_matrix = cv.transform(df['preprocessed'].values)

    verboseprint('exporting dataframe...')
    N_df = pd.DataFrame(data=N_matrix.toarray().transpose(),
                        columns=df.index,
                        index=cv.get_feature_names_out())

    return N_df