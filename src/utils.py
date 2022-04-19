from youtube_transcript_api import YouTubeTranscriptApi
from youtubesearchpython import Playlist, Channel, playlist_from_channel_id
from sklearn.utils import shuffle
import pandas as pd
from tqdm import tqdm
import spacy
from pandarallel import pandarallel


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
    # tqdm.pandas()
    verboseprint = define_print(verbose=verbose)
    pandarallel.initialize(progress_bar=True)
    filterwords = load_filter()
    nlp = spacy.load("de_core_news_sm")

    verboseprint(f"lemmatizing transcript data of {len(df.index)} videos...")
    # df["preprocessed"] = df["transcript"].progress_apply(lemmatize)
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