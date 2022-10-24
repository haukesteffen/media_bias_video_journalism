import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from pandarallel import pandarallel
from youtube_transcript_api import YouTubeTranscriptApi
from youtubesearchpython import Playlist, Channel, Video, playlist_from_channel_id
from tqdm import tqdm
from utils import filter_N_by_information_score, get_N_matrix, preprocess, load_filter
tqdm.pandas()



def fetch_video_transcript(video_id):
    try:
        return YouTubeTranscriptApi.get_transcript(video_id=video_id, languages=["de"])
    except:
        return np.nan

def fetch_video_info(video_id):
    info = Video.getInfo(video_id)
    return [info["title"], info["duration"]["secondsText"], info["publishDate"], info["description"], info["category"]]

def get_raw_df(channel_id):
    pandarallel.initialize(progress_bar=True)
    df = pd.DataFrame(columns=["medium","title","id","duration","transcript","date","description","category"])
    print('fetching videos...')
    channel = Channel.get(channel_id)
    playlist = Playlist(playlist_from_channel_id(channel["id"]))
    while playlist.hasMoreVideos:
        playlist.getNextVideos()
    print('fetching metadata and transcripts...')
    df['id'] = [video.get("id") for video in tqdm(playlist.videos)]
    df['transcript'] = df['id'].parallel_apply(fetch_video_transcript)
    df['medium'] = channel.get("title")
    df.loc[:, ["title","duration","date","description","category"]] = df['id'].parallel_apply(fetch_video_info).to_list()
    return df

def transcript_by_minute(transcript):
    transcript_by_minute = {}
    for segment in transcript:
        minute = int(np.floor(segment['start']/60.0))
        if minute not in transcript_by_minute:
            transcript_by_minute[minute] = ''
        segment['minute'] = minute
        transcript_by_minute[segment['minute']] += (segment['text'] + ' ')
    return transcript_by_minute

def get_minutewise_df(df):
    df = df.dropna(subset=['transcript'])
    df['transcript_by_minute'] = df['transcript'].parallel_apply(transcript_by_minute)
    temp_df = pd.DataFrame([*df['transcript_by_minute']], df.index).stack()\
      .rename_axis([None,'minute']).reset_index(1, name='transcript')
    new_df = pd.concat([temp_df, df.drop(columns=['transcript', 'transcript_by_minute'])], join='outer', axis=1)
    new_df = new_df[['medium', 'id', 'title', 'description', 'duration', 'date', 'category', 'minute', 'transcript']]
    return new_df

def get_topic_df(df, save_model=True):
    stop_words = frozenset(load_filter())
    df.dropna(subset=['transcript'], inplace=True)
    docs = df['transcript'].astype(str).to_numpy()
    vectorizer_model = CountVectorizer(stop_words=stop_words, ngram_range=(1,1))
    topic_model = BERTopic(vectorizer_model = vectorizer_model, verbose=1, language='multilingual', min_topic_size=500)
    topics, _ = topic_model.fit_transform(docs)
    topic_info = topic_model.get_topic_info()
    topic_dict = pd.Series(topic_info.Name.values,index=topic_info.Topic).to_dict()
    df['topic'] = topics
    df['topic'] = df['topic'].apply(lambda row: topic_dict[row])
    if save_model:
        topic_model.save('assets/bertopic_model')
    return df

channels = {
    'junge Welt': 'UC_wVoja0mx0IFOP8s1nfRSg',
    'ZEIT ONLINE': 'UC_YnP7DDnKzkkVl3BaHiZoQ',
    'faz': 'UCcPcua2PF7hzik2TeOBx3uw',
    'Süddeutsche Zeitung': 'UC6bx8B0W0x_5NQFAF3Nbd-A',
    'NZZ Neue Zürcher Zeitung': 'UCK1aTcR0AckQRLTlK0c4fuQ',
    'WELT': 'UCZMsvbAhhRblVGXmEXW8TSA',
    'Bayerischer Rundfunk': 'UCZuFrqyZWfw_Zf0OnXWUXyQ',
    'Der Tagesspiegel': 'UCFemltyr6criZZsWFHUSHPQ',
    'Tagesschau': 'UC5NOEUbkLheQcaaRldYW5GA',
    'ARD': 'UCqmQ1b96-PNH4coqgHTuTlA',
    'ZDFinfo Dokus & Reportagen': 'UC7FeuS5wwfSR9IwOPkBV7SQ',
    'ZDF': 'UC_EnhS-vNDc6Eo9nyQHC2YQ',
    'ntv Nachrichten': 'UCSeil5V81-mEGB1-VNR7YEA',
    'stern TV': 'UC2cJbBzyHM48MVFB6eOW9og',
    'RTL': 'UC2w2teNMpadicMg3Sd_yiyg',
    'FOCUS Online': 'UCgAPgHNmQSG_ySHRiOVeF4Q',
    'COMPACTTV': 'UCgvFsn6bRKqND1cW3HpzDrA',
    'taz': 'UCPzGYQqM_lZ3mJvi89SF6mg',
    'NachDenkSeiten': 'UCE7b8qctaEGmST38-sfdOsA',
    'DER SPIEGEL': 'UC1w6pNGiiLdZgyNpXUnA4Zw',
    'ZDFheute Nachrichten': 'UCeqKIgPQfNInOswGRWt48kQ',
    'BILD': 'UC4zcMHyrT_xyWlgy5WGpFFQ',
    'Junge Freiheit': 'UCXJBRgiZRZvfilIGQ4wN5CQ',
}

def main():
    '''data_df = pd.DataFrame()
    for name, id in channels.items():
        raw_df = get_raw_df(id)
        minutewise_df = get_minutewise_df(raw_df)
        data_df = pd.concat([data_df, minutewise_df], axis=0)'''
    data_df = pd.read_pickle('data/data.pkl')
    topic_df = get_topic_df(data_df)
    topic_df.to_pickle('data/topic.pkl')
    #preprocessed_df = preprocess(topic_df)
    #preprocessed_df.to_pickle('data/data.pkl')

if __name__ == "__main__":
    main() 