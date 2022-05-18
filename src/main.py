import pandas as pd
import numpy as np
from pandarallel import pandarallel
from youtube_transcript_api import YouTubeTranscriptApi
from youtubesearchpython import Playlist, Channel, Video, Transcript, playlist_from_channel_id
from tqdm import tqdm
tqdm.pandas()
pandarallel.initialize(progress_bar=True)

def fetch_video_transcript(video_id):
    try:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=["de", 'en'])
    except:
        return np.nan

def fetch_video_info(video_id):
    info = Video.getInfo(video_id)
    return [info["title"], info["duration"]["secondsText"], info["publishDate"], info["description"], info["category"]]

def scrape_videos(channel_id):
    df = pd.DataFrame(columns=["medium","title","id","duration","transcript","date","description","category"])
    channel = Channel.get(channel_id)
    
    playlist = Playlist(playlist_from_channel_id(channel["id"]))
    while playlist.hasMoreVideos:
        playlist.getNextVideos()
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

def get_transcript_by_minute_df(df):
    df = df.dropna(subset=['transcript'])
    df['transcript_by_minute'] = df['transcript'].parallel_apply(transcript_by_minute)
    temp_df = pd.DataFrame([*df['transcript_by_minute']], df.index).stack()\
      .rename_axis([None,'minute']).reset_index(1, name='transcript')
    new_df = pd.concat([temp_df, df.drop(columns=['transcript', 'transcript_by_minute'])], join='outer', axis=1)
    new_df = new_df[['medium', 'id', 'title', 'description', 'duration', 'date', 'category', 'minute', 'transcript']]
    return new_df

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
    for name, id in channels.items():
        df = pd.read_pickle(f'data/raw/{name}.pkl')
        df_new = get_transcript_by_minute_df(df)
        df_new.to_pickle(f'data/topics_by_minute/{name}_topics_by_minute.pkl')

if __name__ == "__main__":
    main()
