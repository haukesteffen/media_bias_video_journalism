from youtube_transcript_api import YouTubeTranscriptApi
from youtubesearchpython import *
import pandas as pd
from tqdm import tqdm

def getTranscript(video_id):
    '''
    Downloads Transcript from YouTube video and returns it in a DataFrame.
    :param video_id: String of YouTube video ID
    :return: Pandas DataFrame of Transcript
    '''
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['de', 'en'])
    transcript_df = pd.DataFrame(transcript)
    return transcript_df

def scrape(channels_df=pd.DataFrame(), to_csv=False):
    '''
    Scrape YouTube channels and return transcript data
    :param channels_df: Pandas DataFrame with columns 'name' and 'id' of YouTube channels to scrape
           to_csv: if true, saves DataFrames to {channel name}.csv in /data/raw folder
    :return: DataFrame of all scraped captions
    '''
    ### initializing captions DataFrame
    captions_dict = {'id':[], 'transcript':[], 'medium':[]}
    captions_df = pd.DataFrame(captions_dict)

    ### initializing DataFrame of youtube channels to scrape captions data from
    if channels_df.empty:
        channels_dict = {'name':['NachDenkSeiten', 'Spiegel', 'ZDFheute', 'BILD', 'Junge Freiheit'],
                        'id':['UCE7b8qctaEGmST38-sfdOsA', 'UC1w6pNGiiLdZgyNpXUnA4Zw', 'UCeqKIgPQfNInOswGRWt48kQ', 'UC4zcMHyrT_xyWlgy5WGpFFQ', 'UCXJBRgiZRZvfilIGQ4wN5CQ']}
        channels_df = pd.DataFrame(channels_dict)

    ### iterating over channels in input DataFrame
    for index, channel in channels_df.iterrows():
        print('getting info on youtube channel ' + channel['name'] + '...')
        playlist = Playlist(playlist_from_channel_id(channel['id']))

        ### retrieving video ids from channel id
        while playlist.hasMoreVideos:
            try:
                playlist.getNextVideos()
            except:
                print('could not get more videos')
        print(f'videos retrieved: {len(playlist.videos)}')
        print('getting video transcripts...')

        ### getting transcript from video ids
        transcript_dict = {'id':[], 'transcript':[], 'medium':[]}
        for video in tqdm(range(len(playlist.videos))):
            text = ''
            #print('getting transcript of video number ' + str(video) + ' with id ' + playlist.videos[video]['id'])
            try:
                captions = getTranscript(playlist.videos[video]['id'])
                captions = captions[captions['start'] >= 5.0] #start getting captions after 5s
                for line in captions['text'] :
                    text += line + ' '
                transcript_dict['id'].append(playlist.videos[video]['id'])
                transcript_dict['transcript'].append(text)
                transcript_dict['medium'].append(channel['name'])
            except:
                print('could not get transcript for video number ' + str(video) + ' with id ' + playlist.videos[video]['id'])

        ### converting to dataframe and saving as csv
        transcript_df = pd.DataFrame(transcript_dict)
        if to_csv:
            transcript_df.to_csv('data\\raw\\'+ channel['name'] + '.csv')

        ### concat new data
        captions_df = pd.concat([captions_df, transcript_df])
    return captions_df