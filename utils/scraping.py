from youtube_transcript_api import YouTubeTranscriptApi
from youtubesearchpython import *
import pandas as pd


def getTranscript(video_id):
    """
    Downloads Transcript from YouTube video and returns it in a DataFrame.
    :param video_id: String of YouTube video ID
    :return: Pandas DataFrame of Transcript
    """
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["de", "en"])
    transcript_df = pd.DataFrame(transcript)
    return transcript_df


### initializing DataFrame of youtube channels to scrape captions data from
channels_dict = {
    "name": ["NachDenkSeiten", "Spiegel", "ZDFheute", "BILD", "Junge Freiheit"],
    "id": [
        "UCE7b8qctaEGmST38-sfdOsA",
        "UC1w6pNGiiLdZgyNpXUnA4Zw",
        "UCeqKIgPQfNInOswGRWt48kQ",
        "UC4zcMHyrT_xyWlgy5WGpFFQ",
        "UCXJBRgiZRZvfilIGQ4wN5CQ",
    ],
}
channels_df = pd.DataFrame(channels_dict)

### iterating over channels in DataFrame
for index, channel in channels_df.iterrows():
    print("getting info on youtube channel " + channel["name"] + "...")
    playlist = Playlist(playlist_from_channel_id(channel["id"]))

    ### retrieving video ids from channel id
    while playlist.hasMoreVideos:
        try:
            playlist.getNextVideos()
        except:
            pass
    print(f"videos retrieved: {len(playlist.videos)}")

    ### getting transcripts from video id
    transcript_dict = {"id": [], "transcript": []}
    for video in range(len(playlist.videos)):
        text = ""
        print(
            "getting transcript of video number "
            + str(video)
            + " with id "
            + playlist.videos[video]["id"]
        )
        try:
            captions = getTranscript(playlist.videos[video]["id"])
            captions = captions[
                captions["start"] >= 5.0
            ]  # start getting captions after 5s
            for line in captions["text"]:
                text += line + " "
            transcript_dict["id"].append(playlist.videos[video]["id"])
            transcript_dict["transcript"].append(text)
        except:
            print(
                "could not get transcript for video number "
                + str(video)
                + " with id "
                + playlist.videos[video]["id"]
            )

    ### converting to dataframe and saving as csv
    transcript_df = pd.DataFrame(transcript_dict)
    print(transcript_df.head())
    transcript_df.to_csv("data\\raw\\" + channel["name"] + ".csv")
