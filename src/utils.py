from youtube_transcript_api import YouTubeTranscriptApi
from youtubesearchpython import Playlist, Channel, playlist_from_channel_id
import pandas as pd
from tqdm import tqdm


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
