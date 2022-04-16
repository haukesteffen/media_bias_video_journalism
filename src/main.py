from utils import scrape

channels = [
    "UCE7b8qctaEGmST38-sfdOsA",
    "UC1w6pNGiiLdZgyNpXUnA4Zw",
    "UCeqKIgPQfNInOswGRWt48kQ",
    "UC4zcMHyrT_xyWlgy5WGpFFQ",
    "UCXJBRgiZRZvfilIGQ4wN5CQ",
]


def main():
    for channel in channels:
        scrape(channel)


if __name__ == "__main__":
    main()
