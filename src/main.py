from utils import extract_topics
import pandas as pd


def main():
    df = pd.read_csv("data/preprocessed/Junge Freiheit_preprocessed.csv", index_col=0)
    extract_topics(df)


if __name__ == "__main__":
    main()
