from utils import preprocess
import pandas as pd


def main():
    df = pd.read_csv("data/raw/BILD_raw.csv", index_col=0)
    preprocess(df)


if __name__ == "__main__":
    main()
