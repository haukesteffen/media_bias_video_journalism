from utils import create_samples
import pandas as pd


def main():
    df1 = pd.read_csv("data/preprocessed/BILD_preprocessed.csv", index_col=0)
    df2 = pd.read_csv("data/preprocessed/DER SPIEGEL_preprocessed.csv", index_col=0)
    df3 = pd.read_csv("data/preprocessed/ZDFheute Nachrichten_preprocessed.csv", index_col=0)
    df4 = pd.read_csv("data/preprocessed/Junge Freiheit_preprocessed.csv", index_col=0)
    df5 = pd.read_csv("data/preprocessed/NachDenkSeiten_preprocessed.csv", index_col=0)
    create_samples([df1, df2, df3, df4, df5])


if __name__ == "__main__":
    main()
