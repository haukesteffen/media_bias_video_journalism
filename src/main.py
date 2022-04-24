# from utils import extract_topics, sort_topics
from utils import get_N_matrix, filter_N_by_information_score
import pandas as pd


def main():
    """media = ['NachDenkSeiten', 'DER SPIEGEL', 'ZDFheute Nachrichten', 'BILD', 'Junge Freiheit']
    dfs = [pd.read_csv('data/preprocessed/'+medium+'_preprocessed.csv', index_col=0) for medium in media]
    for df in dfs:
        extract_topics(df)

    dfs = [pd.read_csv('data/labeled/'+medium+'_labeled.csv', index_col=0) for medium in media]
    sort_topics(dfs)"""

    N = filter_N_by_information_score(get_N_matrix("Ukrainekonflikt"))
    print(N)


if __name__ == "__main__":
    main()
