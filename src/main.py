from utils import preprocess, load_filter
import pandas as pd


def main():
    
    df = pd.read_csv('data/raw/Junge Freiheit_raw.csv')
    preprocess(df)


if __name__ == "__main__":
    main()
