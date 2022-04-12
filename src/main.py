from utils import scrape
import pandas as pd

def main():
    df = pd.DataFrame({'name':['Junge Freiheit'], 'id':['UCXJBRgiZRZvfilIGQ4wN5CQ']})
    test = scrape(channels_df = df, to_csv=True)
    print(test.head())

if __name__ == "__main__":
    main()