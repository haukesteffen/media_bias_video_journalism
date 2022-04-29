from utils import sort_topics, extract_topics
import pandas as pd
import joblib


channels = {
    'junge Welt': 'UC_wVoja0mx0IFOP8s1nfRSg',
    'ZEIT ONLINE': 'UC_YnP7DDnKzkkVl3BaHiZoQ',
    'faz': 'UCcPcua2PF7hzik2TeOBx3uw',
    'Süddeutsche Zeitung': 'UC6bx8B0W0x_5NQFAF3Nbd-A',
    'NZZ Neue Zürcher Zeitung': 'UCK1aTcR0AckQRLTlK0c4fuQ',
    'WELT': 'UCZMsvbAhhRblVGXmEXW8TSA',
    'Bayerischer Rundfunk': 'UCZuFrqyZWfw_Zf0OnXWUXyQ',
    'Der Tagesspiegel': 'UCFemltyr6criZZsWFHUSHPQ',
    'Tagesschau': 'UC5NOEUbkLheQcaaRldYW5GA',
    'ARD': 'UCqmQ1b96-PNH4coqgHTuTlA',
    'ZDFinfo Dokus & Reportagen': 'UC7FeuS5wwfSR9IwOPkBV7SQ',
    'ZDF': 'UC_EnhS-vNDc6Eo9nyQHC2YQ',
    'ntv Nachrichten': 'UCSeil5V81-mEGB1-VNR7YEA',
    'stern TV': 'UC2cJbBzyHM48MVFB6eOW9og',
    'RTL': 'UC2w2teNMpadicMg3Sd_yiyg',
    'FOCUS Online': 'UCgAPgHNmQSG_ySHRiOVeF4Q',
    'COMPACTTV': 'UCgvFsn6bRKqND1cW3HpzDrA',
    'taz': 'UCPzGYQqM_lZ3mJvi89SF6mg',
    'NachDenkSeiten': 'UCE7b8qctaEGmST38-sfdOsA',
    'DER SPIEGEL': 'UC1w6pNGiiLdZgyNpXUnA4Zw',
    'ZDFheute Nachrichten': 'UCeqKIgPQfNInOswGRWt48kQ',
    'BILD': 'UC4zcMHyrT_xyWlgy5WGpFFQ',
    'Junge Freiheit': 'UCXJBRgiZRZvfilIGQ4wN5CQ',
}

def main():
    cv_model = joblib.load('data/cv_model.pkl')
    lda_model = joblib.load('data/lda_model.pkl')
    dfs = []
    for channel, _ in channels.items():
        df = pd.read_csv('data/preprocessed/'+channel+'_preprocessed.csv', index_col=0)
        dfs.append(df)
        extract_topics(df, cv_model=cv_model, lda_model=lda_model)
    sort_topics(dfs)

if __name__ == "__main__":
    main()
