import warnings
warnings.filterwarnings('ignore')

from utils import get_N_matrix, filter_N_by_information_score, sort_topics, extract_topics
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt
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
    topic = 'Wirtschaft'
    N_df = get_N_matrix(topic=topic)
    N_df = filter_N_by_information_score(N_df)
    scaler = StandardScaler()
    model = TruncatedSVD(n_components=3)

    N_scaled = scaler.fit_transform(N_df.values)
    N_df_trunc = model.fit_transform(N_scaled)

    sns.set(palette="coolwarm", style='whitegrid')
    sns.scatterplot(
        model.components_[0], 
        model.components_[1], 
        hue=N_df.columns,
        palette='coolwarm',
        ).set(title=f'Thema "{topic}" - Hauptachsen 0 und 1')
    plt.figure()
    sns.scatterplot(
        model.components_[1], 
        model.components_[2], 
        hue=N_df.columns,
        palette='coolwarm',
        ).set(title=f'Thema "{topic}" - Hauptachsen 1 und 2')
    plt.show()

if __name__ == "__main__":
    main()
