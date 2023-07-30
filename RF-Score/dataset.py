import os
import numpy as np
from utils import read_pickle

def load_dataset(feature_root, data_df):
    feature_list = []
    label_list = []
    for i, row in data_df.iterrows():
        cid = row['pdbid']
        feature_path = os.path.join(feature_root, f'{cid}.pkl')
        if os.path.exists(feature_path):
            feature, label = read_pickle(feature_path)
            feature_list.append(feature)
            label_list.append(label)

    features = np.concatenate(feature_list, axis=0)
    labels = np.array(label_list)

    return features, labels