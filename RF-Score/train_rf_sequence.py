# %%
import os
import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from dataset import load_dataset

if __name__ == '__main__':
    feature_root = './feature'

    train_sequence_df = pd.read_csv(os.path.join('./data', 'train_sequence_identity.csv'))
    valid_sequence_df = pd.read_csv(os.path.join('./data', 'valid_sequence_identity.csv'))
    test_sequence_df = pd.read_csv(os.path.join('./data', 'test_sequence_identity.csv'))

    train_sequence_feats, train_sequence_labels = load_dataset(feature_root, train_sequence_df)
    valid_sequence_feats, valid_sequence_labels = load_dataset(feature_root, valid_sequence_df)
    test_sequence_feats, test_sequence_labels = load_dataset(feature_root, test_sequence_df)

    performance_dict = defaultdict(list)

    # three independent run
    for i in range(3):
        # n_estimators=500, max_features=5 according to the RF-Score paper
        # min_samples_split=6 according to the OODT
        model = RandomForestRegressor(n_estimators=500, max_features=5, n_jobs=20, min_samples_split=6)
        model.fit(train_sequence_feats, train_sequence_labels)
        pred_sequence = model.predict(test_sequence_feats)

        pr = np.corrcoef(pred_sequence, test_sequence_labels)[0, 1]
        rmse = np.sqrt(mean_squared_error(test_sequence_labels, pred_sequence))

        performance_dict['rmse'].append(rmse)
        performance_dict['pr'].append(pr)

    performance_df = pd.DataFrame(performance_dict)
    performance_df.to_csv('./results/test_sequence.csv', index=False)

# %%
