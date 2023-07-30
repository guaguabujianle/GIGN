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

    train_scaffold_df = pd.read_csv(os.path.join('./data', 'train_scaffold.csv'))
    valid_scaffold_df = pd.read_csv(os.path.join('./data', 'valid_scaffold.csv'))
    test_scaffold_df = pd.read_csv(os.path.join('./data', 'test_scaffold.csv'))

    train_scaffold_feats, train_scaffold_labels = load_dataset(feature_root, train_scaffold_df)
    valid_scaffold_feats, valid_scaffold_labels = load_dataset(feature_root, valid_scaffold_df)
    test_scaffold_feats, test_scaffold_labels = load_dataset(feature_root, test_scaffold_df)

    performance_dict = defaultdict(list)

    # three independent run
    for i in range(20):
        # n_estimators=500, max_features=5 according to the RF-Score paper
        # min_samples_split=6 according to the OODT
        model = RandomForestRegressor(n_estimators=500, max_features=5, n_jobs=20, min_samples_split=6)
        model.fit(train_scaffold_feats, train_scaffold_labels)
        pred_scaffold = model.predict(test_scaffold_feats)

        pr = np.corrcoef(pred_scaffold, test_scaffold_labels)[0, 1]
        rmse = np.sqrt(mean_squared_error(test_scaffold_labels, pred_scaffold))

        performance_dict['rmse'].append(rmse)
        performance_dict['pr'].append(pr)

    performance_df = pd.DataFrame(performance_dict)
    performance_df.to_csv('./results_20runs/test_scaffold.csv', index=False)

# %%
