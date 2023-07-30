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

    train_random_df = pd.read_csv(os.path.join('./data', 'train_random.csv'))
    valid_random_df = pd.read_csv(os.path.join('./data', 'valid_random.csv'))
    test_random_df = pd.read_csv(os.path.join('./data', 'test_random.csv'))

    train_random_feats, train_random_labels = load_dataset(feature_root, train_random_df)
    valid_random_feats, valid_random_labels = load_dataset(feature_root, valid_random_df)
    test_random_feats, test_random_labels = load_dataset(feature_root, test_random_df)

    performance_dict = defaultdict(list)

    # three independent run
    for i in range(20):
        # n_estimators=500, max_features=5 according to the RF-Score paper
        # min_samples_split=6 according to the OODT
        model = RandomForestRegressor(n_estimators=500, max_features=5, n_jobs=20, min_samples_split=6)
        model.fit(train_random_feats, train_random_labels)
        pred_random = model.predict(test_random_feats)

        pr = np.corrcoef(pred_random, test_random_labels)[0, 1]
        rmse = np.sqrt(mean_squared_error(test_random_labels, pred_random))

        performance_dict['rmse'].append(rmse)
        performance_dict['pr'].append(pr)

    performance_df = pd.DataFrame(performance_dict)
    performance_df.to_csv('./results_20runs/test_random.csv', index=False)

# %%
