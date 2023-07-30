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

    train_df = pd.read_csv(os.path.join('./data', 'train.csv'))
    valid_df = pd.read_csv(os.path.join('./data', 'valid.csv'))
    test2013_df = pd.read_csv(os.path.join('./data', 'test2013.csv'))
    test2016_df = pd.read_csv(os.path.join('./data', 'test2016.csv'))
    test2019_df = pd.read_csv(os.path.join('./data', 'test2019.csv'))
    csar_set1_df = pd.read_csv(os.path.join('./data', 'CSAR_set1.csv'))
    csar_set2_df = pd.read_csv(os.path.join('./data', 'CSAR_set2.csv'))
    csar_df = pd.concat([csar_set1_df, csar_set2_df])

    train_feats, train_labels = load_dataset(feature_root, train_df)
    test2013_feats, test2013_labels = load_dataset(feature_root, test2013_df)
    test2016_feats, test2016_labels = load_dataset(feature_root, test2016_df)
    test2019_feats, test2019_labels = load_dataset(feature_root, test2019_df)
    csar_feats, csar_labels = load_dataset(feature_root, csar_df)

    performance_dict = defaultdict(list)

    for i in range(3):
        # n_estimators=500, max_features=5 according to the RF-Score paper
        # min_samples_split=6 according to the OODT
        model = RandomForestRegressor(n_estimators=500, max_features=5, n_jobs=20, min_samples_split=6)
        model.fit(train_feats, train_labels)
        pred2013 = model.predict(test2013_feats)
        pred2016 = model.predict(test2016_feats)
        pred2019 = model.predict(test2019_feats)
        predcsar = model.predict(csar_feats)

        pr2013 = np.corrcoef(pred2013, test2013_labels)[0, 1]
        rmse2013 = np.sqrt(mean_squared_error(test2013_labels, pred2013))

        pr2016 = np.corrcoef(pred2016, test2016_labels)[0, 1]
        rmse2016 = np.sqrt(mean_squared_error(test2016_labels, pred2016))

        pr2019 = np.corrcoef(pred2019, test2019_labels)[0, 1]
        rmse2019 = np.sqrt(mean_squared_error(test2019_labels, pred2019))

        prcsar = np.corrcoef(predcsar, csar_labels)[0, 1]
        rmsecsar = np.sqrt(mean_squared_error(csar_labels, predcsar))

        performance_dict['test2013_rmse'].append(rmse2013)
        performance_dict['test2013_pr'].append(pr2013)

        performance_dict['test2016_rmse'].append(rmse2016)
        performance_dict['test2016_pr'].append(pr2016)

        performance_dict['test2019_rmse'].append(rmse2019)
        performance_dict['test2019_pr'].append(pr2019)

        performance_dict['csar_rmse'].append(rmsecsar)
        performance_dict['csar_pr'].append(prcsar)

    performance_df = pd.DataFrame(performance_dict)
    performance_df.to_csv('./results/test.csv', index=False)
    
 # %%
