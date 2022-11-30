
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import pandas as pd
from dataset_GNNDTI import GraphDataset, PLIDataLoader
import numpy as np
from utils import *
from GNNDTI import gnn
from sklearn.metrics import mean_squared_error
import warnings
import torch
import pandas as pd
warnings.filterwarnings('ignore')

# %%
def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        H, A1, A2, Y, V, keys = data
        H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device),\
                            Y.to(device), V.to(device)
        label = Y

        with torch.no_grad():
            pred = model.train_model((H, A1, A2, V))
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return rmse, coff

if __name__ == '__main__':
    data_root = './data'
    graph_type = 'Graph_GNNDTI'

    valid_dir = os.path.join(data_root, 'valid')
    test2013_dir = os.path.join(data_root, 'test2013')
    test2016_dir = os.path.join(data_root, 'test2016')
    test2019_dir = os.path.join(data_root, 'test2019')

    valid_df = pd.read_csv(os.path.join(data_root, 'valid.csv'))
    test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
    test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))
    test2019_df = pd.read_csv(os.path.join(data_root, 'test2019.csv'))

    valid_set = GraphDataset(valid_dir, valid_df, graph_type=graph_type, create=False)
    test2016_set = GraphDataset(test2016_dir, test2016_df, graph_type=graph_type, create=False)
    test2013_set = GraphDataset(test2013_dir, test2013_df, graph_type=graph_type, create=False)
    test2019_set = GraphDataset(test2019_dir, test2019_df, graph_type=graph_type, create=False)

    valid_loader = PLIDataLoader(valid_set, batch_size=128, shuffle=False, num_workers=4)
    test2016_loader = PLIDataLoader(test2016_set, batch_size=128, shuffle=False, num_workers=4)
    test2013_loader = PLIDataLoader(test2013_set, batch_size=128, shuffle=False, num_workers=4)
    test2019_loader = PLIDataLoader(test2019_set, batch_size=128, shuffle=False, num_workers=4)

    device = torch.device('cuda:0')
    model = gnn().to(device)
    load_model_dict(model, './model/20221121_154002_GNNDTI_repeat0/model/epoch-436, train_loss-1.3832, train_rmse-1.1761, valid_rmse-1.3019, valid_pr-0.7103.pt')

    valid_rmse, valid_pr = val(model, valid_loader, device)
    test2013_rmse, test2013_pr = val(model, test2013_loader, device)
    test2016_rmse, test2016_pr = val(model, test2016_loader, device)
    test2019_rmse, test2019_pr = val(model, test2019_loader, device)
    msg = "valid_rmse-%.4f, valid_pr-%.4f, test2013_rmse-%.4f, test2013_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f, test2019_rmse-%.4f, test2019_pr-%.4f," \
                % (valid_rmse, valid_pr, test2013_rmse, test2013_pr, test2016_rmse, test2016_pr, test2019_rmse, test2019_pr)
    print(msg)


# %%
