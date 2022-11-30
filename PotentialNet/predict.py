
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import pandas as pd
from dataset_PN import GraphDataset, collate_fn
import numpy as np
from utils import *
from dgllife.model import PotentialNet
from torch.utils.data import DataLoader
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
        bg, bg_lp, label = data
        bg, bg_lp, label = bg.to(device), bg_lp.to(device), label.to(device)

        with torch.no_grad():
            pred = model(bg, bg_lp).view(-1)
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
    graph_type = 'Graph_PN'

    valid_dir = os.path.join(data_root, 'valid')
    test2013_dir = os.path.join(data_root, 'test2013')
    test2016_dir = os.path.join(data_root, 'test2016')
    test2019_dir = os.path.join(data_root, 'test2019')

    valid_df = pd.read_csv(os.path.join(data_root, 'valid.csv'))
    test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
    test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))
    test2019_df = pd.read_csv(os.path.join(data_root, 'test2019.csv'))

    valid_set = GraphDataset(valid_dir, valid_df, graph_type=graph_type, create=False)
    test2013_set = GraphDataset(test2013_dir, test2013_df, graph_type=graph_type, create=False)
    test2016_set = GraphDataset(test2016_dir, test2016_df, graph_type=graph_type, create=False)
    test2019_set = GraphDataset(test2019_dir, test2019_df, graph_type=graph_type, create=False)

    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=8)
    test2013_loader = DataLoader(test2013_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=8)
    test2016_loader = DataLoader(test2016_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=8)
    test2019_loader = DataLoader(test2019_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=8)

    device = torch.device('cuda:0')
    model = PotentialNet(44, 48, 48, 48, 9, 2, 1, [48, 24], [0.25, 0.25, 0.25]).to(device)
    load_model_dict(model, './model/20221121_110321_PN_repeat0/model/epoch-404, train_loss-0.8771, train_rmse-0.9366, valid_rmse-1.3519, valid_pr-0.7052.pt')

    valid_rmse, valid_pr = val(model, valid_loader, device)
    test2013_rmse, test2013_pr = val(model, test2013_loader, device)
    test2016_rmse, test2016_pr = val(model, test2016_loader, device)
    test2019_rmse, test2019_pr = val(model, test2019_loader, device)
    msg = "valid_rmse-%.4f, valid_pr-%.4f, test2013_rmse-%.4f, test2013_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f, test2019_rmse-%.4f, test2019_pr-%.4f," \
                % (valid_rmse, valid_pr, test2013_rmse, test2013_pr, test2016_rmse, test2016_pr, test2019_rmse, test2019_pr)
    print(msg)


# %%
