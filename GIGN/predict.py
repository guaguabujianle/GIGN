
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pandas as pd
import torch
from GIGN import GIGN
from dataset_GIGN import GraphDataset, PLIDataLoader
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error

# %%
def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            label = data.y

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return rmse, coff
    
# %%
data_root = './data'
graph_type = 'Graph_GIGN'
batch_size = 128

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

valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=4)
test2013_loader = PLIDataLoader(test2013_set, batch_size=batch_size, shuffle=False, num_workers=4)
test2019_loader = PLIDataLoader(test2019_set, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device('cuda:0')
model = GIGN(35, 256).to(device)
load_model_dict(model, './model/20221121_074758_GIGN_repeat0/model/epoch-532, train_loss-0.1162, train_rmse-0.3408, valid_rmse-1.1564, valid_pr-0.7813.pt')
model = model.cuda()

valid_rmse, valid_coff = val(model, valid_loader, device)
test2013_rmse, test2013_coff = val(model, test2013_loader, device)
test2016_rmse, test2016_coff = val(model, test2016_loader, device)
test2019_rmse, test2019_coff = val(model, test2019_loader, device)

msg = "valid_rmse-%.4f, valid_r-%.4f, test2013_rmse-%.4f, test2013_r-%.4f, test2016_rmse-%.4f, test2016_r-%.4f, test2019_rmse-%.4f, test2019_r-%.4f" \
    % (valid_rmse, valid_coff, test2013_rmse, test2013_coff, test2016_rmse, test2016_coff, test2019_rmse, test2019_coff)

print(msg)
# %%
