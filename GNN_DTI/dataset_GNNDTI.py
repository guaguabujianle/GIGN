# %%
import os
from torch.utils.data import Dataset, DataLoader
from utils import atom_feature
import numpy as np
import torch
import random
from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import pickle
import multiprocessing
from itertools import repeat
import pandas as pd

random.seed(0)

# %%
def get_atom_feature(m, is_ligand=True):
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        H.append(atom_feature(m, i, None, None))
    H = np.array(H)        
    if is_ligand:
        H = np.concatenate([H, np.zeros((n,28))], 1)
    else:
        H = np.concatenate([np.zeros((n,28)), H], 1)
    return H      

def mols2graphs(complex_path, label, save_path, dis_threshold=5):
    key = complex_path.split('/')[-2]

    with open(complex_path, 'rb') as f:
        m1, m2 = pickle.load(f)

    #prepare ligand
    n1 = m1.GetNumAtoms()
    c1 = m1.GetConformers()[0]
    d1 = np.array(c1.GetPositions())
    adj1 = GetAdjacencyMatrix(m1)+np.eye(n1)
    H1 = get_atom_feature(m1, True)

    #prepare protein
    n2 = m2.GetNumAtoms()
    c2 = m2.GetConformers()[0]
    d2 = np.array(c2.GetPositions())
    adj2 = GetAdjacencyMatrix(m2)+np.eye(n2)
    H2 = get_atom_feature(m2, False)

    #aggregation
    H = np.concatenate([H1, H2], 0)
    agg_adj1 = np.zeros((n1+n2, n1+n2))
    agg_adj1[:n1, :n1] = adj1
    agg_adj1[n1:, n1:] = adj2
    agg_adj2 = np.copy(agg_adj1)
    dm = distance_matrix(d1,d2)
    agg_adj2[:n1,n1:] = np.copy(dm)
    agg_adj2[n1:,:n1] = np.copy(np.transpose(dm))

    #node indice for aggregation
    valid = np.zeros((n1+n2,))
    valid[:n1] = 1

    Y = label

    sample = {
                'H':H, \
                'A1': agg_adj1, \
                'A2': agg_adj2, \
                'Y': Y, \
                'V': valid, \
                'key': key, \
                }
                
    torch.save(sample, save_path)

# %%
class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, dis_threshold=5, graph_type='Graph_GNNDTI', num_process=48, create=False):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self._pre_process()

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type
        dis_thresholds = repeat(self.dis_threshold, len(data_df))

        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
        for i, row in data_df.iterrows():
            cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
            complex_dir = os.path.join(data_dir, cid)
            graph_path = os.path.join(complex_dir, f"{graph_type}-{cid}_{self.dis_threshold}A.th")
            complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(cid)
            pKa_list.append(pKa)
            graph_path_list.append(graph_path)

        if self.create:
            print('Generate complex graph...')
            # multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, pKa_list, graph_path_list, dis_thresholds))
            pool.close()
            pool.join()

        self.graph_paths = graph_path_list
        self.complex_ids = complex_id_list

    def __getitem__(self, idx):
        return torch.load(self.graph_paths[idx])

    def collate_fn(self, batch):
        max_natoms = max([len(item['H']) for item in batch if item is not None])
        
        H = np.zeros((len(batch), max_natoms, 56))
        A1 = np.zeros((len(batch), max_natoms, max_natoms))
        A2 = np.zeros((len(batch), max_natoms, max_natoms))
        Y = np.zeros((len(batch),))
        V = np.zeros((len(batch), max_natoms))
        keys = []
        
        for i in range(len(batch)):
            natom = len(batch[i]['H'])
            
            H[i,:natom] = batch[i]['H']
            A1[i,:natom,:natom] = batch[i]['A1']
            A2[i,:natom,:natom] = batch[i]['A2']
            Y[i] = batch[i]['Y']
            V[i,:natom] = batch[i]['V']
            keys.append(batch[i]['key'])

        H = torch.from_numpy(H).float()
        A1 = torch.from_numpy(A1).float()
        A2 = torch.from_numpy(A2).float()
        Y = torch.from_numpy(Y).float()
        V = torch.from_numpy(V).float()
        
        return H, A1, A2, Y, V, keys

    def __len__(self):
        return len(self.data_df)



if __name__ == '__main__':
    data_root = './data'
    toy_dir = os.path.join(data_root, 'toy_set')
    toy_df = pd.read_csv(os.path.join(data_root, "toy_examples.csv"))
    toy_set = GraphDataset(toy_dir, toy_df, graph_type='Graph_GNNDTI', dis_threshold=5, create=True)

# %%
