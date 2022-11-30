# %%
import os
import pandas as pd
import shutil
from glob import glob

# %%
def clean_dir():
    data_root = './data/toy_set'
    complex_path = glob(os.path.join(data_root, '*', '*_opt*')) + glob(os.path.join(data_root, '*', '????')) + glob(os.path.join(data_root, '*', '*ligand.pdb')) + glob(os.path.join(data_root, '*', 'Pocket8*')) \
        + glob(os.path.join(data_root, '*', 'Pocket_*')) + glob(os.path.join(data_root, '*', '????_5A')) + glob(os.path.join(data_root, '*', '????_10A')) + \
        glob(os.path.join(data_root, '*', '*.dgl')) + glob(os.path.join(data_root, '*', '*.rdkit')) + glob(os.path.join(data_root, '*', '*.pyg'))  + glob(os.path.join(data_root, '*', '*.th'))
    [os.remove(path) for path in complex_path]

clean_dir()

# %%
def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

src_root = '/data2/yzd/docking/pdbbind2016_pyg'
dst_root = './data'
src_train_dir = os.path.join(src_root, 'train')
src_test2013_dir = os.path.join(src_root, 'test2013')
src_test2016_dir = os.path.join(src_root, 'test2016')
src_test2019_dir = os.path.join(src_root, 'holdout2019')

dst_train_dir = os.path.join(dst_root, 'train')
dst_valid_dir = os.path.join(dst_root, 'valid')
dst_test2013_dir = os.path.join(dst_root, 'test2013')
dst_test2016_dir = os.path.join(dst_root, 'test2016')
dst_test2019_dir = os.path.join(dst_root, 'test2019')

create_dir([dst_train_dir, dst_valid_dir, dst_test2013_dir, dst_test2016_dir, dst_test2019_dir])

train_df = pd.read_csv('./data/train.csv')
valid_df = pd.read_csv('./data/valid.csv')
test2013_df = pd.read_csv('./data/test2013.csv')
test2016_df = pd.read_csv('./data/test2016.csv')
test2019_df = pd.read_csv('./data/test2019.csv')

def copy_files(src_data_dir, dst_data_dir, df, src_graph_type='Graph_GNNDTI', dst_graph_type='Graph_GNNDTI'):
    for i, row in df.iterrows():
        pdbid = row['pdbid']
        pKa = row['-logKd/Ki']
        src_path = os.path.join(src_data_dir, pdbid, f"{src_graph_type}-{pdbid}_5A.pyg")
        dst_path = os.path.join(dst_data_dir, pdbid, f"{dst_graph_type}-{pdbid}_5A.th")
        create_dir([os.path.join(dst_data_dir, pdbid)])

        if os.path.exists(dst_path):
            os.remove(dst_path)
            shutil.copy(src_path, dst_path)
        else:
            shutil.copy(src_path, dst_path)

copy_files(src_train_dir, dst_train_dir, train_df, src_graph_type='Graph_GNNDTI', dst_graph_type='Graph_GNNDTI')
copy_files(src_train_dir, dst_valid_dir, valid_df, src_graph_type='Graph_GNNDTI', dst_graph_type='Graph_GNNDTI')
copy_files(src_test2013_dir, dst_test2013_dir, test2013_df, src_graph_type='Graph_GNNDTI', dst_graph_type='Graph_GNNDTI')
copy_files(src_test2016_dir, dst_test2016_dir, test2016_df, src_graph_type='Graph_GNNDTI', dst_graph_type='Graph_GNNDTI')
copy_files(src_test2019_dir, dst_test2019_dir, test2019_df, src_graph_type='Graph_GNNDTI', dst_graph_type='Graph_GNNDTI')

# %%