# %%
import os
import pandas as pd
import oddt
from oddt.scoring.descriptors import close_contacts_descriptor
from utils import write_pickle
import multiprocessing

# %%
class FeatureGenerator(object):
    """
    # Extract 36 features
    References
    ----------
    .. [1] Ballester PJ, Mitchell JBO. A machine learning approach to
        predicting protein-ligand binding affinity with applications to
        molecular docking. Bioinformatics. 2010;26: 1169-1175.
        doi:10.1093/bioinformatics/btq112
    """
    def __init__(self):
        cutoff = 12
        ligand_atomic_nums = [6, 7, 8, 9, 15, 16, 17, 35, 53]
        protein_atomic_nums = [6, 7, 8, 16]
        self.rfscore_descriptor_cc = close_contacts_descriptor(
        cutoff = cutoff,
        protein_types = protein_atomic_nums,
        ligand_types = ligand_atomic_nums)
    
    def build(self, ligand, protein):
        return self.rfscore_descriptor_cc.build(ligand, protein)

def complex2feature(fg, cid, ligand_path, protein_path, label):
    protein = next(oddt.toolkits.ob.readfile('pdb', protein_path))
    ligand = next(oddt.toolkits.ob.readfile('sdf', ligand_path))
    feature = fg.build(ligand, protein)
    write_pickle(os.path.join('./feature', f'{cid}.pkl'), (feature, label))


if __name__ == "__main__":
    # replace data_root with your own dir
    data_root = '/data2/yzd/docking/pdbbind2016_pyg'

    train_dir = os.path.join(data_root, 'train')
    test2013_dir = os.path.join(data_root, 'test2013')
    test2016_dir = os.path.join(data_root, 'test2016')
    test2019_dir = os.path.join(data_root, 'holdout2019')
    # replace csar_set1_dir and csar_set2_dir with your own dirs
    csar_set1_dir = '/data2/yzd/docking/CSAR_NRC_HiQ_Set/Structures/set1'
    csar_set2_dir = '/data2/yzd/docking/CSAR_NRC_HiQ_Set/Structures/set2'

    train_df = pd.read_csv(os.path.join('./data', 'train.csv'))
    valid_df = pd.read_csv(os.path.join('./data', 'valid.csv'))
    test2013_df = pd.read_csv(os.path.join('./data', 'test2013.csv'))
    test2016_df = pd.read_csv(os.path.join('./data', 'test2016.csv'))
    test2019_df = pd.read_csv(os.path.join('./data', 'test2019.csv'))

    csar_set1_df = pd.read_csv(os.path.join('./data', 'CSAR_set1.csv'))
    csar_set2_df = pd.read_csv(os.path.join('./data', 'CSAR_set2.csv'))
    
    fg = FeatureGenerator()
    data_dict = {}

    fg_list = []
    cid_list = []
    protein_paths = []
    ligand_paths = []
    label_list = []

    for data_type in ['train', 'valid', 'test2013', 'test2016', 'test2019', 'CSAR_set1', 'CSAR_set2']:
        if data_type == 'train':
            data_df = train_df
            data_dir = train_dir
        elif data_type == 'valid':
            data_df = valid_df
            data_dir = train_dir    
        elif data_type == 'test2013':
            data_df = test2013_df
            data_dir = test2013_dir    
        elif data_type == 'test2016':
            data_df = test2016_df
            data_dir = test2016_dir    
        elif data_type == 'test2019':
            data_df = test2019_df
            data_dir = test2019_dir    
        elif data_type == 'CSAR_set1':
            data_df = csar_set1_df
            data_dir = csar_set1_dir    
        elif data_type == 'CSAR_set2':
            data_df = csar_set2_df
            data_dir = csar_set2_dir    
    
        for i, row in data_df.iterrows():
            cid, pKa = row['pdbid'], float(row['-logKd/Ki'])

            if data_type == 'CSAR_set1' or data_type == 'CSAR_set2':
                number = str(row['number'])
                protein_path = os.path.join(data_dir, number, f'Protein.pdb')
                ligand_path = os.path.join(data_dir, number, f'Ligand.sdf')
            else:
                protein_path = os.path.join(data_dir, cid, f'{cid}_protein.pdb')
                ligand_path = os.path.join(data_dir, cid, f'{cid}_ligand.sdf')

            fg_list.append(fg)
            cid_list.append(cid)
            protein_paths.append(protein_path)
            ligand_paths.append(ligand_path)
            label_list.append(pKa)

    print("Number of samples: ", len(protein_paths))

    pool = multiprocessing.Pool(32)
    pool.starmap(complex2feature,
                    zip(fg_list, cid_list, ligand_paths, protein_paths, label_list))
    pool.close()
    pool.join()





# %%
