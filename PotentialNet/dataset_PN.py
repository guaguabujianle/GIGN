# %%
from functools import partial
import dgl
import numpy as np
import torch
import pandas as pd
import dgl.backend as F
import multiprocessing
import os
from functools import partial
import dgl.backend as F
from dgl import graph, batch
from dgllife.utils.mol_to_graph import k_nearest_neighbors, mol_to_bigraph
from dgllife.utils.featurizers import BaseAtomFeaturizer, BaseBondFeaturizer, ConcatFeaturizer, atom_type_one_hot, atom_total_degree_one_hot, atom_formal_charge_one_hot, atom_is_aromatic, atom_implicit_valence_one_hot, atom_explicit_valence_one_hot, bond_type_one_hot, bond_is_in_ring
torch.multiprocessing.set_sharing_strategy('file_system')
import pickle
import warnings
warnings.filterwarnings('ignore')

def mols2graphs(dir, label, save_path):
    """Graph construction and featurization for `PotentialNet for Molecular Property Prediction
     <https://pubs.acs.org/doi/10.1021/acscentsci.8b00507>`__.

    Parameters
    ----------
    ligand_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.
    protein_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.
    ligand_coordinates : Float Tensor of shape (V1, 3)
        Atom coordinates in a ligand.
    protein_coordinates : Float Tensor of shape (V2, 3)
        Atom coordinates in a protein.
    max_num_ligand_atoms : int or None
        Maximum number of atoms in ligands for zero padding, which should be no smaller than
        ligand_mol.GetNumAtoms() if not None. If None, no zero padding will be performed.
        Default to None.
    max_num_protein_atoms : int or None
        Maximum number of atoms in proteins for zero padding, which should be no smaller than
        protein_mol.GetNumAtoms() if not None. If None, no zero padding will be performed.
        Default to None.
    max_num_neighbors : int
        Maximum number of neighbors allowed for each atom when constructing KNN graph. Default to 4.
    distance_bins : list of float
        Distance bins to determine the edge types.
        Edges of the first edge type are added between pairs of atoms whose distances are less than `distance_bins[0]`.
        The length matches the number of edge types to be constructed.
        Default `[1.5, 2.5, 3.5, 4.5]`.
    strip_hydrogens : bool
        Whether to exclude hydrogen atoms. Default to False.

    Returns
    -------
    complex_bigraph : DGLGraph
        Bigraph with the ligand and the protein (pocket) combined and canonical features extracted.
        The atom features are stored as DGLGraph.ndata['h'].
        The edge types are stored as DGLGraph.edata['e'].
        The bigraphs of the ligand and the protein are batched together as one complex graph.
    complex_knn_graph : DGLGraph
        K-nearest-neighbor graph with the ligand and the protein (pocket) combined and edge features extracted based on distances.
        The edge types are stored as DGLGraph.edata['e'].
        The knn graphs of the ligand and the protein are batched together as one complex graph.

    """
    # try:
    max_num_ligand_atoms = None
    max_num_protein_atoms = None
    max_num_neighbors = 4
    # distance_bins = [1.5, 2.5]
    distance_bins = [1.5, 2.5, 3.5, 4.5]

    with open(dir, 'rb') as f:
        ligand_mol, protein_mol = pickle.load(f)

    if max_num_ligand_atoms is not None:
        assert max_num_ligand_atoms >= ligand_mol.GetNumAtoms(), \
            'Expect max_num_ligand_atoms to be no smaller than ligand_mol.GetNumAtoms(), ' \
            'got {:d} and {:d}'.format(max_num_ligand_atoms, ligand_mol.GetNumAtoms())
    if max_num_protein_atoms is not None:
        assert max_num_protein_atoms >= protein_mol.GetNumAtoms(), \
            'Expect max_num_protein_atoms to be no smaller than protein_mol.GetNumAtoms(), ' \
            'got {:d} and {:d}'.format(max_num_protein_atoms, protein_mol.GetNumAtoms())

    # Node featurizer for stage 1
    atoms = ['H','N','O','C','P','S','F','Br','Cl','I','Fe','Zn','Mg','Na','Mn','Ca','Co','Ni','Se','Cu','Cd','Hg','K']
    atom_total_degrees = list(range(5))
    atom_formal_charges = [-1, 0, 1]
    atom_implicit_valence = list(range(4))
    atom_explicit_valence = list(range(8))
    atom_concat_featurizer = ConcatFeaturizer([partial(atom_type_one_hot, allowable_set=atoms), 
                                            partial(atom_total_degree_one_hot, allowable_set=atom_total_degrees),
                                            partial(atom_formal_charge_one_hot, allowable_set=atom_formal_charges),
                                            atom_is_aromatic,
                                            partial(atom_implicit_valence_one_hot, allowable_set=atom_implicit_valence),
                                            partial(atom_explicit_valence_one_hot, allowable_set=atom_explicit_valence)])
    PN_atom_featurizer = BaseAtomFeaturizer({'h': atom_concat_featurizer})

    # Bond featurizer for stage 1
    bond_concat_featurizer = ConcatFeaturizer([bond_type_one_hot, bond_is_in_ring])
    PN_bond_featurizer = BaseBondFeaturizer({'e': bond_concat_featurizer})

    # construct graphs for stage 1
    ligand_bigraph = mol_to_bigraph(ligand_mol, add_self_loop=False,
                                    node_featurizer=PN_atom_featurizer,
                                    edge_featurizer=PN_bond_featurizer,
                                    canonical_atom_order=False) # Keep the original atomic order)
    protein_bigraph = mol_to_bigraph(protein_mol, add_self_loop=False,
                                    node_featurizer=PN_atom_featurizer,
                                    edge_featurizer=PN_bond_featurizer,
                                    canonical_atom_order=False)
    complex_bigraph = batch([ligand_bigraph, protein_bigraph])

    # Construct knn graphs for stage 2
    ligand_coordinates = ligand_mol.GetConformers()[0].GetPositions()
    protein_coordinates = protein_mol.GetConformers()[0].GetPositions()

    complex_coordinates = np.concatenate([ligand_coordinates, protein_coordinates])
    complex_srcs, complex_dsts, complex_dists = k_nearest_neighbors(
            complex_coordinates, distance_bins[-1], max_num_neighbors)
    complex_srcs = np.array(complex_srcs)
    complex_dsts = np.array(complex_dsts)
    complex_dists = np.array(complex_dists)

    complex_knn_graph = graph((complex_srcs, complex_dsts), num_nodes=len(complex_coordinates))
    d_features = np.digitize(complex_dists, bins=distance_bins, right=True)
    d_one_hot = int_2_one_hot(d_features)
    
    # add bond types and bonds (from bigraph) to stage 2
    u, v = complex_bigraph.edges()    
    complex_knn_graph.add_edges(u.to(F.int64), v.to(F.int64))
    n_d, f_d = d_one_hot.shape
    n_e, f_e = complex_bigraph.edata['e'].shape
    complex_knn_graph.edata['e'] = F.zerocopy_from_numpy(
        np.block([
            [d_one_hot, np.zeros((n_d, f_e))],
            [np.zeros((n_e, f_d)), np.array(complex_bigraph.edata['e'])]
        ]).astype(np.long)
    )
    torch.save((ligand_bigraph, protein_bigraph, complex_knn_graph, torch.FloatTensor([label])), save_path)

def int_2_one_hot(a):
    """Convert integer encodings on a vector to a matrix of one-hot encoding"""
    n = len(a)
    b = np.zeros((n, 4))
    b[np.arange(n), a] = 1
    return b   


def collate_fn(data_batch):
    """
    used for dataset generated from GraphDatasetV2MulPro class
    :param data_batch:
    :return:
    """
    graph_l, graph_p, graph_lp, label = map(list, zip(*data_batch))
    bg_list = []
    for l, p in zip(graph_l, graph_p):
        bg_list.append(l)
        bg_list.append(p)
    bg = dgl.batch(bg_list)
    bg_lp = dgl.batch(graph_lp)
    y = torch.cat(label, dim=0)
    return bg, bg_lp, y

# %%
class GraphDataset(object):
    """
    This class is used for generating graph objects using multi process
    """

    def __init__(self, data_dir, data_df, dis_threshold=5, graph_type='Graph_PN', num_process=48, create=True):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process=num_process

        self._pre_process()

    def _pre_process(self):

        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type

        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
        for i, row in data_df.iterrows():
            cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
            complex_dir = os.path.join(data_dir, cid)
            graph_path = os.path.join(complex_dir, f"{graph_type}-{cid}_{self.dis_threshold}A.dgl")
            complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(cid)
            pKa_list.append(pKa)
            graph_path_list.append(graph_path)

        if self.create:
            print('Generate complex graph...')
            # multiprocessing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, pKa_list, graph_path_list))
            pool.close()
            pool.join()

        self.graph_paths = graph_path_list
        self.complex_ids = complex_id_list

    def __getitem__(self, idx):
        return torch.load(self.graph_paths[idx])

    def __len__(self):
        return len(self.data_df)

if __name__ == '__main__':
    data_root = './data'
    toy_dir = os.path.join(data_root, 'toy_set')
    toy_df = pd.read_csv(os.path.join(data_root, "toy_examples.csv"))
    toy_set = GraphDataset(toy_dir, toy_df, graph_type='Graph_PN', dis_threshold=5, create=True)
# %%