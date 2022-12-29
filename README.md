## Note:
This project contains several GNN-based models for protein-ligand binding affinity prediction, which are mainly taken from  
PotentialNet: https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/model_zoo/potentialnet.py                                  
GNN_DTI: https://github.com/jaechanglim/GNN_DTI  
IGN: https://github.com/zjujdj/InteractionGraphNet/tree/master  
SchNet: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/schnet.html  
EGNN: https://github.com/vgsatorras/egnn  
Each baseline folder has a requirement.txt listing the version of the packages.  

## Dataset:
All data used in this paper are publicly available and can be accessed here:  
- PDBbind v2016 and v2019: http://www.pdbbind.org.cn/download.php  
- 2013 and 2016 core sets: http://www.pdbbind.org.cn/casf.php  
You can also download the processed data from

## Requirements:
matplotlib==3.3.4  
networkx==2.5  
numpy==1.19.2  
pandas==1.1.5  
pymol==0.1.0  
rdkit==2022.9.2  
scikit_learn==1.1.3  
scipy==1.5.2  
seaborn==0.11.2  
torch==1.10.2  
torch_geometric==2.0.3  
tqdm==4.63.0  

## Usage:
We provide a demo to show how to train, validate and test GIGN. First, cd ./GIGN
### 1. Model training
Firstly, download the preprocessed datasets from , and organize them as './data/train', './data/valid', './data/test2013/', './data/test2016/', and  './data/test2019/'.  
Secondly, run train.py using `python train.py`.  

### 2. Model testing
Run test.py using `python predict.py`.  
You may need to modify some file paths in the source code before running it.  
You can also use `python evaluate.py`, which will return the mean (std) of performance for the three independent runs.

### 3. Process raw data
We provide a demo to explain how to process the raw data. This demo use ./data/toy_examples.csv and ./data/toy_set/ as examples.  
Firstly, run preprocessing.py using `python preprocessing.py`.    
Secondly, run dataset_GIGN.py using `python dataset_GIGN.py`.  
Thirdly, run train.py using `python train_example.py`.    

### 4. Test the trained model in other external test sets
Firstly, please organize the data as a structure similar to './data/toy_set' folder.  
-data  
&ensp;&ensp;-external_test  
&ensp; &ensp;&ensp;&ensp; -pdb_id  
&ensp; &ensp; &ensp;&ensp;&ensp;&ensp;-pdb_id_ligand.mol2  
&ensp; &ensp; &ensp;&ensp;&ensp;&ensp;-pdb_id_protein.pdb  
Secondly, run preprocessing.py using `python preprocessing.py`.  
Thirdly, run dataset_GIGN.py using `python dataset_GIGN.py`.  
Fourth, run predict.py using `python predict.py`.  
You may need to modify some file paths in the source code before running it.  

## Other baselines:
The usage of other baselines is similar to GIGN.
