
import pandas as pd
import numpy as np
import dgl
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from dgllife.utils import (
    ConcatFeaturizer,
    BaseAtomFeaturizer,
    BaseBondFeaturizer,
    SMILESToBigraph,
    atom_type_one_hot,
    atom_degree_one_hot,
    atom_explicit_valence_one_hot,
    atom_num_radical_electrons_one_hot,
    atom_is_aromatic_one_hot,
    atom_is_in_ring_one_hot,
    atom_total_num_H_one_hot,
    bond_type_one_hot,
    bond_is_in_ring
)

def load_drug_data(csv_path):
    df = pd.read_csv(csv_path)
    # columns: DRUG_NAME, Canonical_SMILES
    smiles_list = df['Canonical_SMILES'].tolist()
    drug_names = df['DRUG_NAME'].tolist()
    return drug_names, smiles_list

def get_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return zeros if invalid
        return np.zeros((nBits,), dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)

def build_drug_graph(smiles_list, device='cpu'):
    n_drugs = len(smiles_list)
    fps = []
    
    for smi in smiles_list:
        fps.append(get_fingerprint(smi))
    
    fps = np.array(fps)
    intersections = np.dot(fps, fps.T)
    similarities = intersections / 2048.0
    
    src, dst = np.nonzero(np.ones((n_drugs, n_drugs)))
    weights = similarities[src, dst]
    
    g = dgl.graph((src, dst), num_nodes=n_drugs)
    g.edata['w'] = torch.tensor(weights, dtype=torch.float32)
    
    return g

atom_concat_featurizer = ConcatFeaturizer([
    atom_type_one_hot, 
    atom_degree_one_hot, 
    atom_explicit_valence_one_hot, 
    atom_num_radical_electrons_one_hot, 
    atom_is_aromatic_one_hot, 
    atom_is_in_ring_one_hot, 
    atom_total_num_H_one_hot
])

mol_atom_featurizer = BaseAtomFeaturizer({'h': atom_concat_featurizer})

bond_concat_featurizer = ConcatFeaturizer([
    bond_type_one_hot, 
    bond_is_in_ring
])

bond_featurizer = BaseBondFeaturizer({'e': bond_concat_featurizer}, self_loop=True)

smi_to_g = SMILESToBigraph(
    add_self_loop=True, 
    node_featurizer=mol_atom_featurizer,
    edge_featurizer=bond_featurizer
)

def smiles_to_graph(smiles):
    if pd.isna(smiles) or smiles == '':
         dummy = smi_to_g('C')
         g = dgl.graph(([], []), num_nodes=0)
         g.ndata['h'] = torch.zeros(0, dummy.ndata['h'].shape[1])
         g.edata['e'] = torch.zeros(0, dummy.edata['e'].shape[1])
         return g
         
    try:
        g = smi_to_g(smiles)
        if g is None:
             raise ValueError("Graph creation failed")
        return g
    except:
         dummy = smi_to_g('C')
         g = dgl.graph(([], []), num_nodes=0)
         g.ndata['h'] = torch.zeros(0, dummy.ndata['h'].shape[1])
         g.edata['e'] = torch.zeros(0, dummy.edata['e'].shape[1])
         return g


def decompose_smiles(smiles_list, max_len=100):
    unique_chars = set()
    for s in smiles_list:
        unique_chars.update(set(s))
    
    vocab = sorted(list(unique_chars))
    char_to_idx = {c: i+1 for i, c in enumerate(vocab)} # 0 is padding
    
    encoded = []
    for s in smiles_list:
        vec = [char_to_idx[c] for c in s]
        if len(vec) < max_len:
            vec += [0] * (max_len - len(vec))
        else:
            vec = vec[:max_len]
        encoded.append(vec)
        
    return torch.tensor(encoded, dtype=torch.long), len(vocab) + 1
