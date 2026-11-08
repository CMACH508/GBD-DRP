
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from rdkit import Chem
from rdkit.Chem import AllChem

def get_morgan_fingerprint(smiles, radius=2, n_bits=512):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return np.zeros(n_bits)

def load_feature_data(data_dir):
    drug_file = os.path.join(data_dir, "DrugName_AND_Canonical_SMILES.csv")
    df_drug = pd.read_csv(drug_file)
    
    drug_names = df_drug.iloc[:, 1].values
    smiles_list = df_drug.iloc[:, 2].values
    
    print("Generating Drug Fingerprints (512 bits)...")
    drug_tensor_dict = {}
    drug_name_mapping = {}
    
    for idx, (name, sm) in enumerate(zip(drug_names, smiles_list)):
        if pd.isna(sm):
            feat = np.zeros(2048)
        else:
            feat = get_morgan_fingerprint(sm, n_bits=2048)
        drug_tensor_dict[name] = torch.tensor(feat, dtype=torch.float32)
        drug_name_mapping[name] = idx

    print("Loading Cell Features (GEP, CNV, MUT)...")
    gep_path = os.path.join(data_dir, "GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv")
    cnv_path = os.path.join(data_dir, "CNV_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv")
    mut_path = os.path.join(data_dir, "MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv")
    
    df_gep = pd.read_csv(gep_path, index_col=0)
    df_cnv = pd.read_csv(cnv_path, index_col=0)
    df_mut = pd.read_csv(mut_path, index_col=0)
    
    common_cells = sorted(list(set(df_gep.index) & set(df_cnv.index) & set(df_mut.index)))
    print(f"Found {len(common_cells)} common cell lines.")
    
    def process_df(df, cells):
        sub = df.loc[cells]
        mean = sub.mean()
        std = sub.std() + 1e-8
        sub = (sub - mean) / std
        return sub.fillna(0.0)

    df_gep = process_df(df_gep, common_cells)
    df_cnv = process_df(df_cnv, common_cells)
    df_mut = process_df(df_mut, common_cells)
    
    dims = {
        'gep': df_gep.shape[1],
        'cnv': df_cnv.shape[1],
        'mut': df_mut.shape[1]
    }
    
    df_combined = pd.concat([df_gep, df_cnv, df_mut], axis=1)
    
    cell_tensor_dict = {}
    cell_name_mapping = {name: i for i, name in enumerate(common_cells)}
    cell_matrix = torch.tensor(df_combined.values, dtype=torch.float32)
    
    for name, idx in cell_name_mapping.items():
        cell_tensor_dict[name] = cell_matrix[idx]
        
    return drug_tensor_dict, cell_tensor_dict, cell_name_mapping, drug_name_mapping, dims

class CellLineBatchSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.cell_groups = {}
        
        df = self.data_source.samples
        
        grouped = df.groupby('cell_line')
        self.groups = [group.index.tolist() for _, group in grouped]
        
    def __iter__(self):
        perm = torch.randperm(len(self.groups))
        for i in perm:
            yield self.groups[i]
            
    def __len__(self):
        return len(self.groups)

class DRPDataset(Dataset):
    def __init__(self, csv_file, drug_dict, cell_dict, cell_mapping, drug_mapping, cell_masks=None):
        self.data_df = pd.read_csv(csv_file)
        self.drug_dict = drug_dict
        self.cell_dict = cell_dict
        self.cell_mapping = cell_mapping
        self.drug_mapping = drug_mapping
        self.cell_masks = cell_masks
        
        valid_indices = []
        missing_cnt = 0
        for idx, row in self.data_df.iterrows():
            if row['drug'] in self.drug_dict and row['cell_line'] in self.cell_dict:
                valid_indices.append(idx)
            else:
                missing_cnt += 1
        
        print(f"Loaded {len(valid_indices)} samples from {os.path.basename(csv_file)}. (Dropped {missing_cnt})")
        self.samples = self.data_df.loc[valid_indices].reset_index(drop=True)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        drug = row['drug']
        cell = row['cell_line']
        ic50 = float(row['IC50'])
        
        drug_feat = self.drug_dict[drug]
        cell_feat = self.cell_dict[cell]
        
        cell_id = self.cell_mapping[cell]
        drug_idx = self.drug_mapping[drug] 

        mask_cnv = 1.0
        mask_mut = 1.0
        if hasattr(self, 'cell_masks') and self.cell_masks is not None:
            if cell in self.cell_masks:
                mask_cnv = float(self.cell_masks[cell].get('cnv', 1.0))
                mask_mut = float(self.cell_masks[cell].get('mut', 1.0))
        
        return {
            'drug_feat': drug_feat,
            'cell_feat': cell_feat,
            'opt_feat': torch.empty(0),
            'cell_id': torch.tensor(cell_id, dtype=torch.long),
            'drug_idx': torch.tensor(drug_idx, dtype=torch.long),
            'mask': torch.tensor(1.0), # Legacy
            'mask_cnv': torch.tensor(mask_cnv, dtype=torch.float32),
            'mask_mut': torch.tensor(mask_mut, dtype=torch.float32),
            'ic50': torch.tensor(ic50, dtype=torch.float32)
        }

def get_dataloaders(data_dir, cv_type, fold, batch_size=32, num_workers=4, features_cache=None, use_cell_batch=False, cell_masks=None):
    if features_cache is None:
        drug_dict, cell_dict, cell_mapping, drug_mapping, _ = load_feature_data(data_dir)
    else:
        drug_dict, cell_dict, cell_mapping, drug_mapping = features_cache
    
    folder_map = {
        'mixed': 'mixed',
        'cell_blind': 'cell_blind',
        'drug_blind': 'drug_blind'
    }
    prefix_map = {
        'mixed': 'MixedSet',
        'cell_blind': 'CellBlind',
        'drug_blind': 'DrugBlind'
    }
    
    if cv_type not in folder_map:
        raise ValueError(f"Invalid cv_type: {cv_type}")
        
    folder = folder_map[cv_type]
    prefix = prefix_map[cv_type]
    
    train_path = os.path.join(data_dir, "10_fold_data", folder, f"{prefix}_train_Fold{fold}.csv")
    test_path = os.path.join(data_dir, "10_fold_data", folder, f"{prefix}_test_Fold{fold}.csv")
    
    train_dataset = DRPDataset(train_path, drug_dict, cell_dict, cell_mapping, drug_mapping, cell_masks=cell_masks)
    val_dataset = DRPDataset(test_path, drug_dict, cell_dict, cell_mapping, drug_mapping, cell_masks=cell_masks) # Validate with masks too? Or full info? Usually full info for val.

    
    if use_cell_batch:
        print("Using CellLineBatchSampler for Training...")
        batch_sampler = CellLineBatchSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    meta = {
        'drug_dim': 2048,
        'cell_dim': next(iter(cell_dict.values())).shape[0],
        'num_cells': len(cell_mapping),
        'vocab_size': 0
    }
    
    return train_loader, val_loader, meta
