import argparse
import torch
import numpy as np
import os
import pandas as pd
from src.data.dataset import get_dataloaders, load_feature_data
from src.models.gbd_missing import GBDDRPMissing
from src.training.trainer_missing import TrainerMissing
from src.utils.graph_utils import load_drug_data, smiles_to_graph

def compute_drug_mean_response(train_loader, num_drugs, device):
    drug_sum = torch.zeros(num_drugs, device=device)
    drug_count = torch.zeros(num_drugs, device=device)
    
    print("Computing Drug Mean Response using Training Data...")
    for batch in train_loader:
        ic50 = batch['ic50'].to(device).view(-1)
        drug_idx = batch['drug_idx'].to(device)
        
        drug_sum.index_add_(0, drug_idx, ic50)
        drug_count.index_add_(0, drug_idx, torch.ones_like(ic50))
        
    drug_mean = drug_sum / (drug_count + 1e-8)
    return drug_mean


def load_fixed_masks(data_dir, rate):
    mask_dir = os.path.join(data_dir, "missing_masks")
    print(f"Loading fixed masks (rate={rate}) from {mask_dir}...")
    
    df_cnv = pd.read_csv(os.path.join(mask_dir, f"mask_cnv_{rate}.csv"))
    df_mut = pd.read_csv(os.path.join(mask_dir, f"mask_mut_{rate}.csv"))
    
    mask_dict = {}
    
    for _, row in df_cnv.iterrows():
        cell = row['cell_line']
        val = row['mask']
        if cell not in mask_dict: mask_dict[cell] = {}
        mask_dict[cell]['cnv'] = val
        
    for _, row in df_mut.iterrows():
        cell = row['cell_line']
        val = row['mask']
        if cell not in mask_dict: mask_dict[cell] = {}
        mask_dict[cell]['mut'] = val
        
    return mask_dict

def run_fold(args, fold, features_cache=None, cell_dims=None, fixed_masks=None):
    print(f"Running GBDDRP (Missing Modality Improved) on {args.cv_type} Fold {fold}...")
    
    train_loader, val_loader, meta = get_dataloaders(
        args.data_dir, 
        args.cv_type, 
        fold, 
        batch_size=args.batch_size,
        features_cache=features_cache,
        use_cell_batch=False,
        cell_masks=fixed_masks 
    )
    
    drug_dim = meta['drug_dim'] 
    cell_dim = meta['cell_dim']
    num_cells = meta['num_cells']
    opt_dim = 0 
    
    print("Constructing Drug Graphs...")
    drug_names_file = os.path.join(args.data_dir, "DrugName_AND_Canonical_SMILES.csv")
    drug_names, smiles_list = load_drug_data(drug_names_file)
    
    drug_graphs = []
    for sm in smiles_list:
        if pd.isna(sm):
            drug_graphs.append(smiles_to_graph('')) 
        else:
            drug_graphs.append(smiles_to_graph(sm))
            
    num_drugs = len(drug_graphs)
    vocab_size = 0 
    
    drug_mean_response = compute_drug_mean_response(train_loader, num_drugs, args.device)
    
    print(f"Cell Dims: {cell_dims}")
    model = GBDDRPMissing(drug_dim, cell_dim, opt_dim, num_cells, vocab_size, embed_dim=args.embed_dim, cell_dims=cell_dims, dropout=args.dropout)
    
    lambdas = {
        'drug_mean': args.w_drug_mean,
        'res_norm': args.w_res_norm,
        'recon': args.w_recon,
        'orth': args.w_orth,
        'class': args.w_class,
        'cons': args.w_cons
    }

    log_dir_name = f"MissingAware_{args.cv_type}_fold{fold}_M{args.missing_rate}"
    log_dir = os.path.join(args.data_dir, "../logs", log_dir_name)
    ckpt_dir = os.path.join(args.data_dir, "../checkpoints", log_dir_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    trainer = TrainerMissing(
        model, 
        None, 
        train_loader, 
        val_loader, 
        {'lr': args.lr}, 
        device=args.device,
        drug_graphs=drug_graphs,
        drug_mean_response=drug_mean_response,
        cell_dims=cell_dims,
        lambdas_config=lambdas,
        log_dir=log_dir,
        missing_rate=args.missing_rate,
        use_fixed_masks=args.use_fixed_masks
    )
    
    best_metrics = None
    best_mse = float('inf')

    best_epoch = -1
    
    for epoch in range(args.epochs):
        loss, loss_dict = trainer.train_epoch(epoch)
        loss_str = " | ".join([f"{k} {v:.4f}" for k, v in loss_dict.items()])
        
        metrics = trainer.validate(epoch=epoch)
        
        print(f"Fold {fold} Epoch {epoch}: Total {loss:.4f} | {loss_str}")
        print(f"Val MSE {metrics['mse']:.4f} PCC {metrics['pcc']:.4f} R2 {metrics['r2']:.4f}")
        
        if metrics['mse'] < best_mse:
            best_mse = metrics['mse']
            best_metrics = metrics
            trainer.save_checkpoint(os.path.join(ckpt_dir, "best_model.pth"))
            print(f"Saved best model to {ckpt_dir}/best_model.pth")
            best_epoch = epoch

        if  epoch - best_epoch >= 30:
            print("Early stopping triggered.")
            break
            
    print(f"Fold {fold} Best: MSE {best_metrics['mse']:.4f} RMSE {best_metrics['rmse']:.4f} PCC {best_metrics['pcc']:.4f} R2 {best_metrics['r2']:.4f}")
    return best_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    parser.add_argument('--w_drug_mean', type=float, default=0.1)
    parser.add_argument('--w_res_norm', type=float, default=0.5)
    parser.add_argument('--w_recon', type=float, default=0.01)
    parser.add_argument('--w_orth', type=float, default=1.0)
    parser.add_argument('--w_class', type=float, default=0.1)
    parser.add_argument('--w_cons', type=float, default=1.0)
    
    parser.add_argument('--missing_rate', type=float, default=0.5, help="Simulated missing rate for CNV/MUT modalities during training")
    parser.add_argument('--use_fixed_masks', action='store_true', help="Use pre-generated fixed masks from disk instead of random generation")
    
    parser.add_argument('--data_dir', type=str, default='/data2/zengbiyang/DA-DRP/data_process_sample/data')
    parser.add_argument('--cv_type', type=str, default='mixed', 
                        choices=['mixed', 'cell_blind', 'drug_blind'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--all_folds', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    if args.all_folds:
        print(f"Starting 10-Fold CV for {args.cv_type} with Missing Rate {args.missing_rate}...")
        d_dict, c_dict, c_map, d_map, cell_dims = load_feature_data(args.data_dir)
        features_cache = (d_dict, c_dict, c_map, d_map)
        
        mask_dir = os.path.join(args.data_dir, "missing_masks")
        if not os.path.exists(mask_dir):
            print("Warning: Missing masks directory not found. Fixed masks will NOT be used.")
            fixed_masks = None
        else:
            if args.use_fixed_masks:
                fixed_masks = load_fixed_masks(args.data_dir, args.missing_rate)
            else:
                fixed_masks = None

        results = []
        for k in range(10):
            metrics = run_fold(args, k, features_cache, cell_dims, fixed_masks)
            results.append(metrics)
            
        mses = [r['mse'] for r in results]
        
        print("\n=== 10-Fold CV Results ===")
        print(f"MSE:  {np.mean(mses):.4f} +/- {np.std(mses):.4f}")
        
    else:
        _, _, _, _, cell_dims = load_feature_data(args.data_dir)
        
        fixed_masks = None
        if args.use_fixed_masks:
             fixed_masks = load_fixed_masks(args.data_dir, args.missing_rate)
             
        run_fold(args, args.fold, cell_dims=cell_dims, fixed_masks=fixed_masks)

if __name__ == '__main__':
    main()
