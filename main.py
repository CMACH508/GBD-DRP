
import argparse
import torch
import numpy as np
import os
from src.data.dataset import get_dataloaders, load_feature_data
from src.models.gbd import GBDDRP
from src.training.trainer import Trainer
from src.utils.graph_utils import load_drug_data, smiles_to_graph
import pandas as pd

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

def run_fold(args, fold, features_cache=None, cell_dims=None):
    print(f"Running GBD-DRP (Function Decomposition) on {args.cv_type} Fold {fold}...")
    
    train_loader, val_loader, meta = get_dataloaders(
        args.data_dir, 
        args.cv_type, 
        fold, 
        batch_size=args.batch_size,
        features_cache=features_cache,
        use_cell_batch=False
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
    model = GBDDRP(drug_dim, cell_dim, opt_dim, num_cells, vocab_size, embed_dim=args.embed_dim, cell_dims=cell_dims, dropout=args.dropout)
    
    lambdas = {
        'drug_mean': args.w_drug_mean,
        'res_norm': args.w_res_norm,
        'recon': args.w_recon,
        'orth': args.w_orth,
        'class': args.w_class
    }

    log_dir_name = f"{args.cv_type}_fold{fold}_{args.embed_dim}_dr{args.dropout}_lr{args.lr}"
    log_dir = os.path.join(args.data_dir, "../logs", log_dir_name)
    ckpt_dir = os.path.join(args.data_dir, "../checkpoints", log_dir_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    trainer = Trainer(
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
        log_dir=log_dir
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

        if  epoch - best_epoch >= 50:
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
    parser.add_argument('--w_recon', type=float, default=0.01, help="Weight for reconstruction loss")
    parser.add_argument('--w_orth', type=float, default=1.0, help="Weight for orthogonality loss")
    parser.add_argument('--w_class', type=float, default=0.2, help="Weight for cell classification loss")
    
    parser.add_argument('--data_dir', type=str, default='/data')
    parser.add_argument('--cv_type', type=str, default='mixed', 
                        choices=['mixed', 'cell_blind', 'drug_blind'],
                        help="Cross-validation strategy")
    parser.add_argument('--fold', type=int, default=0, help="Fold index to run (0-9)")
    parser.add_argument('--all_folds', action='store_true', help="Run all 10 folds and report average")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    if args.all_folds:
        print(f"Starting 10-Fold Cross-Validation for {args.cv_type}...")

        d_dict, c_dict, c_map, d_map, cell_dims = load_feature_data(args.data_dir)
        features_cache = (d_dict, c_dict, c_map, d_map) 
        
        results = []
        for k in range(10):
            metrics = run_fold(args, k, features_cache, cell_dims)
            results.append(metrics)
            
        mses = [r['mse'] for r in results]
        rmses = [r['rmse'] for r in results]
        pccs = [r['pcc'] for r in results]
        
        print("\n=== 10-Fold CV Results ===")
        print(f"MSE:  {np.mean(mses):.4f} +/- {np.std(mses):.4f}")
        print(f"RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
        print(f"PCC:  {np.mean(pccs):.4f} +/- {np.std(pccs):.4f}")
        
    else:
        _, _, _, _, cell_dims = load_feature_data(args.data_dir)
        run_fold(args, args.fold, cell_dims=cell_dims)

if __name__ == '__main__':
    main()
