
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import dgl

def orthogonal_loss(z1, z2):
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)
    
    cosine_sim = torch.sum(z1_norm * z2_norm, dim=1)
    return torch.mean(cosine_sim ** 2)

class Trainer:
    def __init__(self, model, loss_module, train_loader, val_loader, optimizer_config, device='cuda', drug_graphs=None, drug_mean_response=None, cell_dims=None, lambdas_config=None, log_dir=None):
        self.model = model.to(device)
        self.loss_module = loss_module 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cell_dims = cell_dims
        
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None
        
        self.drug_graphs = drug_graphs
             
        if drug_mean_response is not None:
            self.drug_mean_response = drug_mean_response.to(device)
        else:
            self.drug_mean_response = None
        
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=optimizer_config.get('lr', 1e-3),
                                    weight_decay=optimizer_config.get('weight_decay', 1e-4))
        
        default_lambdas = {
            'drug_mean': 0.1,
            'res_norm': 0.5,
            'recon': 0.01,
            'orth': 0.1,
            'class': 0.1 
        }
        
        if lambdas_config is not None:
            default_lambdas.update(lambdas_config)
            
        self.lambdas = default_lambdas
        
    def _prepare_data_and_forward(self, batch):
        drug_feat = batch['drug_feat'].to(self.device)
        cell_feat = batch['cell_feat'].to(self.device)
        opt_feat = batch['opt_feat'].to(self.device)
        cell_id = batch['cell_id'].to(self.device)
        ic50 = batch['ic50'].to(self.device)
        mask = batch['mask'].to(self.device)
        drug_idx = batch['drug_idx'].to(self.device)
        
        split_dict = {}
        curr = 0
        split_dict['gep'] = cell_feat[:, curr:curr+self.cell_dims['gep']]
        curr += self.cell_dims['gep']
        split_dict['cnv'] = cell_feat[:, curr:curr+self.cell_dims['cnv']]
        curr += self.cell_dims['cnv']
        split_dict['mut'] = cell_feat[:, curr:curr+self.cell_dims['mut']]
        
        return drug_feat, split_dict, opt_feat, cell_id, ic50, mask, drug_idx

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loss_dict_agg = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            drug_feat_batch, cell_feat_dict, opt_feat, cell_id, ic50, mask, drug_idx = self._prepare_data_and_forward(batch)
            
            batch_graphs_list = [self.drug_graphs[i] for i in drug_idx.tolist()]
            batch_graph = dgl.batch(batch_graphs_list).to(self.device)
            
            self.optimizer.zero_grad()
            
            results = self.model(
                batch_graph,
                drug_feat_batch, 
                cell_feat_dict, 
                opt_feat, 
                cell_id, 
                mask
            )
            
            pred_ic50, loss_recon_total, pred_cell_logits, preds_tuple, embeddings_tuple = results
            
            pred_inv, pred_expl, pred_res = preds_tuple
            z_inv, z_gep, z_cnv, z_mut, z_res = embeddings_tuple

            loss_task = F.mse_loss(pred_ic50.view(-1), ic50.view(-1))
            
            if self.drug_mean_response is not None:
                target_drug_mean = self.drug_mean_response[drug_idx]
                loss_drug_mean = F.mse_loss(pred_inv.view(-1), target_drug_mean.view(-1))
            else:
                loss_drug_mean = torch.tensor(0.0).to(self.device)
            
            loss_res_norm = torch.mean(pred_res ** 2)
            
            loss_class = F.cross_entropy(pred_cell_logits, cell_id)
            
            loss_orth = 0
            loss_orth += orthogonal_loss(z_inv, z_res)
            loss_orth += orthogonal_loss(z_res, z_gep)
            loss_orth += orthogonal_loss(z_res, z_cnv)
            loss_orth += orthogonal_loss(z_res, z_mut)
            
            loss = loss_task + \
                   self.lambdas['drug_mean'] * loss_drug_mean + \
                   self.lambdas['res_norm'] * loss_res_norm + \
                   self.lambdas['recon'] * loss_recon_total + \
                   self.lambdas['orth'] * loss_orth + \
                   self.lambdas['class'] * loss_class
                   
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                pred_cls = torch.argmax(pred_cell_logits, dim=1)
                acc = (pred_cls == cell_id).float().mean()
            
            current_loss_dict = {
                'task': loss_task.item(),
                'drug_mean': loss_drug_mean.item(),
                'orth': loss_orth.item() if isinstance(loss_orth, torch.Tensor) else loss_orth,
                'recon': loss_recon_total.item(),
                'class': loss_class.item(),
                'acc': acc.item()
            }
             
            for k, v in current_loss_dict.items():
                loss_dict_agg[k] = loss_dict_agg.get(k, 0) + v
                
            pbar.set_postfix({'loss': loss.item(), 'task': loss_task.item()})
            
        if len(self.train_loader) > 0:
            for k in loss_dict_agg:
                loss_dict_agg[k] /= len(self.train_loader)
            avg_loss = total_loss / len(self.train_loader)
        else:
            avg_loss = 0
            
        if self.writer:
            self.writer.add_scalar('train/total_loss', avg_loss, epoch)
            for k, v in loss_dict_agg.items():
                self.writer.add_scalar(f'train/{k}', v, epoch)
            
        return avg_loss, loss_dict_agg

    def validate(self, epoch=None):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                drug_feat_batch, cell_feat_dict, opt_feat, cell_id, ic50, mask, drug_idx = self._prepare_data_and_forward(batch)
                
                batch_graphs_list = [self.drug_graphs[i] for i in drug_idx.tolist()]
                batch_graph = dgl.batch(batch_graphs_list).to(self.device)
                
                results = self.model(
                    batch_graph,
                    drug_feat_batch, 
                    cell_feat_dict, 
                    opt_feat, 
                    cell_id, 
                    mask
                )
                
                pred_ic50, _, _, preds_tuple, _ = results
                pred_inv, pred_expl, pred_res = preds_tuple
                
                preds = pred_inv + pred_expl
                
                preds = preds.view(-1)
                
                loss = F.mse_loss(preds, ic50.view(-1))
                total_loss += loss.item()
                
                all_preds.append(preds.cpu())
                all_targets.append(ic50.cpu().view(-1))
        
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        mse = np.mean((all_preds - all_targets)**2)
        rmse = np.sqrt(mse)
        if np.std(all_preds) < 1e-9:
             pcc = 0.0
        else:
            pcc, _ = pearsonr(all_preds, all_targets)
            
        r2 = r2_score(all_targets, all_preds)
        
        val_loss = total_loss / len(self.val_loader)
        
        if self.writer and epoch is not None:
            self.writer.add_scalar('val/loss', val_loss, epoch)
            self.writer.add_scalar('val/mse', mse, epoch)
            self.writer.add_scalar('val/rmse', rmse, epoch)
            self.writer.add_scalar('val/pcc', pcc, epoch)
            self.writer.add_scalar('val/r2', r2, epoch)
        
        return {'loss': val_loss, 'mse': mse, 'rmse': rmse, 'pcc': pcc, 'r2': r2}

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))
