import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import dgl
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from src.training.trainer import Trainer

def weighted_orthogonal_loss(z1, z2, weights=None):
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)
    
    cosine_sim_sq = torch.sum(z1_norm * z2_norm, dim=1) ** 2
    
    if weights is not None:
        weights = weights.view(-1)
        loss = (cosine_sim_sq * weights).sum() / (weights.sum() + 1e-8)
    else:
        loss = torch.mean(cosine_sim_sq)
        
    return loss

class TrainerMissing(Trainer):
    def __init__(self, model, loss_module, train_loader, val_loader, optimizer_config, device='cuda', 
                 drug_graphs=None, drug_mean_response=None, cell_dims=None, lambdas_config=None, log_dir=None,
                 missing_rate=0.0, use_fixed_masks=False):
        super().__init__(model, loss_module, train_loader, val_loader, optimizer_config, device, 
                         drug_graphs, drug_mean_response, cell_dims, lambdas_config, log_dir)
        self.missing_rate = missing_rate
        self.use_fixed_masks = use_fixed_masks
        
    def _prepare_masks(self, batch_size, batch_data):
        if self.use_fixed_masks and 'mask_cnv' in batch_data and 'mask_mut' in batch_data:
            return {
                'cnv': batch_data['mask_cnv'].to(self.device).view(-1, 1),
                'mut': batch_data['mask_mut'].to(self.device).view(-1, 1)
            }
        
        return self._generate_masks(batch_size)

    def _generate_masks(self, batch_size):
        if self.missing_rate > 0:
            probs = torch.ones(batch_size, 1).to(self.device) * (1 - self.missing_rate)
            mask_cnv = torch.bernoulli(probs)
            mask_mut = torch.bernoulli(probs)
        else:
            mask_cnv = torch.ones(batch_size, 1).to(self.device)
            mask_mut = torch.ones(batch_size, 1).to(self.device)
            
        return {'cnv': mask_cnv, 'mut': mask_mut}

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loss_dict_agg = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} (Missing rate {self.missing_rate})")
        for batch in pbar:
            drug_feat_batch, cell_feat_dict, opt_feat, cell_id, ic50, mask_valid, drug_idx = self._prepare_data_and_forward(batch)
            
            masks = self._prepare_masks(drug_feat_batch.shape[0], batch)
            
            batch_graphs_list = [self.drug_graphs[i] for i in drug_idx.tolist()]
            batch_graph = dgl.batch(batch_graphs_list).to(self.device)
            
            self.optimizer.zero_grad()
            
            results = self.model(
                batch_graph,
                drug_feat_batch, 
                cell_feat_dict, 
                opt_feat, 
                cell_id, 
                masks
            )
            
            pred, recon_losses_tuple, pred_cell_logits, preds_tuple, embeddings_tuple = results
            l_rec_gep_vec, l_rec_cnv_vec, l_rec_mut_vec = recon_losses_tuple

            if len(preds_tuple) == 4:
                pred_inv, pred_expl, pred_res, loss_cons_from_model = preds_tuple
            else:
                pred_inv, pred_expl, pred_res = preds_tuple
                loss_cons_from_model = 0
            
            z_inv, z_gep, z_cnv, z_mut, z_res = embeddings_tuple
            loss_task = F.mse_loss(pred.view(-1), ic50.view(-1))
            
            if self.drug_mean_response is not None:
                target_drug_mean = self.drug_mean_response[drug_idx]
                loss_drug_mean = F.mse_loss(pred_inv.view(-1), target_drug_mean.view(-1))
            else:
                loss_drug_mean = torch.tensor(0.0).to(self.device)
            
            loss_res_norm = torch.mean(pred_res ** 2)
            
            loss_recon = torch.mean(l_rec_gep_vec)
            loss_recon += (l_rec_cnv_vec * masks['cnv'].view(-1)).sum() / (masks['cnv'].sum() + 1e-8)
            loss_recon += (l_rec_mut_vec * masks['mut'].view(-1)).sum() / (masks['mut'].sum() + 1e-8)
            
            loss_orth = 0
            
            loss_orth += weighted_orthogonal_loss(z_gep, z_res)
            loss_orth += weighted_orthogonal_loss(z_inv, z_res)
            loss_orth += weighted_orthogonal_loss(z_cnv, z_res, weights=masks['cnv'])
            loss_orth += weighted_orthogonal_loss(z_mut, z_res, weights=masks['mut'])
            
            loss_class = F.cross_entropy(pred_cell_logits, cell_id)
            
            loss = loss_task + \
                   self.lambdas['drug_mean'] * loss_drug_mean + \
                   self.lambdas['res_norm'] * loss_res_norm + \
                   self.lambdas['recon'] * loss_recon + \
                   self.lambdas['orth'] * loss_orth + \
                   self.lambdas['class'] * loss_class
                   
            if isinstance(loss_cons_from_model, torch.Tensor):
                loss += self.lambdas.get('cons', 1.0) * loss_cons_from_model
                   
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                pred_cls = torch.argmax(pred_cell_logits, dim=1)
                acc = (pred_cls == cell_id).float().mean()
            
            current_loss_dict = {
                'task': loss_task.item(),
                'drug_mean': loss_drug_mean.item(),
                'orth': loss_orth.item(),
                'recon': loss_recon.item(),
                'class': loss_class.item(),
                'cons': loss_cons_from_model.item() if isinstance(loss_cons_from_model, torch.Tensor) else 0,
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

    def validate(self, unseen_domain=False, epoch=None):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                drug_feat_batch, cell_feat_dict, opt_feat, cell_id, ic50, mask_valid, drug_idx = self._prepare_data_and_forward(batch)
            
                batch['mask_cnv'] = batch.get('mask_cnv')
                batch['mask_mut'] = batch.get('mask_mut')
                
                masks = self._prepare_masks(drug_feat_batch.shape[0], batch)

                batch_graphs_list = [self.drug_graphs[i] for i in drug_idx.tolist()]
                batch_graph = dgl.batch(batch_graphs_list).to(self.device)
                
                results = self.model(
                    batch_graph,
                    drug_feat_batch, 
                    cell_feat_dict, 
                    opt_feat, 
                    cell_id, 
                    masks
                )
                
                pred, _, _, preds_tuple, _ = results
                if len(preds_tuple) == 4:
                    pred_inv, pred_expl, pred_res, _ = preds_tuple
                else:
                    pred_inv, pred_expl, pred_res = preds_tuple
                
                if unseen_domain:
                    preds = pred_inv + pred_expl 
                else:
                    preds = pred
                
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
        
        return {'loss': val_loss, 'mse': mse, 'rmse': rmse, 'pcc': pcc, 'r2': r2}
