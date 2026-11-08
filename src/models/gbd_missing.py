import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from src.models.gbd import GBDDRP

class GBDDRPMissing(GBDDRP):
    def __init__(self, drug_dim, cell_dim, opt_dim, num_cells, vocab_size, embed_dim=128, cell_dims=None, dropout=0.1):
        super(GBDDRPMissing, self).__init__(drug_dim, cell_dim, opt_dim, num_cells, vocab_size, embed_dim, cell_dims, dropout)
        
        fp_emb_dim = 512
        in_dim_f2 = embed_dim * 2 + fp_emb_dim + self.cell_dims['gep']
        
        self.f2 = self._build_mlp(in_dim_f2, dropout)
        
        in_dim_cnv = embed_dim * 2 + fp_emb_dim + self.cell_dims['cnv']
        self.f_cnv = self._build_mlp(in_dim_cnv, dropout)
        
        in_dim_mut = embed_dim * 2 + fp_emb_dim + self.cell_dims['mut']
        self.f_mut = self._build_mlp(in_dim_mut, dropout)
        
        self.f3 = nn.Sequential(
            nn.Linear(embed_dim + 2, embed_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim * 2),
            nn.Dropout(dropout),
            
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
            
            nn.Linear(embed_dim, 1)
        )

    def _build_mlp(self, in_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(dropout),
            
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            
            nn.Linear(512, 1)
        )

    def forward(self, drug_graph, drug_feat, cell_feat_dict, opt_feat, cell_id, masks=None):
        if masks is None:
            device = drug_feat.device
            batch_size = drug_feat.shape[0]
            masks = {
                'cnv': torch.ones(batch_size, 1).to(device),
                'mut': torch.ones(batch_size, 1).to(device)
            }

        z_inv, h_nodes, fp_emb, g_emb = self.E_inv_attn(drug_graph, drug_feat)
        
        def process_branch(cell_feat, attn_module, proj_module, decoder):
            h_attn_agg = attn_module(drug_graph, h_nodes, cell_feat)
            combined = torch.cat([h_attn_agg, cell_feat, z_inv], dim=1)
            z_expl = proj_module(combined)
            rec_feat = decoder(z_expl)
            loss_rec = F.mse_loss(rec_feat, cell_feat, reduction='none').mean(dim=1)
            return z_expl, loss_rec

        z_gep, l_rec_gep = process_branch(cell_feat_dict['gep'], self.Attn_gep, self.Proj_gep, self.Dec_gep)
        z_cnv, l_rec_cnv = process_branch(cell_feat_dict['cnv'], self.Attn_cnv, self.Proj_cnv, self.Dec_cnv)
        z_mut, l_rec_mut = process_branch(cell_feat_dict['mut'], self.Attn_mut, self.Proj_mut, self.Dec_mut)

        cell_emb_vec = self.cell_emb(cell_id)
        combined_res = torch.cat([g_emb, fp_emb, cell_emb_vec], dim=1)
        z_res = self.E_res(combined_res)
        pred_cell_logits = self.C_res(z_res)

        pred_inv = self.f1(z_inv)
        
        feat_base = torch.cat([z_gep, z_inv, fp_emb, cell_feat_dict['gep']], dim=1)
        pred_base = self.f2(feat_base)
        
        feat_cnv = torch.cat([z_cnv, z_inv, fp_emb, cell_feat_dict['cnv']], dim=1)
        pred_cnv = self.f_cnv(feat_cnv)
        
        feat_mut = torch.cat([z_mut, z_inv, fp_emb, cell_feat_dict['mut']], dim=1)
        pred_mut = self.f_mut(feat_mut)
        
        mask_vec = torch.cat([masks['cnv'], masks['mut']], dim=1)
        feat_res = torch.cat([z_res, mask_vec], dim=1)
        pred_res = self.f3(feat_res)
        
        pred = pred_inv + pred_base + pred_res + \
               (pred_cnv * masks['cnv']) + \
               (pred_mut * masks['mut'])

        mask_vec_cnv_0 = torch.cat([torch.zeros_like(masks['cnv']), masks['mut']], dim=1)
        mask_vec_cnv_1 = torch.cat([torch.ones_like(masks['cnv']), masks['mut']], dim=1)
        
        p_res_cnv_0 = self.f3(torch.cat([z_res, mask_vec_cnv_0], dim=1))
        p_res_cnv_1 = self.f3(torch.cat([z_res, mask_vec_cnv_1], dim=1))
        
        diff_cnv = p_res_cnv_0 - (p_res_cnv_1 + pred_cnv)
        loss_cons_cnv = (diff_cnv ** 2) * masks['cnv']

        mask_vec_mut_0 = torch.cat([masks['cnv'], torch.zeros_like(masks['mut'])], dim=1)
        mask_vec_mut_1 = torch.cat([masks['cnv'], torch.ones_like(masks['mut'])], dim=1)
        
        p_res_mut_0 = self.f3(torch.cat([z_res, mask_vec_mut_0], dim=1))
        p_res_mut_1 = self.f3(torch.cat([z_res, mask_vec_mut_1], dim=1))
        
        diff_mut = p_res_mut_0 - (p_res_mut_1 + pred_mut)
        loss_cons_mut = (diff_mut ** 2) * masks['mut']

        loss_cons_total = loss_cons_cnv.mean() + loss_cons_mut.mean()
        
        pred_expl_composite = pred_base + (pred_cnv * masks['cnv']) + (pred_mut * masks['mut'])
        
        recon_losses = (l_rec_gep, l_rec_cnv, l_rec_mut)
        
        embeddings_tuple = (z_inv, z_gep, z_cnv, z_mut, z_res)
        
        return pred, recon_losses, pred_cell_logits, (pred_inv, pred_expl_composite, pred_res, loss_cons_total), embeddings_tuple
