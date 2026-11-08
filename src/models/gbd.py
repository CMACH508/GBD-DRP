
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import EdgeGATConv
from dgl.nn.pytorch.glob import AvgPooling, WeightAndSum


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=2, batch_norm=True, dropout=0.0):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        in_d = input_dim
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(in_d, hidden_dim))
            self.layers.append(nn.ReLU())
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        self.layers.append(nn.Linear(in_d, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class InvariantDrugGCN(nn.Module):
    def __init__(self, atom_dim, hidden_dim, out_dim, drug_fp_dim=512, drug_fp_out_dim=512, edge_dim=6, num_heads=3):
        super(InvariantDrugGCN, self).__init__()
        self.conv1 = EdgeGATConv(atom_dim, edge_dim, hidden_dim, num_heads=num_heads)
        self.conv2 = EdgeGATConv(hidden_dim * num_heads, edge_dim, hidden_dim, num_heads=num_heads)
        self.readout = WeightAndSum(hidden_dim * num_heads)
        
        self.fp_mlp = nn.Sequential(
            nn.Linear(drug_fp_dim, drug_fp_out_dim), 
            nn.ReLU(),
            nn.Linear(drug_fp_out_dim, drug_fp_out_dim),
            nn.ReLU()
        )
        
        self.proj = nn.Linear(hidden_dim * num_heads + drug_fp_out_dim, out_dim)
        
    def forward(self, g, drug_fp):
        h = g.ndata['h']
        e = g.edata['e']
        
        h = self.conv1(g, h, e)
        h = F.relu(h)
        h = h.flatten(1)
        
        h = self.conv2(g, h, e)
        h = F.relu(h)
        h = h.flatten(1)
        
        g_emb = self.readout(g, h)
        
        fp_emb = self.fp_mlp(drug_fp)
    
        combined = torch.cat([g_emb, fp_emb], dim=1)
        z_inv = self.proj(combined)
        
        return z_inv, h, fp_emb, g_emb

class Reconstructor(nn.Module):
    def __init__(self, z_dim, feature_dim):
        super(Reconstructor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, feature_dim)
        )
    def forward(self, z):
        return self.net(z)

class CellDrugAttention(nn.Module):
    def __init__(self, node_dim, cell_dim, num_heads=8):
        super(CellDrugAttention, self).__init__()
        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"
        self.d_k = node_dim // num_heads
        self.num_heads = num_heads
        self.node_dim = node_dim
        
        self.W_q = nn.Linear(cell_dim, node_dim)
        self.W_k = nn.Linear(node_dim, node_dim)
        self.W_v = nn.Linear(node_dim, node_dim)
        
    def forward(self, g, h_nodes, cell_feat):
        Q = self.W_q(cell_feat).view(-1, self.num_heads, self.d_k)
        K = self.W_k(h_nodes).view(-1, self.num_heads, self.d_k)
        V = self.W_v(h_nodes).view(-1, self.num_heads, self.d_k)
        
        Q_expanded = dgl.broadcast_nodes(g, Q)
        
        scores = (Q_expanded * K).sum(dim=-1, keepdim=True) 
        scores = scores / (self.d_k ** 0.5)
        
        g.ndata['attn_score'] = scores
        weights = dgl.softmax_nodes(g, 'attn_score')
        
        g.ndata['h_w'] = V * weights
        h_weighted = dgl.sum_nodes(g, 'h_w')
        
        out = h_weighted.view(-1, self.node_dim)
        return out

class CellClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super(CellClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class GBDDRP(nn.Module):
    def __init__(self, drug_dim, cell_dim, opt_dim, num_cells, vocab_size, embed_dim=128, cell_dims=None, dropout=0.1):
        super(GBDDRP, self).__init__()
        
        self.embed_dim = embed_dim
        self.cell_dims = cell_dims
        self.dropout = dropout
        
        gcn_hidden = 256 // 8
        gcn_heads = 8
        fp_emb_dim = 512

        self.E_inv_attn = InvariantDrugGCN(
            atom_dim=74, 
            hidden_dim=gcn_hidden, 
            out_dim=embed_dim, 
            drug_fp_dim=drug_dim,
            drug_fp_out_dim=fp_emb_dim,
            edge_dim=6,
            num_heads=gcn_heads
        )
        
        node_dim = gcn_hidden * gcn_heads
        
        def make_attn_branch(cell_dim):
            attn = CellDrugAttention(node_dim, cell_dim)
            proj = nn.Sequential(
                nn.Linear(node_dim + cell_dim + embed_dim, 512),
                nn.ReLU(),
                nn.Linear(512, embed_dim),
                nn.ReLU()
            )
            return attn, proj
            
        self.Attn_gep, self.Proj_gep = make_attn_branch(self.cell_dims['gep'])
        self.Attn_cnv, self.Proj_cnv = make_attn_branch(self.cell_dims['cnv'])
        self.Attn_mut, self.Proj_mut = make_attn_branch(self.cell_dims['mut'])
        
        self.cell_emb = nn.Embedding(num_cells, embed_dim)
        
        interaction_dim = (gcn_hidden * gcn_heads) + fp_emb_dim
        
        self.E_res = nn.Sequential(
                nn.Linear(interaction_dim + embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU()
        )
        
        self.f1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim * 2),
            nn.Dropout(dropout),
            
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
            
            nn.Linear(embed_dim, 1)
        )
        
        total_cell_feat_dim = self.cell_dims['gep'] + self.cell_dims['cnv'] + self.cell_dims['mut']
        in_dim_f2 = embed_dim * 4 + total_cell_feat_dim + fp_emb_dim
        
        self.f2 = nn.Sequential(
            nn.Linear(in_dim_f2, 2048),
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
        
        self.f3 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim * 2),
            nn.Dropout(dropout),
            
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
            
            nn.Linear(embed_dim, 1)
        )
        
        self.Dec_gep = Reconstructor(embed_dim, self.cell_dims['gep'])
        self.Dec_cnv = Reconstructor(embed_dim, self.cell_dims['cnv'])
        self.Dec_mut = Reconstructor(embed_dim, self.cell_dims['mut'])
        
        self.C_res = CellClassifier(embed_dim, num_cells, dropout=dropout)
        
    def forward(self, drug_graph, drug_feat, cell_feat_dict, opt_feat, cell_id, mask, alpha=1.0):
        z_inv, h_nodes, fp_emb, g_emb = self.E_inv_attn(drug_graph, drug_feat)
        
        def process_branch(cell_feat, attn_module, proj_module, decoder):
            h_attn_agg = attn_module(drug_graph, h_nodes, cell_feat)
            
            combined = torch.cat([h_attn_agg, cell_feat, z_inv], dim=1)
            
            z_expl = proj_module(combined)
            
            rec_feat = decoder(z_expl)
            loss_rec = F.mse_loss(rec_feat, cell_feat)
                         
            return z_expl, loss_rec

        z_gep, l_rec_gep = process_branch(cell_feat_dict['gep'], self.Attn_gep, self.Proj_gep, self.Dec_gep)
        z_cnv, l_rec_cnv = process_branch(cell_feat_dict['cnv'], self.Attn_cnv, self.Proj_cnv, self.Dec_cnv)
        z_mut, l_rec_mut = process_branch(cell_feat_dict['mut'], self.Attn_mut, self.Proj_mut, self.Dec_mut)

        cell_emb_vec = self.cell_emb(cell_id)
        
        combined_res = torch.cat([g_emb, fp_emb, cell_emb_vec], dim=1)
        z_res = self.E_res(combined_res)
        
        pred_cell_logits = self.C_res(z_res)
        
        pred_inv = self.f1(z_inv)
        
        z_expl_concat = torch.cat([
            z_gep, z_cnv, z_mut, z_inv, fp_emb,
            cell_feat_dict['gep'], cell_feat_dict['cnv'], cell_feat_dict['mut']
        ], dim=-1)
        pred_expl = self.f2(z_expl_concat)
        
        pred_res = self.f3(z_res)
        
        pred = pred_inv + pred_expl + pred_res
        
        loss_rec_total = l_rec_gep + l_rec_cnv + l_rec_mut
        return pred, loss_rec_total, pred_cell_logits, (pred_inv, pred_expl, pred_res), (z_inv, z_gep, z_cnv, z_mut, z_res)
