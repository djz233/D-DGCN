import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from torch.autograd import Variable
from graphmask import GraphMaskAdjMatProbe as L2C

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

class Attention(nn.Module):
    def __init__(self, in_dim):

        super(Attention, self).__init__()
        self.in_dim = in_dim
        self.weight_proj = nn.Linear(in_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, feature, mask):
        att = self.tanh(self.weight_proj(feature)) # (B, N 1)
        att = att.squeeze(2)  # (B, N)
        att_score = F.softmax(mask_logits(att,mask), dim = 1)
        att_score = att_score.unsqueeze(2)
        out = torch.bmm(feature.transpose(1, 2), att_score)
        out = out.squeeze(2)
        return out


class AdjMatrix(nn.Module):
    """
    learn-to-connect module
    """

    def __init__(self, args, in_dim, hidden_size=64, mem_dim=300):
        super(AdjMatrix, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.threshold = args.threshold
        self.leakyrelu = nn.LeakyReLU(1e-2)

        a_layers = [
            nn.Linear(2 * mem_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 2)]
        self.afcs = nn.Sequential(*a_layers)
        layers = [nn.Linear(in_dim, self.mem_dim), nn.ReLU()]
        self.W  = nn.Sequential(*layers)

    def forward(self, pmask, feature):
        # pmask: (B, N)
        B, N = pmask.size(0), pmask.size(1)
        adj_mask = torch.eq((pmask.unsqueeze(1).repeat(1,N,1)+ pmask.unsqueeze(2).repeat(1,1,N)), 2).float() # (B, N, N)

        h = self.W(feature)  # (B, N, D)
        a_input = torch.cat([h.repeat(1, 1, N).view(  # (B, N, N*D)->(B, N*N, D) {1,1,...,2,2...} cat (B, N*N, D) {1,2,3...,1,2,3}
                  B, N * N, -1), h.repeat(1, N, 1)], dim=2)  # (B, N*N, 2*D) {1-1; 1-2; 1-3...}
        e = self.leakyrelu(self.afcs(a_input))  # (B, N*N , 1)
        adj = F.softmax(e, dim=-1) # (B, N*N * 2)

        adj = torch.index_select(adj, -1, torch.LongTensor([0]).cuda()).squeeze(2).view(adj_mask.size()) # (B, N, N)
        eyes = torch.eye(N).unsqueeze(0).repeat(B, 1, 1).float().cuda()
        adj = torch.ge((adj + eyes).view(B, N*N),  self.threshold).float()# (B, N*N)

        masked_adj = adj_mask * adj.view(B, N, -1)

        return masked_adj, adj_mask

class GCN(nn.Module):
    """
    GCN module operated on graphs
    """

    def __init__(self, args, in_dim, hidden_size, num_layers):
        super(GCN, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(args.gcn_dropout)
        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else hidden_size
            self.W.append(nn.Linear(input_dim, hidden_size))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def forward(self, adj, feature):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for l in range(self.num_layers):
            Ax = adj.bmm(feature)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](feature)  # self loop
            AxW /= denom

            gAxW = F.relu(AxW)
            # gAxW = AxW
            feature = self.dropout(gAxW) if l < self.num_layers - 1 else gAxW
        return feature, mask

class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out

class Dynamic_GCN(nn.Module):
    """
    GCN module operated on L2C
    """

    def __init__(self, args, in_dim, hidden_size, num_layers):
        super(Dynamic_GCN, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(args.gcn_dropout)
        self.layer_norm = LayerNormalization(in_dim)
        # gcn layer
        self.W = nn.ModuleList()
        self.A = nn.ModuleList()

        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else hidden_size
            self.W.append(nn.Linear(input_dim, hidden_size))
            self.A.append(L2C(input_dim, input_dim, "train")) #L2C

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def sparse_statistics(self, adj, pmask):
        p_sum = pmask.sum(-1) #(B)
        c_adj = adj[:,-1,:-1] #(B,N)
        g_adj = adj[:,:-1,:-1] #(B, N, N)
        g_edges = g_adj.sum(-1).sum(-1) #(B)
        c_node_sparsity = c_adj.sum(-1) / p_sum
        g_edge_sparsity = g_edges / p_sum**2
        return c_node_sparsity, g_edge_sparsity

    def forward(self, pmask, feature):
        # gcn layer
        p_mask = pmask.unsqueeze(-1) #(B,N+1,1)
        full_adj = p_mask.bmm(p_mask.transpose(2,1)) #(B,N+1,N+1)
        c_sparsity, g_sparsity = [], []
        total_l0loss = 0

        for l in range(self.num_layers):
            residual = feature
            adj, l0_loss = self.A[l](feature, full_adj)
            #adj = adj.detach()
            total_l0loss += l0_loss
            c_spar, g_spar = self.sparse_statistics(adj, pmask)
            c_sparsity.append(c_spar)
            g_sparsity.append(g_spar)            
            denom = torch.diag_embed(adj.sum(2))  # B, N, N
            deg_inv_sqrt = denom.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            deg_inv_sqrt = deg_inv_sqrt.detach()
            adj_ = deg_inv_sqrt.bmm(adj)
            adj_ = adj_.bmm(deg_inv_sqrt)

            Ax = adj_.transpose(-1, -2).bmm(feature)
            AxW = self.W[l](Ax)
            gAxW = F.relu(AxW)

            feature = self.dropout(gAxW) + residual
        c_sparsity = torch.stack(c_sparsity, dim=-1) #(B, L)
        g_sparsity = torch.stack(g_sparsity, dim=-1) #(B, L)
        total_l0loss /= self.num_layers
        l0_attr = (total_l0loss, c_sparsity, g_sparsity)            
        return feature, l0_attr

class Multi_DGCN(nn.Module):
    def __init__(self, args):
        # hidden_dim: the dimension fo hidden vector
        super(Multi_DGCN, self).__init__()

        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.in_dim = args.gcn_hidden_size

        self.dgcn1 = Dynamic_DeeperGCN(args, in_dim=self.args.embedding_dim, hidden_size=self.args.gcn_hidden_size,
                       num_layers=self.args.gcn_num_layers)
        self.dgcn2 = Dynamic_DeeperGCN(args, in_dim=self.args.embedding_dim, hidden_size=self.args.gcn_hidden_size,
                       num_layers=self.args.gcn_num_layers)
        self.dgcn3 = Dynamic_DeeperGCN(args, in_dim=self.args.embedding_dim, hidden_size=self.args.gcn_hidden_size,
                       num_layers=self.args.gcn_num_layers)
        self.dgcn4 = Dynamic_DeeperGCN(args, in_dim=self.args.embedding_dim, hidden_size=self.args.gcn_hidden_size,
                       num_layers=self.args.gcn_num_layers)

        last_hidden_size = self.in_dim
        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs1 = nn.Sequential(*layers)
        self.fcs2 = nn.Sequential(*layers)
        self.fcs3 = nn.Sequential(*layers)
        self.fcs4 = nn.Sequential(*layers)

        self.fc_final1 = nn.Linear(args.final_hidden_size, args.num_classes)
        self.fc_final2 = nn.Linear(args.final_hidden_size, args.num_classes)
        self.fc_final3 = nn.Linear(args.final_hidden_size, args.num_classes)
        self.fc_final4 = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, pmask, feature, c_node):
        B = pmask.size(0)
        
        #n_pmask = pmask #-u
        # New pmask: (B, N+1)
        n_pmask = torch.cat((pmask, torch.ones(B, 1).cuda()), 1) if not self.args.no_special_node else pmask
        

        #special node
        if not self.args.no_special_node:
            feature1 = torch.cat((feature, c_node), 1)  # (B, N+1, D)
            feature2 = torch.cat((feature, c_node), 1)
            feature3 = torch.cat((feature, c_node), 1)
            feature4 = torch.cat((feature, c_node), 1)
        #GCN or GAT
        if isinstance(self.dgcn1, Dynamic_GCN) or isinstance(self.dgcn1, Dynamic_GAT):
            gcn_out1,l0_attr1 = self.dgcn1(n_pmask, feature1)  # (B, N+1, D)
            gcn_out2,l0_attr2 = self.dgcn2(n_pmask, feature2)
            gcn_out3,l0_attr3 = self.dgcn3(n_pmask, feature3)
            gcn_out4,l0_attr4 = self.dgcn4(n_pmask, feature4)
        else:
            gcn_out1,rs1,_,l0_attr1 = self.dgcn1(n_pmask, feature1)  # (B, N+1, D) -u feature
            gcn_out2,rs2,_,l0_attr2 = self.dgcn2(n_pmask, feature2)
            gcn_out3,rs3,_,l0_attr3 = self.dgcn3(n_pmask, feature3)
            gcn_out4,rs4,_,l0_attr4 = self.dgcn4(n_pmask, feature4)

        if not self.args.no_special_node:
            out1 = gcn_out1[:, -1] # (B, D)
            out2 = gcn_out2[:, -1] # (B, D)
            out3 = gcn_out3[:, -1] # (B, D)
            out4 = gcn_out4[:, -1] # (B, D)
        # -u
        else:
            out1 = gcn_out1.mean(dim=-2) # (B, D)
            out2 = gcn_out2.mean(dim=-2) # (B, D)
            out3 = gcn_out3.mean(dim=-2) # (B, D)
            out4 = gcn_out4.mean(dim=-2) # (B, D)
        
        #############################################################################################
        x1 = self.dropout(out1)
        x1 = self.fcs1(x1)
        logit1 = self.fc_final1(x1)

        x2 = self.dropout(out2)
        x2 = self.fcs2(x2)
        logit2 = self.fc_final2(x2)

        x3 = self.dropout(out3)
        x3 = self.fcs3(x3)
        logit3 = self.fc_final3(x3)

        x4 = self.dropout(out4)
        x4 = self.fcs4(x4)
        logit4 = self.fc_final4(x4)

        l0_attr = (l0_attr1, l0_attr2, l0_attr3, l0_attr4)

        retain_scores = None
        #DGCN
        if isinstance(self.dgcn1, Dynamic_DeeperGCN):
            retain_scores1 = rs1[:, -1] # (B, D)
            retain_scores2 = rs2[:, -1]
            retain_scores3 = rs3[:, -1]
            retain_scores4 = rs4[:, -1]
            retain_scores = (retain_scores1, retain_scores2, retain_scores3, retain_scores4)

        return logit1, logit2, logit3, logit4, l0_attr, retain_scores

class Dynamic_DeeperGCN(nn.Module):
    """
    GCN module operated on L2C
    """

    def __init__(self, args, in_dim, hidden_size, num_layers):
        super(Dynamic_DeeperGCN, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(args.gcn_dropout)
        # self.layer_norm = LayerNormalization(in_dim)
        # gcn layer
        self.A = nn.ModuleList()

        self.proj = nn.Linear(self.in_dim, 1)

        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else hidden_size
            self.A.append(L2C(input_dim, input_dim, "train"))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def sparse_statistics(self, adj, pmask):
        p_sum = pmask.sum(-1) #(B)
        c_adj = adj[:,-1,:-1] #(B,N)
        g_adj = adj[:,:-1,:-1] #(B, N, N)
        g_edges = g_adj.sum(-1).sum(-1) #(B)
        c_node_sparsity = c_adj.sum(-1) / p_sum
        g_edge_sparsity = g_edges / p_sum**2
        return c_node_sparsity, g_edge_sparsity


    def forward(self, pmask, feature):
        B = pmask.size(0)
        p_mask = pmask.unsqueeze(-1) #(B,N+1,1)
        full_adj = p_mask.bmm(p_mask.transpose(2,1)) #(B,N+1,N+1)

        preds = []
        preds.append(feature)
        adjs = []
        c_sparsity, g_sparsity = [], []
        total_l0loss = 0


        for l in range(self.num_layers):
            residual = feature
            #if l == 0: #single-hop
            adj, l0_loss = self.A[l](feature, full_adj) 
            #adj = adj.detach()
            total_l0loss += l0_loss
            c_spar, g_spar = self.sparse_statistics(adj, pmask)
            adjs.append(adj)
            c_sparsity.append(c_spar)
            g_sparsity.append(g_spar)
            
            denom = torch.diag_embed(adj.sum(2)) # B, N, N
            deg_inv_sqrt = denom.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            deg_inv_sqrt = deg_inv_sqrt.detach()
            adj_ = deg_inv_sqrt.bmm(adj)
            adj_ = adj_.bmm(deg_inv_sqrt)

            feature = adj_.transpose(-1, -2).bmm(feature)
            preds.append(feature)
        #
        c_sparsity = torch.stack(c_sparsity, dim=-1) #(B, L)
        g_sparsity = torch.stack(g_sparsity, dim=-1) #(B, L)
        total_l0loss /= self.num_layers
        l0_attr = (total_l0loss, c_sparsity, g_sparsity)
        pps = torch.stack(preds, dim=2) # (B, N, L+1, D)
        retain_score = self.proj(pps) # (B, N, L+1, 1)
        retain_score0 = torch.sigmoid(retain_score).view(-1, self.num_layers+1, 1) # (B*N, L+1, 1)
        #retain_score0 = torch.softmax(retain_score, dim=-2).view(-1, self.num_layers+1, 1) # (B*N, L+1, 1)        
        retain_score = retain_score0.transpose(-1, -2) # (B* N+1, 1, L+1)
        out = retain_score.bmm(pps.view(-1, self.num_layers + 1, self.in_dim)) # (B*N, 1, L+1) * (B*N, L+1, D) = (B* N+1, 1, D)
        out = out.squeeze(1).view(B, -1, self.in_dim)  # (B, N+1, D)

        return out, retain_score0.view(B, -1, self.num_layers+1), torch.stack(adjs, dim = -1), l0_attr

class Dynamic_GAT(nn.Module):
    def __init__(self, args, in_dim, hidden_size, num_layers):
        super(Dynamic_GAT, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.dropout = nn.Dropout(args.gcn_dropout)
        self.args = args
        self.n_head = 6
        self.d_model = args.d_model            
        assert(self.d_model % self.n_head == 0)
        self.d_head = int(self.d_model / self.n_head)
        self.num_layers = num_layers
        self.dropout = nn.Dropout(args.gcn_dropout)

        self.W = nn.Linear(args.d_model, args.d_model, bias = False)   
        self.A = nn.ModuleList() 
        self.attn = nn.Linear(self.d_head * 2, self.n_head, bias = False)
        self.actv = nn.Tanh()

        for layer in range(num_layers):
            self.A.append(L2C(in_dim, in_dim, "train"))

    def sparse_statistics(self, adj, pmask):
        p_sum = pmask.sum(-1) #(B)
        c_adj = adj[:,-1,:-1] #(B,N)
        g_adj = adj[:,:-1,:-1] #(B, N, N)
        g_edges = g_adj.sum(-1).sum(-1) #(B)
        c_node_sparsity = c_adj.sum(-1) / p_sum
        g_edge_sparsity = g_edges / p_sum**2
        return c_node_sparsity, g_edge_sparsity

    def forward(self, pmask, feature):
        B = pmask.size(0)
        p_mask = pmask.unsqueeze(-1) #(B,N+1,1)
        full_adj = p_mask.bmm(p_mask.transpose(2,1)) #(B,N+1,N+1)

        preds = []
        preds.append(feature)
        adjs = []
        c_sparsity, g_sparsity = [], []
        total_l0loss = 0   

        for l in range(self.num_layers):     
            residual = feature
            adj, l0_loss = self.A[l](feature, full_adj) #(B,N+1,N+1)
            total_l0loss += l0_loss
            c_spar, g_spar = self.sparse_statistics(adj, pmask)
            adjs.append(adj)
            c_sparsity.append(c_spar)
            g_sparsity.append(g_spar)
            
            head_list = list(range(self.n_head))
            qk = self.W(feature).view(B, self.args.max_post + 1, self.n_head, self.d_head)
            mh_q, mh_k = qk[:,:,None,:,:].expand(-1, -1, self.args.max_post+1, -1, -1), qk[:,None,:,:,:].expand(-1, self.args.max_post+1, -1, -1, -1)
            mh_attn = self.attn(torch.cat([mh_q, mh_k], dim=-1))[:, :, :, head_list, head_list]
            mh_attn = self.actv(mh_attn)
            mh_attn = mh_attn.masked_fill((1-adj)[:,:,:,None].expand_as(mh_attn).bool(), -1e-8)
            mh_attn = F.softmax(mh_attn, dim=-2)
            mh_attn = self.dropout(mh_attn)
            mh_hid = torch.tanh(torch.einsum('bqkn, bknd->bqnd', mh_attn, qk))
            feature = residual + mh_hid.reshape(B, self.args.max_post+1, -1)

        c_sparsity = torch.stack(c_sparsity, dim=-1) #(B, L)
        g_sparsity = torch.stack(g_sparsity, dim=-1) #(B, L)
        total_l0loss /= self.num_layers
        l0_attr = (total_l0loss, c_sparsity, g_sparsity)            
        return feature, l0_attr