import math
import sys
import torch.nn as nn
import torch
from torch.nn import ReLU, Linear

from utils_gm.torch_utils.hard_concrete import HardConcrete
from utils_gm.torch_utils.multiple_inputs_layernorm_linear import MultipleInputsLayernormLinear
from utils_gm.torch_utils.squeezer import Squeezer


class GraphMaskAdjMatProbe(nn.Module):

    device = None

    def __init__(self, v_dim, h_dim, train):
        super(GraphMaskAdjMatProbe, self).__init__()

        training = train.find("train")!=-1
        self.hard_gates = HardConcrete(train=training)

        self.transforms = torch.nn.Sequential(
                Linear(v_dim, h_dim, False),
                ReLU(),
                Linear(h_dim, v_dim, False),
            )


    def forward(self, nodes_emb, adj_mat):

        srcs = self.transforms(nodes_emb)

        squeezed_a = torch.bmm(srcs, nodes_emb.transpose(-1, -2))
        
        if False: #undirected graph for ablation experiments
            tril_a = torch.tril(squeezed_a)
            tril_no_diag = torch.tril(squeezed_a, diagonal=-1)
            squeezed_a = tril_a + tril_no_diag.transpose(-1, -2)

        gate, penalty = self.hard_gates(squeezed_a, summarize_penalty=False)

        gate = gate * (adj_mat > 0).float()
        penalty_norm = (adj_mat > 0).sum().float()
        penalty = (penalty * (adj_mat > 0).float()).sum() / (penalty_norm + 1e-8)

        return gate, penalty
