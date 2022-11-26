import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from transformers import XLNetConfig, XLNetModel, XLNetTokenizer
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers.modeling_utils import SequenceSummary
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model_utils import Multi_DGCN

class UPF_dgcn(nn.Module):
    '''
    The proposed UPF model
    '''
    def __init__(self, args):
        super(UPF_dgcn, self).__init__()
        self.args = args

        # Bert
        if args.pretrain_type == 'bert':
            config = BertConfig.from_pretrained(args.model_dir)
            self.pretrain_models = BertModel.from_pretrained(
                args.model_dir, config=config, from_tf=False)
        # XLNet
        elif args.pretrain_type == 'xlnet':
            config = XLNetConfig.from_pretrained(args.model_dir)
            self.pretrain_models = XLNetModel.from_pretrained(args.model_dir, config=config, from_tf=False)
            # self.sequence_summary = SequenceSummary(config)

        # Roberta
        elif args.pretrain_type == 'roberta':
            config = RobertaConfig.from_pretrained(args.model_dir)
            self.pretrain_models = RobertaModel.from_pretrained(
                args.model_dir, config=config, from_tf=False)
        else:
            raise NotImplementedError

        # an additional node
        self.dropout = nn.Dropout(args.dropout)
        self.args.embedding_dim = args.d_model  # 768
        self.multi_dgcn = Multi_DGCN(args).to(args.device)

    def forward(self, post_tokens_id):
        if self.args.pretrain_type == 'bert':
            pad_id = 0
        elif self.args.pretrain_type == 'xlnet':
            pad_id = 5
        elif self.args.pretrain_type == 'roberta':
            pad_id = 1
        else:
            raise NotImplementedError
        
        amask = (post_tokens_id != pad_id).float() # (B, N, L)
        pmask = (amask.sum(-1) > 0).float()  # (B, N)

        input_ids = post_tokens_id.view(-1, self.args.max_len)  # (B*N, L)
        att_mask = amask.view(-1, self.args.max_len)  # (B*N, L)
        '''
        if self.args.pretrain_type == 'bert':
            encoder_outputs = self.pretrain_models(input_ids=input_ids, attention_mask=att_mask, output_hidden_states=True)
            first_layer_token, last_layer_token = encoder_outputs[-1][1], encoder_outputs[-1][-1]
            first_layer_rep = first_layer_token.masked_fill((1-att_mask.unsqueeze(-1).expand(-1, self.args.max_len, self.args.embedding_dim)).bool(), 0)
            last_layer_rep = last_layer_token.masked_fill((1-att_mask.unsqueeze(-1).expand(-1, self.args.max_len, self.args.embedding_dim)).bool(), 0)
            post_rep = torch.cat((first_layer_rep, last_layer_rep), dim=-2)
            cls_token = post_rep.mean(dim=-2)
        else:
        '''
        cls_token = self.pretrain_models(input_ids=input_ids, attention_mask=att_mask)[0][:, :1]
        cls_token = cls_token.view(-1, self.args.max_post, self.args.embedding_dim)  # (B, N, D)        
        '''
        if self.args.pretrain_type == 'bert':
            outputs = self.pretrain_models(input_ids=input_ids, attention_mask=att_mask)[0]  # (B*N, L, D)
            outputs = torch.index_select(outputs, 1, torch.LongTensor([0]).cuda()).squeeze(1)
        elif self.args.pretrain_type == 'xlnet':
            outputs = self.pretrain_models(input_ids=input_ids, attention_mask=att_mask)[0]  # (B*N, L, D)
            outputs = torch.index_select(outputs, 1, torch.LongTensor([0]).cuda()).squeeze(1)
        else:
            raise NotImplementedError
        '''

        # add an additional node for classification
        c_node = (cls_token.masked_fill((1 - pmask[:, :, None].expand_as(cls_token)).bool(), 0).sum(dim=1) / pmask.sum(
            dim=-1)[:, None].expand(-1, self.args.embedding_dim)).unsqueeze(1)  # (B, D)
        
        # outputs = torch.cat((cls_token, c_node), 1)

        logit1, logit2, logit3, logit4, l0_attr, rs = self.multi_dgcn(pmask, cls_token, c_node)
        return logit1, logit2, logit3, logit4, l0_attr, rs