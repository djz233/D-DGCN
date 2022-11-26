# coding=utf-8
import argparse
import logging
import os
import random

import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from transformers import BertConfig, BertForTokenClassification, BertTokenizer
from transformers import XLNetModel, XLNetConfig,  XLNetTokenizer
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader

from datasets import load_split_datasets
from model import UPF_dgcn

from trainer import train, get_labels
from collections import OrderedDict
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument('--output_dir', type=str, default='output1',
                        help='Directory to store intermedia data, such as models, records.')

    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes of each personality dimension.')
    parser.add_argument('--option', type=str, default='train', help='train or test the model')

    parser.add_argument('--seed', type=int, default=321,
                        help='random seed for initialization')
    parser.add_argument('--task', type=str, default='kaggle', choices=['kaggle', 'pandora'], 
                        help='task name')
    #parser.add_argument('--max_post', type=int, default=50,
    #                    help='Number of max post. decide by task')
    parser.add_argument('--max_len', type=int, default=70,
                        help='Number of max len.')

    # Model parameters

    # parser.add_argument('--model_dir', type=str, default='../../bert-gcn/xlnet-base-cased',
    #                     help='Path to pre-trained model.')
    parser.add_argument('--model_dir', type=str, default='../scr4/bert-base-cased',
                        help='Path to pre-trained model.')
    # parser.add_argument('--model_dir', type=str, default='../../bert-gcn/roberta-base',
    #                     help='Path to pre-trained model.')

    parser.add_argument('--pretrain_type', type=str,default='bert', choices=['bert', 'xlnet', 'roberta'])

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for embedding.')
    parser.add_argument('--d_model', type=int, default=768,
                        help='Model dimension of Bert or XLNet.')
    parser.add_argument('--num_mlps', type=int, default=2,
                        help='Number of mlps in the last of model.')
    parser.add_argument('--final_hidden_size', type=int, default=128,
                        help='Hidden size of attention.')
    # GCN
    parser.add_argument('--gcn_hidden_size', type=int, default=768,
                        help='hidden_size for gcn.')
    parser.add_argument('--gcn_dropout', type=float, default=0.2,
                        help='Dropout rate for gcn.')
    parser.add_argument('--gcn_mem_dim', type=int, default=64,
                        help='hidden size of gnn')
    parser.add_argument('--gcn_num_layers', type=int, default=2,
                        help='Number of layers of gat.')

    # Training parameters
    parser.add_argument("--all_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--all_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,   # 2e-5
                        help="The initial learning rate for pretrain models")
    parser.add_argument("--other_learning_rate", default=1e-3, type=float,  # 1e-4
                        help="The initial learning rate for other components.")
    parser.add_argument('--alpha_learning_rate', default=1e-2, type=float,
                        help="alpha_optimizer_lr in LagrangianOptimization")
    parser.add_argument("--gm_learning_rate", default=1e-5, type=float,
                        help="L2C learning rate")
    parser.add_argument('--max_alpha', default=100, type=int, 
                        help="max_alpha in LagrangianOptimization")                    
    
    #parser.add_argument("--weight_decay", default=0.0, type=float,
    #                    help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float, 
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=25,
                        help="Log every X updates steps.")
    parser.add_argument('--l0', action='store_true', help='whether to use L0 loss function')
    parser.add_argument('--single_hop', action='store_true', help='single_hop variant for ablation experiments')
    parser.add_argument('--no_special_node', action='store_true', help='remove-special node variant for ablation experiments')
    parser.add_argument('--no_dart', action='store_true', help='no post-training')
    
    return parser.parse_args()


def check_args(args):
    '''
    eliminate confilct situations
    
    '''
    logger.info(vars(args))

if __name__ == "__main__":
    # Parse args
    args = parse_args()
    if args.task == 'kaggle':
        args.max_post = 50
    elif args.task == 'pandora':
        args.max_post = 100
    check_args(args)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        #filename=log_file,
                        level=logging.INFO)    

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Setup CUDA, GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('hyperparameter = {}'.format(args))

    # Set seed
    set_seed(args)
    # load pretrained model and tokenizer
    if args.pretrain_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_dir)
        args.tokenizer = tokenizer
    elif args.pretrain_type == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained(args.model_dir)
        args.tokenizer = tokenizer
    elif args.pretrain_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_dir)
        args.tokenizer = tokenizer
    else:
        raise NotImplementedError

    if args.option == 'train':
        # Load datasets
        train_dataset, eval_dataset, test_dataset = load_split_datasets(args)
        # Build Model
        model = UPF_dgcn(args)
        '''
        load parameters
        '''
        if not args.no_dart:
            name_ptm = 'pretrain_models.'
            original_msd = torch.load(os.path.join('bert-pretrained', 'bert-pretrained.pth'), map_location = 'cpu')
            only_bert_msd = {}
            for key, values in original_msd.items():
                if 'module.ptm.bert.' in key:
                    only_bert_msd[key.replace('module.ptm.bert.', name_ptm)] = values
            total_msd = model.state_dict()
            total_msd.update(only_bert_msd)
            model.load_state_dict(total_msd)
        '''
        '''
        model = nn.DataParallel(model)
        model.to(args.device)
   
        # # Train
        _, _, all_eval_results = train(args, train_dataset, model, eval_dataset, test_dataset)

    elif args.option == 'test': # single-GPU for test
        # Load datasets
        train_dataset, eval_dataset, test_dataset = load_split_datasets(args)
        # Load Model
        model = UPF_dgcn(args)
        model.to(args.device)

        model_state_dict = torch.load(os.path.join(args.output_dir, 'best_f1_net.pth'))

        new_state_dict = OrderedDict()
        for k,v in model_state_dict.items():
            name = k[7:]  # remove 'module'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


        preds1, preds2, preds3, preds4, out_label_ids1, out_label_ids2, out_label_ids3, out_label_ids4 = get_labels(args, model, test_dataset)

        save_data= pd.DataFrame(out_label_ids1,columns=['T1'])
        save_data['T2'] = out_label_ids2
        save_data['T3'] = out_label_ids3
        save_data['T4'] = out_label_ids4
        save_data['P1'] = preds1
        save_data['P2'] = preds2
        save_data['P3'] = preds3
        save_data['P4'] = preds4
        save_data.to_excel('../output/test.xlsx')
    else:
        raise NotImplementedError

