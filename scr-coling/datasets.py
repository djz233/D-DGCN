from torch.nn.utils.rnn import pad_sequence
import logging
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import transformers
logger = logging.getLogger(__name__)
from sklearn.model_selection import StratifiedKFold, KFold
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from transformers import XLNetModel, XLNetConfig,  XLNetTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from tqdm import tqdm


def load_split_datasets(args):
    # train set
    train = pickle.load(open('../data/'+args.task+'/train.pkl', 'rb'))
    train_text = train['posts_text']
    train_label = train['annotations']
    #eval = test = train
    # eval set
    eval = pickle.load(open('../data/'+args.task+'/eval.pkl', 'rb'))
    eval_text = eval['posts_text']
    eval_label = eval['annotations']
    # test set
    test = pickle.load(open('../data/'+args.task+'/test.pkl', 'rb'))
    test_text = test['posts_text']
    test_label = test['annotations']

    # process
    deal_train_data = process_data(train_text, train_label)
    deal_eval_data = process_data(eval_text, eval_label)
    deal_test_data = process_data(test_text, test_label)
    train_dataset = MBTI_Dataset(deal_train_data, args)
    #eval_dataset, test_dataset = train_dataset, train_dataset
    eval_dataset = MBTI_Dataset(deal_eval_data, args)
    test_dataset = MBTI_Dataset(deal_test_data, args)
    return train_dataset, eval_dataset, test_dataset

def process_data(poster, label):
    label_lookup = {'E': 1, 'I': 0, 'S': 1, 'N':0, 'T': 1, 'F': 0, 'J': 1, 'P':0}
    persona_lookup = {}
    poster_data = [{'posts': t, 'label0': label_lookup[list(label[i])[0]],
                    'label1': label_lookup[list(label[i])[1]],'label2': label_lookup[list(label[i])[2]],
                    'label3': label_lookup[list(label[i])[3]]} for i,t in enumerate(poster)]
    I,E,S,N,T,F,P,J=0,0,0,0,0,0,0,0
    for t in label:
        if 'I' in t:
            I+=1
        if 'E' in t:
            E += 1
        if 'S' in t:
            S+=1
        if 'N' in t:
            N+=1
        if 'T' in t:
            T+=1
        if 'F' in t:
            F+=1
        if 'P' in t:
            P+=1
        if 'J' in t:
            J+=1
        if t not in persona_lookup:
            persona_lookup[t] = 1
        else:
            persona_lookup[t] += 1
    print('I', I)
    print('E', E)
    print('S', S)
    print('N', N)
    print('T', T)
    print('F', F)
    print('P', P)
    print('J', J)
    #print("persona number:", persona_lookup)
    return poster_data


class MBTI_Dataset(Dataset):

    def __init__(self, data, args):
        self.data = data
        self.args = args
        if self.args.pretrain_type == 'bert':
            self.pad, self.cls = self.args.tokenizer.convert_tokens_to_ids(['[PAD]', '[CLS]'])
        elif self.args.pretrain_type == 'xlnet':
            self.pad, self.cls = self.args.tokenizer.convert_tokens_to_ids(['<pad>', '<cls>'])
        elif self.args.pretrain_type == 'roberta':
            self.pad, self.cls = self.args.tokenizer.convert_tokens_to_ids(['<pad>', '<s>'])
        else:
            raise NotImplementedError

        self.convert_features()

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text):
        '''
        '''
        return self.args.tokenizer.build_inputs_with_special_tokens(self.args.tokenizer.convert_tokens_to_ids(self.args.tokenizer.tokenize(text)))

    def __getitem__(self, idx):
        e = self.data[idx]

        items = e['post_tokens_id'], e['label0'], e['label1'], e['label2'], e['label3']

        items_tensor = tuple(torch.tensor(t) for i,t in enumerate(items))

        return items_tensor

    def convert_feature(self, i):
        """
        convert sentence to feature.
        """

        post_tokens_id=[]

        for post in self.data[i]['posts'][:self.args.max_post]:

            input_ids = self._tokenize(post)
            pad_len = self.args.max_len - len(input_ids)

            if pad_len > 0:
                if self.args.pretrain_type == 'bert':
                    input_ids += [self.pad] * pad_len
                elif self.args.pretrain_type == 'xlnet':
                    input_ids = [input_ids[-1]] + input_ids[:-1]
                    input_ids += [self.pad] * pad_len
                elif self.args.pretrain_type == 'roberta':
                    input_ids += [self.pad] * pad_len
                else:
                    raise NotImplementedError
            else:
                if self.args.pretrain_type == 'bert':
                    input_ids = input_ids[:self.args.max_len - 1] + input_ids[-1:]
                elif self.args.pretrain_type == 'xlnet':
                    input_ids = [input_ids[-1]]+ input_ids[:self.args.max_len - 2] + [input_ids[-2]]
                elif self.args.pretrain_type == 'roberta':
                    input_ids = input_ids[:self.args.max_len - 1] + input_ids[-1:]
                else:
                    raise NotImplementedError

            assert (len(input_ids) == self.args.max_len)

            post_tokens_id.append(input_ids)

        real_post = len(post_tokens_id)
        for j in range(self.args.max_post-real_post):
            post_tokens_id.append([self.pad]*self.args.max_len)

        self.data[i]['post_tokens_id'] = post_tokens_id


    def convert_features(self):
        '''
        Convert sentence to ids.
        '''
        for i in tqdm(range(len(self.data))):

            self.convert_feature(i)

def my_collate(batch):
    '''
    Turn all into tensors.
    '''
    post_tokens_id, label0, label1, label2, label3= zip(*batch)# from Dataset.__getitem__()
    post_tokens_id = torch.stack(post_tokens_id)
    label0 = torch.tensor(label0)
    label1 = torch.tensor(label1)
    label2 = torch.tensor(label2)
    label3 = torch.tensor(label3)


    return post_tokens_id, label0, label1, label2, label3
