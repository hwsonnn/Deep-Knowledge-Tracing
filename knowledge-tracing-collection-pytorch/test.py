import os
import argparse

import pickle
import json

import numpy as np
import pandas as pd

import torch 
# from torch.utils.data import DataLoader, random_split

from models.dkt import DKT

from data_loaders.assist2009 import ASSIST2009

from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics

from collections import defaultdict


def match_seq_len_small(q_seq, r_seq, seq_len, pad_val=-1):
    proc_q_seqs = []
    proc_r_seqs = []

    i = 0
    while i + seq_len + 1 < len(q_seq):
        proc_q_seqs.append(q_seq[i:i + seq_len + 1])
        proc_r_seqs.append(r_seq[i:i + seq_len + 1])

        i += seq_len + 1

    proc_q_seqs.append(
        np.concatenate(
            [
                q_seq[i:],
                np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
            ]
        )
    )
    proc_r_seqs.append(
        np.concatenate(
            [
                r_seq[i:],
                np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
            ]
        )
    )

    return proc_q_seqs, proc_r_seqs
    
    
from torch.nn.utils.rnn import pad_sequence

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

def get_test_input(q_seq, r_seq, pad_val=-1, seq_len=100):
    q_seq, r_seq = match_seq_len_small(q_seq, r_seq, seq_len)
#     print(q_seq, r_seq)
    
    q_seqs = [FloatTensor(q_seq[0][:-1])]
    r_seqs = [FloatTensor(r_seq[0][:-1])]
    qshft_seqs = [FloatTensor(q_seq[0][1:])]
    rshft_seqs = [FloatTensor(r_seq[0][1:])]
    
    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )

    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs, r_seqs, qshft_seqs, rshft_seqs = \
        q_seqs * mask_seqs, r_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        rshft_seqs * mask_seqs

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs
    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--q_seq', default='1,2,3,4,5', type=str, help='question sequence by 1 student')
    parser.add_argument('--r_seq', default='1,0,1,0,1', type=str, help='correct sequence by 1 student')
    parser.add_argument('--seq_len', default='10', type=int, help='sequence length')
    
    args = parser.parse_args()

    return args


def get_list(s):
    '''
        1,2,3,4,5 형태
    '''
    ret = s.split(',')
    ret = list(map(int, ret))
    
    return ret


def main(args):
    dataset = ASSIST2009(100)
    
    with open("config.json") as f:
        config = json.load(f)
        model_config = config["dkt"]
        train_config = config["train_config"]
    
    model = DKT(dataset.num_q, **model_config)
    model.load_state_dict(torch.load('./ckpts/dkt/ASSIST2009/model.ckpt'))
    model.eval()
    
    DATASET_DIR = "datasets/ASSIST2009/"
    with open(os.path.join(DATASET_DIR, "q2idx.pkl"), "rb") as f:
        q2idx = pickle.load(f)
    idx2q = {v:k for k,v in q2idx.items()}

    args.q_seq = get_list(args.q_seq)
    args.r_seq = get_list(args.r_seq)
#     print(args.q_seq)
    
    args.q_seq = np.array(args.q_seq)
    args.r_seq = np.array(args.r_seq)
    print("문제 skill id 시퀀스: {}".format(args.q_seq))
    print("각 문제 정오답 여부: {}\n".format(args.r_seq))
    
    data = get_test_input(args.q_seq, args.r_seq)
    q, r, qshft, rshft, m = data
    
    ############################################################
    outputs = model(q.long(), r.long()) 
    outputs = (outputs * one_hot(qshft.long(), model.num_q)).sum(-1)
    
    outputs = torch.masked_select(outputs, m).detach().cpu() 
    t = torch.masked_select(rshft, m).detach().cpu()
    
    idx_q = torch.masked_select(qshft, m).detach().cpu()
    idx_q = idx_q.tolist()
    
#     try:
#         roc_auc_s = metrics.roc_auc_score(y_true=t.numpy(), y_score=outputs.numpy())
#         print("roc_auc_score : {}\n".format(roc_auc_s))
# #         fpr, tpr, thresholds = metrics.roc_curve(y_true=t.numpy(), y_score=outputs.numpy())
# #         optimal_idx = np.argmax(tpr-fpr)
# #         optimal_threshold = thresholds[optimal_idx]
# #         print('threshold: ', optimal_threshold,'\n\n')
#     except:
#         roc_auc_s = "all respond are same"
#         print("전체 roc_auc_score : {}\n".format(roc_auc_s))
    
    
    
    r_dict = defaultdict(list)
    for i, r in enumerate(t.tolist()):
        r_dict[idx_q[i]].append(r)

    q_dict = defaultdict(list)
    for j, p in enumerate(outputs.tolist()):
        q_dict[idx_q[j]].append(p)

#     q_mean_dict = {}
#     for k,v in q_dict.items():
#         q_mean_dict[k] = np.mean(v)
    
    q_last_dict = {}
    for k, v in q_dict.items():
        q_last_dict[k] = v[-1]
        
    ############################################################
    last_r_dict = dict()
    for idx, corr in enumerate(t.tolist()):
        last_r_dict[idx_q[idx]]=int(corr)
    print('각 skill 별 실제 마지막 정오답: ',last_r_dict, '\n')
    
    y_true = []
    for v in last_r_dict.values():
        y_true.append(v)

    y_pred_last = []
    for v in q_last_dict.values():
        y_pred_last.append(v)

#     y_pred_mean = []
#     for v in q_mean_dict.values():
#         y_pred_mean.append(v)

    try:
        print("last roc_auc_score: ", metrics.roc_auc_score(y_true=np.array(y_true), y_score=np.array(y_pred_last)))
        fpr, tpr, thresholds = metrics.roc_curve(y_true=np.array(y_true), y_score=np.array(y_pred_last))
        optimal_idx = np.argmax(tpr-fpr)
        optimal_threshold = thresholds[optimal_idx]
        print('threshold: ', optimal_threshold,'\n\n')
        
    except:
        if y_true[0] == 1: print('모두 맞을 듯\n')
        else: print('모두 틀릴 듯\n')
    ############################################################   
        
#     print("각 skill 마다 다음 번에 맞출 확률:\n{}\n".format(q_mean_dict))
    print("각 skill 마다 다음 번에 맞출 확률:\n{}\n".format(q_last_dict))
    
    print("각 skill 마다 실제 정답률:")
    for k,v in r_dict.items():
        print("{}: {}".format(k,np.mean(v)), end=', ')
    print('\n')
        
    
       
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args) 
