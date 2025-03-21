"""Unified interface to all dynamic graph model experiments"""
import math
import time
import sys
import random
import argparse

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder

from my_dataloader import data_load, Temporal_Dataloader, Dynamic_Dataloader, Temporal_Splitting


class LR(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=True)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


random.seed(222)
np.random.seed(222)
torch.manual_seed(222)

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on node classification')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=30, help='batch_size')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_degree', type=int, default=50, help='number of neighbors to sample')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--tune', action='store_true', help='parameters tunning mode, use train-test split on training data only.')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=None, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=None, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')

parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--snapshot', default=10, help='number of temporal graph in a given dataset')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
SNAPSHOT = args.snapshot
VIEW = SNAPSHOT - 2


graph, idxloader = data_load(dataset = DATA)
graph_list = Temporal_Splitting(graph=graph).temporal_splitting(time_mode='view', snapshot = SNAPSHOT, views = VIEW)
temporaloader = Dynamic_Dataloader(graph_list, graph=graph)

for sp in range(VIEW):
    g_df: Temporal_Dataloader = temporaloader.get_temporal()
    e_feat = g_df.edge_pos
    n_feat = g_df.node_pos

    time_attr = g_df.edge_attr
    val_time, test_time = list(np.quantile(time_attr, [0.70, 0.85]))

    src_l = g_df.edge_index[0]
    dst_l = g_df.edge_index[1]
    e_idx_l = np.arange(len(src_l))
    label_l = g_df.y
    ts_l = time_attr

    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())

    total_node_set = set(np.unique(g_df.edge_index))

    val_idx = int(len(src_l) * 0.85)
    valid_train_flag = (e_idx_l <= val_idx)  
    valid_val_flag = (e_idx_l > test_time) 
    assignment = np.random.randint(0, 10, len(valid_train_flag))
    valid_train_flag *= (assignment >= 2)
    valid_val_flag *= (assignment < 2)

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]

    # use the validation as test dataset
    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    val_ts_l = ts_l[valid_val_flag]
    val_e_idx_l = e_idx_l[valid_val_flag]
    val_label_l = label_l[valid_val_flag]


    ### Initialize the data structure for graph and edge sampling
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

    # full graph with all the data for the test and validation purpose
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

    ### Model initialize
    device = torch.device('cuda:{}'.format(GPU))
    tgan = TGAN(train_ngh_finder, n_feat, e_feat,
                num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
                seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
    tgan = tgan.to(device)


    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / BATCH_SIZE)
    print('num of training instances: {}'.format(num_instance))
    print('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list) 

    print('loading saved TGAN model')
    # model_path = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'
    # tgan.load_state_dict(torch.load(model_path))
    tgan.eval()
    print('TGAN models loaded')
    print('Start training node classification task')

    lr_model = LR(n_feat.shape[1])
    lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)
    lr_model = lr_model.to(device)
    tgan.ngh_finder = full_ngh_finder
    idx_list = np.arange(len(train_src_l))
    lr_criterion = torch.nn.BCELoss()
    lr_criterion_eval = torch.nn.BCELoss()

    def eval_epoch(src_l, dst_l, ts_l, label_l, batch_size, lr_model, tgan, num_layer=NODE_LAYER):
        pred_prob = np.zeros(len(src_l))
        loss = 0
        num_instance = len(src_l)
        num_batch = math.ceil(num_instance / batch_size)
        with torch.no_grad():
            lr_model.eval()
            tgan.eval()
            for k in range(num_batch):          
                s_idx = k * batch_size
                e_idx = min(num_instance - 1, s_idx + batch_size)
                src_l_cut = src_l[s_idx:e_idx]
                dst_l_cut = dst_l[s_idx:e_idx]
                ts_l_cut = ts_l[s_idx:e_idx]
                label_l_cut = label_l[s_idx:e_idx]
                size = len(src_l_cut)
                src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, num_layer)            
                src_label = torch.from_numpy(label_l_cut).float().to(device)
                lr_prob = lr_model(src_embed).sigmoid()
                loss += lr_criterion_eval(lr_prob, src_label).item()
                pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()

        auc_roc = roc_auc_score(label_l, pred_prob)
        return auc_roc, loss / num_instance



    for epoch in tqdm(range(args.n_epoch)):
        lr_pred_prob = np.zeros(len(train_src_l))
        np.random.shuffle(idx_list)
        tgan = tgan.eval()
        lr_model = lr_model.train()
        #num_batch
        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut = train_src_l[s_idx:e_idx]
            dst_l_cut = train_dst_l[s_idx:e_idx]
            ts_l_cut = train_ts_l[s_idx:e_idx]
            label_l_cut = train_label_l[s_idx:e_idx]
            
            size = len(src_l_cut)
            
            lr_optimizer.zero_grad()
            with torch.no_grad():
                src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)
            
            src_label = torch.from_numpy(label_l_cut).float().to(device)
            lr_prob = lr_model(src_embed).sigmoid()
            lr_loss = lr_criterion(lr_prob, src_label)
            lr_loss.backward()
            lr_optimizer.step()

        train_auc, train_loss = eval_epoch(train_src_l, train_dst_l, train_ts_l, train_label_l, BATCH_SIZE, lr_model, tgan)
        # test_auc, test_loss = eval_epoch(test_src_l, test_dst_l, test_ts_l, test_label_l, BATCH_SIZE, lr_model, tgan)
        #torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
        # print(f'train auc: {train_auc}, test auc: {test_auc}')

    # test_auc, test_loss = eval_epoch(test_src_l, test_dst_l, test_ts_l, test_label_l, BATCH_SIZE, lr_model, tgan)
    #torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
    # print(f'test auc: {test_auc}')




 




