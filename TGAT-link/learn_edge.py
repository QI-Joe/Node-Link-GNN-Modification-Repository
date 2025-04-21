"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse

import torch
import pandas as pd
import numpy as np
#import numba

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler
from my_dataloader import data_load, Temporal_Dataloader, Dynamic_Dataloader, Temporal_Splitting
from time_evaluation import TimeRecord
from typing import Optional, List, Dict, Tuple

def t2t1_node_alignment(t_nodes: set, t: Temporal_Dataloader, t1: Temporal_Dataloader):
    t_list = t.my_n_id.node.values
    t1_list = t1.my_n_id.node.values

    t2t1 = t_list[np.isin(t_list[:, 0], list(t_nodes)), 1].tolist()
    t1_extra = list(set(t1_list[:,1]) - set(t_list[:,1]))

    new_nodes = sorted(t2t1+t1_extra) # here the node is original nodes
    resort_nodes = t1_list[np.isin(t1_list[:,1], new_nodes), 0].tolist() # here we match the original nodes back to new idxed node
    
    t1_src = np.isin(t1.edge_index[0], resort_nodes)
    t1_dst = np.isin(t1.edge_index[1], resort_nodes)

    return t1_src*t1_dst, ~t1_src*~t1_dst

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--snapshot', default=10, type=int,help='number of temporal graph in a given dataset')

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
SNAPSHOT = args.snapshot
VIEW = SNAPSHOT - 2
epoch_tester = 5
EMB_DIM = 64

graph, idxloader = data_load(dataset = DATA, emb_size=EMB_DIM)
graph_list = Temporal_Splitting(graph=graph).temporal_splitting(time_mode='view', snapshot = SNAPSHOT, views = VIEW)
temporaloader = Dynamic_Dataloader(graph_list, graph=graph)
num_cls = max(graph.y) + 1

NODE_DIM = graph.pos[0].shape[1]
TIME_DIM = graph.pos[1].shape[1] # TIME_DIM == EDGE_DIM

device = torch.device('cuda:{}'.format(GPU))
tgan: TGAN = TGAN(num_cls=num_cls, num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
snapshot_list = list()

rscore, rpresent = TimeRecord(model_name="TGAT"), TimeRecord(model_name="TGAT")

rscore.get_dataset(DATA)
rpresent.get_dataset(DATA)
rscore.set_up_logger(name="time_logger")
rpresent.set_up_logger()
rpresent.record_start()

def eval_one_epoch(tgan, sampler, src, dst, ts):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE=30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)

            pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
            
            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)

### Load data and train val test split

tgan = TGAN(num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
 

for sp in range(VIEW):
    g_df: Temporal_Dataloader = temporaloader.get_temporal()
    e_feat = g_df.edge_pos
    n_feat = g_df.node_pos

    time_attr = g_df.edge_attr
    src_l = g_df.edge_index[0]
    dst_l = g_df.edge_index[1]
    e_idx_l = np.arange(len(src_l))
    ts_l = time_attr

    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())
    total_node_set = set(np.unique(g_df.edge_index))
    num_total_unique_nodes = len(total_node_set)

    val_idx = int(len(src_l) * 0.85)
    valid_train_flag = (e_idx_l <= val_idx)  
    valid_val_flag = (e_idx_l > val_idx)

    inductive_node_option_list = set(src_l[valid_val_flag]).union(set(dst_l[valid_val_flag]))
    mask_node_set = set(random.sample(sorted(inductive_node_option_list), int(0.1 * num_total_unique_nodes)))
    inductive_nodes = list(mask_node_set)
    
    random.seed(2025)

    mask_src_flag = np.isin(src_l, inductive_nodes)
    mask_dst_flag = np.isin(dst_l, inductive_nodes)
    
    none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)

    valid_train_flag = valid_train_flag * (none_node_flag > 0)

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    # train_label_l = label_l[valid_train_flag]

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_src_l).union(train_dst_l)
    assert(len(train_node_set - mask_node_set) == len(train_node_set))
    new_node_set = total_node_set - train_node_set

    # select validation and test dataset
    is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
    nn_val_flag = valid_val_flag * is_new_node_edge
    
    t1_test = temporaloader.get_T1graph(sp)

    test_src = t1_test.edge_index[0]
    test_dst = t1_test.edge_index[1]
    test_e_idx = np.arange(len(test_src))
    test_ts = t1_test.edge_attr
    t1_label_src = t1_test.y[test_src]
    t1_laebl_dst = t1_test.y[test_dst]
    valid_test_flag = np.ones(test_e_idx.shape).astype(bool)

    t1_new_node_edge, t1_old_node_edge = t2t1_node_alignment(new_node_set, g_df, t1_test)
    nn_test_flag = valid_test_flag * t1_new_node_edge

    # validation and test with all edges
    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    val_ts_l = ts_l[valid_val_flag]
    val_e_idx_l = e_idx_l[valid_val_flag]
    # val_label_l = label_l[valid_val_flag]

    test_src_l = test_src[valid_test_flag]
    test_dst_l = test_dst[valid_test_flag]
    test_ts_l = t1_test.edge_attr[valid_test_flag]
    test_e_idx_l = test_e_idx[valid_test_flag]

    # validation and test with edges that at least has one new node (not in training set)
    nn_val_src_l = src_l[nn_val_flag]
    nn_val_dst_l = dst_l[nn_val_flag]
    nn_val_ts_l = ts_l[nn_val_flag]
    nn_val_e_idx_l = e_idx_l[nn_val_flag]
    # nn_val_label_l = label_l[nn_val_flag]

    nn_test_src_l = test_src[nn_test_flag]
    nn_test_dst_l = test_dst[nn_test_flag]
    nn_test_ts_l = t1_test.edge_attr[nn_test_flag]
    nn_test_e_idx_l = test_e_idx[nn_test_flag]
    # nn_test_label_l = test_label[nn_test_flag]

    ### Initialize the data structure for graph and edge sampling
    # build the graph for fast query
    # graph only contains the training data (with 10% nodes removal)
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

    max_test_id = max(test_src.max(), test_dst.max())
    full_adj_list_test = [[] for _ in range(max_test_id + 1)]
    for tsrc, tdst, teidx, tts in zip(test_src, test_dst, test_e_idx, test_ts):
        full_adj_list_test[tsrc].append((tdst, teidx, tts))
        full_adj_list_test[tdst].append((tsrc, teidx, tts))
    full_test_ngh_finder = NeighborFinder(full_adj_list_test, uniform=UNIFORM)

    train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
    val_rand_sampler = RandEdgeSampler(src_l, dst_l)
    nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
    test_rand_sampler = RandEdgeSampler(src_l, dst_l)
    nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)

    print(f"training edges: {len(train_src_l)} \
          \nall edges {len(src_l)}\
          \ninductive train learning edges {len(nn_val_src_l)}\
          \ntest edges: {len(test_src)}\
          \ninductive test learning edges: {len(nn_test_src_l)}\n")


    ### Model initialize
    device = torch.device('cuda:{}'.format(GPU))
    
    tgan.temporal_update(train_ngh_finder, n_feat = n_feat, e_feat=e_feat)
    optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    tgan = tgan.to(device)

    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    print('num of training instances: {}'.format(num_instance))
    print('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list) 

    early_stopper = EarlyStopMonitor()
    score_recorder = list()
    for epoch in range(NUM_EPOCH):
        # Training 
        # training use only training graph
        tgan.ngh_finder_update(train_ngh_finder)
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        np.random.shuffle(idx_list)
        print('start {} epoch'.format(epoch))
        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx]
            ts_l_cut = train_ts_l[s_idx:e_idx]
            # label_l_cut = train_label_l[s_idx:e_idx]
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)
            
            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)
            
            optimizer.zero_grad()
            tgan = tgan.train()
            pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
        
            loss = criterion(pos_prob, pos_label)
            loss += criterion(neg_prob, neg_label)
            
            loss.backward()
            optimizer.step()
            # get training results
            with torch.no_grad():
                tgan = tgan.eval()
                pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))
                f1.append(f1_score(true_label, pred_label))
                m_loss.append(loss.item())
                auc.append(roc_auc_score(true_label, pred_score))

        # validation phase use all information
        if (epoch+1) % epoch_tester !=0:
            continue
        
        tgan.ngh_finder_update(full_ngh_finder)
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch(tgan, val_rand_sampler, val_src_l, 
        val_dst_l, val_ts_l)

        nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch(tgan, val_rand_sampler, nn_val_src_l, 
        nn_val_dst_l, nn_val_ts_l)
            
        print('epoch: {}:'.format(epoch))
        print('Epoch mean loss: {}'.format(np.mean(m_loss)))
        print('train acc: {}, val acc: {}, new node val acc: {}'.format(np.mean(acc), val_acc, nn_val_acc))
        print('train auc: {}, val auc: {}, new node val auc: {}'.format(np.mean(auc), val_auc, nn_val_auc))
        print('train ap: {}, val ap: {}, new node val ap: {}'.format(np.mean(ap), val_ap, nn_val_ap))
        print('train f1: {}, val f1: {}, new node val f1: {}'.format(np.mean(f1), val_f1, nn_val_f1))

        if early_stopper.early_stop_check(val_ap):
            print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            print(f'Loading the best model at epoch {early_stopper.best_epoch}')
            # best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            # tgan.load_state_dict(torch.load(best_model_path))
            print(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            tgan.eval()
            break

        # testing phase use all information
        tgan.test_emb_update(full_test_ngh_finder, t1_test.node_pos, t1_test.edge_pos)
        test_acc, test_ap, test_f1, test_auc = eval_one_epoch(tgan, test_rand_sampler, test_src_l, 
        test_dst_l, test_ts_l)

        nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch(tgan, nn_test_rand_sampler, nn_test_src_l, 
        nn_test_dst_l, nn_test_ts_l)
        
        validation_dict = {
            "val_acc": val_acc,
            "val_f1": val_f1,
            "test_acc": test_acc,
            "precision": test_ap,  # Assuming test_ap is used as precision
            "test_roc_auc": test_auc,    # Assuming test_auc is used as recall
            "f1": test_f1  # Assuming test_new_new_ap is used as F1
        }
        
        validation_dict = {**validation_dict, **{"test_new_new_acc": nn_test_acc, "test_new_new_auc": nn_test_auc, "test_new_new_f1": nn_test_f1}}

        print('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))
        print('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap))
        score_recorder.append(validation_dict)
        tgan.train_val_emb_restore()
    
    tgan.test_emb_update(full_test_ngh_finder, t1_test.node_pos, t1_test.edge_pos)
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch(tgan, test_rand_sampler, test_src_l, 
    test_dst_l, test_ts_l)

    nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch(tgan, nn_test_rand_sampler, nn_test_src_l, 
    nn_test_dst_l, nn_test_ts_l)

    print('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))
    print('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap))

    temporaloader.update_event(sp)
    rpresent.score_record(temporal_score_=score_recorder, node_size=g_df.num_nodes, temporal_idx=sp, epoch_interval=epoch_tester, mode='i')
    snapshot_list.append(score_recorder)
    
rpresent.record_end()
rscore.record_end()
rscore.fast_processing('i', snapshot_list)

 




