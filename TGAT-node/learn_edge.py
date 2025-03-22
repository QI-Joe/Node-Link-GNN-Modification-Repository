"""Unified interface to all dynamic graph model experiments"""
import math
import random
import sys
import argparse

import torch
import numpy as np
#import numba

from sklearn.metrics import accuracy_score
from eval import eval_one_epoch

from module import TGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler

from my_dataloader import data_load, Temporal_Dataloader, Dynamic_Dataloader, Temporal_Splitting
from time_evaluation import TimeRecord

def t2t1_node_alignment(t_nodes: set, t: Temporal_Dataloader, t1: Temporal_Dataloader):
    t_list = t.my_n_id.node.values
    t1_list = t1.my_n_id.node.values

    t2t1 = t_list[np.isin(t_list[:, 0], list(t_nodes)), 1].tolist()
    t1_extra = list(set(t1_list[:,1]) - set(t_list[:,1]))

    new_nodes = t2t1+t1_extra

    t1_src = np.isin(t1.edge_index[0], new_nodes)
    t1_dst = np.isin(t1.edge_index[1], new_nodes)

    return t1_src*t1_dst

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use', default='dblp')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--snapshot', default=10, type=int, help='number of temporal graph in a given dataset')

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
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
SNAPSHOT = args.snapshot
VIEW = SNAPSHOT - 2
epoch_tester = 1

EMB_DIM = 37 if DATA.lower() == "cora" else 64

### Load data and train val test split
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

for sp in range(VIEW):
    g_df: Temporal_Dataloader = temporaloader.get_temporal()
    e_feat = g_df.edge_pos
    n_feat = g_df.node_pos

    time_attr = g_df.edge_attr

    src_l = g_df.edge_index[0]
    dst_l = g_df.edge_index[1]
    e_idx_l = np.arange(len(src_l))
    label_src = g_df.y[src_l]
    label_dst = g_df.y[dst_l]
    ts_l = time_attr

    random.seed(2025)

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
    
    mask_src_flag = np.isin(src_l, inductive_nodes)
    mask_dst_flag = np.isin(dst_l, inductive_nodes)
    
    none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)

    valid_train_flag = valid_train_flag * (none_node_flag > 0)

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_src = label_src[valid_train_flag]
    train_label_dst = label_dst[valid_train_flag]

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

    t1_new_node_edge = t2t1_node_alignment(new_node_set, g_df, t1_test)
    nn_test_flag = valid_test_flag * t1_new_node_edge

    # validation and test with all edges
    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    val_ts_l = ts_l[valid_val_flag]
    val_e_idx_l = e_idx_l[valid_val_flag]
    val_label_src = label_src[valid_val_flag]
    val_label_dst = label_dst[valid_val_flag]

    test_src_l = test_src[valid_test_flag]
    test_dst_l = test_dst[valid_test_flag]
    test_ts_l = t1_test.edge_attr[valid_test_flag]
    test_e_idx_l = test_e_idx[valid_test_flag]

    # validation and test with edges that at least has one new node (not in training set)
    nn_val_src_l = src_l[nn_val_flag]
    nn_val_dst_l = dst_l[nn_val_flag]
    nn_val_ts_l = ts_l[nn_val_flag]
    nn_val_e_idx_l = e_idx_l[nn_val_flag]
    nn_val_label_src = label_src[nn_val_flag]
    nn_val_label_dst = label_dst[nn_val_flag]

    nn_test_src_l = test_src[nn_test_flag]
    nn_test_dst_l = test_dst[nn_test_flag]
    nn_test_ts_l = t1_test.edge_attr[nn_test_flag]
    nn_test_e_idx_l = test_e_idx[nn_test_flag]
    nn_test_label_src = t1_label_src[nn_test_flag]
    nn_test_label_dst = t1_laebl_dst[nn_test_flag]

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

    """
    test_src, test_dst, test_e_idx, test_ts
    """
    max_test_id = max(test_src.max(), test_dst.max())
    full_adj_list_test = [[] for _ in range(max_test_id + 1)]
    for tsrc, tdst, teidx, tts in zip(test_src, test_dst, test_e_idx, test_ts):
        full_adj_list_test[tsrc].append((tdst, teidx, tts))
        full_adj_list_test[tdst].append((tsrc, teidx, tts))
    full_test_ngh_finder = NeighborFinder(full_adj_list_test, uniform=UNIFORM)

    train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
    val_rand_sampler = RandEdgeSampler(src_l, dst_l)
    nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
    test_rand_sampler = RandEdgeSampler(test_src, test_dst)
    nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)


    ### Model initialize
    tgan.temproal_update(train_ngh_finder, n_feat=n_feat, e_feat=e_feat)
    optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    tgan = tgan.to(device)

    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    print('num of training instances: {}'.format(num_instance))
    print('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list) 

    early_stopper = EarlyStopMonitor()

    trian_acc_src, train_acc_dst = list(), list()
    score_recoder = list()
    for epoch in range(NUM_EPOCH):
        # Training 
        # training use only training graph
        m_loss = list()
        tgan.ngh_finder = train_ngh_finder
        np.random.shuffle(idx_list)
        print('start {} epoch'.format(epoch))
        for k in range(num_batch):
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx]
            ts_l_cut = train_ts_l[s_idx:e_idx]
            label_l_cut_src = torch.from_numpy(train_label_src[s_idx:e_idx]).to(device=device)
            label_l_cut_dst = torch.from_numpy(train_label_dst[s_idx:e_idx]).to(device=device)
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)
            
            optimizer.zero_grad()
            tgan = tgan.train()
            src_emb, dst_emb = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
            src_pred, dst_pred = tgan.projection(src_emb), tgan.projection(dst_emb)
        
            loss = criterion(src_emb, label_l_cut_src)+criterion(dst_emb, label_l_cut_dst)
            
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())
            # get training results
            if (k+1) % (num_batch//2)==0 & (epoch+1) % epoch_tester ==0:
                with torch.no_grad():
                    tgan.eval()
                    pred_src = src_pred.detach().argmax(dim=-1).cpu().numpy()
                    pred_dest = dst_pred.detach().argmax(dim=-1).cpu().numpy()

                    src_label, dst_label = train_label_src[s_idx:e_idx], train_label_dst[s_idx:e_idx]

                    src_train_acc = accuracy_score(src_label, pred_src)
                    dst_train_acc = accuracy_score(dst_label, pred_dest)

                    trian_acc_src.append(src_train_acc)
                    train_acc_dst.append(dst_train_acc)
                    print(f"Epoch {epoch} - Batch {k}: Source Accuracy: {src_train_acc:.4f}, Destination Accuracy: {dst_train_acc:.4f}, mean loss {np.mean(m_loss):.4f}")

        # validation phase use all information
        if (epoch+1) % epoch_tester ==0:
            tgan.ngh_finder = full_ngh_finder

            val_label_l = (val_label_src, val_label_dst)
            val_src, val_dst = eval_one_epoch(num_cls, tgan, val_rand_sampler, val_src_l,
                                                            val_dst_l, val_ts_l, val_label_l, num_neighbors=NUM_NEIGHBORS)

            nn_val_label_l = (nn_val_label_src, nn_val_label_dst)
            nn_val_src, nn_val_dst = eval_one_epoch(num_cls, tgan, nn_val_rand_sampler, nn_val_src_l, \
                                                    nn_val_dst_l, nn_val_ts_l, nn_val_label_l, num_neighbors=NUM_NEIGHBORS)
                
            print('epoch: {}:'.format(epoch))
            print('Src train acc: {:.4f}, Src val OLD NODE acc: {:.4f}'.format(np.mean(trian_acc_src), val_src["accuracy"]))
            print('Dst train acc: {:.4f}, Dst val OLD NODE acc: {:.4f}'.format(np.mean(train_acc_dst), val_dst["accuracy"]))
            print('Src OLD NODE val precision: {:.4f}, Dst OLD NODE val precision: {:.4f}'.format(val_src["precision"], val_dst["precision"]))
            
            print('Src NEW NODE val acc: {:.4f}, Dst NEW NODE val acc: {:.4f}'.format(nn_val_src["accuracy"], nn_val_dst["accuracy"]))
            print('Src NEW NODE val ap: {:.4f}, Dst NEW NODE val ap: {:.4f}'.format(nn_val_src["precision"], nn_val_dst["precision"]))
            print('Src NEW NODE precision: {:.4f}, Dst NEW NODE precision: {:.4f}'.format(nn_val_src["precision"], nn_val_dst["precision"]))

            val_ap = (val_src["precision"]+val_dst["precision"]) / 2
            nn_val_ap = (nn_val_src["precision"]+nn_val_dst["precision"]) / 2
            if early_stopper.early_stop_check(val_ap):
                print('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                print(f'Loading the best model at epoch {early_stopper.best_epoch}')
                print(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                tgan.eval()
                break
            
            tgan.ngh_finder = full_test_ngh_finder
            test_label_l = (t1_label_src, t1_laebl_dst)
            test_src, test_dst = eval_one_epoch(num_cls, tgan, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, num_neighbors=NUM_NEIGHBORS)

            nn_test_label = (nn_test_label_src, nn_test_label_dst)
            nn_test_src, nn_test_dst = eval_one_epoch(num_cls, tgan, nn_test_rand_sampler, src=nn_test_src_l, dst = nn_test_dst_l, ts = nn_test_ts_l, label=nn_test_label, num_neighbors=NUM_NEIGHBORS)

            test_src["train_acc"], test_dst["train_acc"] = np.mean(trian_acc_src), np.mean(train_acc_dst)
            test_src["val_acc"], test_dst["val_acc"] = val_src["accuracy"], val_dst["accuracy"]
            
            test_acc = (test_src["accuracy"] + test_dst["accuracy"]) / 2
            test_ap = (test_src["precision"] + test_dst["precision"]) / 2
            test_recall = (test_src["recall"] + test_dst["recall"]) / 2
            print('Test statistics: {} all nodes -- acc: {:.4f}, prec: {:.4f}, recall: {:.4f}'.format(args.mode, test_acc, test_ap, test_recall))            

            test_src["train_acc"], test_dst["train_acc"] = np.mean(trian_acc_src), np.mean(train_acc_dst)
            test_src["val_acc"], test_dst["val_acc"] = val_src["accuracy"], val_dst["accuracy"]
            score_recoder.extend([test_src, test_dst])



    # testing phase use all information
    tgan.ngh_finder = full_ngh_finder
    test_old_src, test_old_dst = eval_one_epoch('test for old nodes', tgan, test_rand_sampler, test_src_l, 
    test_dst_l, test_ts_l, test_label_l, num_neighbors=NUM_NEIGHBORS)


    test_new_src, test_new_dst = eval_one_epoch('test for new nodes', tgan, nn_test_rand_sampler, nn_test_src_l, 
    nn_test_dst_l, nn_test_ts_l, nn_test_label, num_neighbors=NUM_NEIGHBORS)

    test_new_acc = (test_new_src["accuracy"] + test_new_dst["accuracy"]) / 2
    test_new_prec = (test_new_src["precision"] + test_new_dst["precision"]) / 2
    test_new_recall = (test_new_src["recall"] + test_new_dst["recall"]) / 2

    test_old_acc = (test_new_src["accuracy"] + test_new_dst["accuracy"]) / 2
    test_old_prec = (test_new_src["precision"] + test_new_dst["precision"]) / 2
    test_old_recall = (test_new_src["recall"] + test_new_dst["recall"]) / 2

    print('Test statistics: Old nodes -- acc: {:.4f}, recall: {:.4f}, precision: {:.4f}'.format(test_new_acc, test_new_recall, test_new_prec))
    print('Test statistics: New nodes -- acc: {:.4f}, recall: {:.4f}, precision: {:.4f}'.format(test_old_acc, test_old_recall, test_old_prec))

    temporaloader.update_event(sp)
    rpresent.score_record(temporal_score_=score_recoder, node_size=g_df.num_nodes, temporal_idx=sp, epoch_interval=epoch_tester)
    snapshot_list.append(score_recoder)


rpresent.record_end()
rscore.record_end()
rscore.fast_processing(snapshot_list)


