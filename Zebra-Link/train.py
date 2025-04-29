import math
import logging
import time
import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from evaluation.evaluation import eval_edge_prediction, eval_node_classification, LogRegression
from model.tgn_model import TGN
from utils.util import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data_TPPR, test_tranucate
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaTypeSafetyWarning
from utils.my_dataloader import to_cuda, Temporal_Splitting, Temporal_Dataloader, data_load
from itertools import chain
from evaluation.time_evaluation import TimeRecord

import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)

very_start = time.time()
def str2bool(order: str)->bool:
  if order in ["True", "1"]:
    return True
  return False

def fast_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return hours, minutes, seconds

parser = argparse.ArgumentParser('Self-supervised training with diffusion models')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',default='mooc')
parser.add_argument('--bs', type=int, default=1000, help='Batch_size')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=7, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--snapshot', type=int, default=7, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.3, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--use_memory', default=True, type=bool, help='Whether to augment the model with a node memory')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',help='Whether to use the embedding of the source node as part of the message')

parser.add_argument('--message_function', type=str, default="identity", choices=["mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"], help='Type of memory updater')
parser.add_argument('--embedding_module', type=str, default="diffusion", help='Type of embedding module')

parser.add_argument('--enable_random', action='store_true',help='use random seeds')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--save_best',action='store_true', help='store the largest model')
parser.add_argument('--tppr_strategy', type=str, help='[streaming|pruning]', default='streaming')
parser.add_argument('--topk', type=int, default=20, help='keep the topk neighbor nodes')
parser.add_argument('--alpha_list', type=float, nargs='+', default=[0.1, 0.1], help='ensemble idea, list of alphas')
parser.add_argument('--beta_list', type=float, nargs='+', default=[0.05, 0.95], help='ensemble idea, list of betas')
parser.add_argument('--dynamic', type=str2bool, default=False)


parser.add_argument('--ignore_edge_feats', action='store_true')
parser.add_argument('--ignore_node_feats', action='store_true')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--memory_dim', type=int, default=100, help='Dimensions of the memory for each user')

args = parser.parse_args()
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
USE_MEMORY = True
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
MEMORY_DIM = args.memory_dim
BATCH_SIZE = args.bs
dynamic: bool = args.dynamic
epoch_tester: int = 1
SNAPSHOT = args.snapshot

round_list, graph_num, graph_feature, edge_num = get_data_TPPR(DATA, snapshot=SNAPSHOT, views=SNAPSHOT-2)
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

training_strategy = "node"
NODE_DIM = round_list[0][0].node_feat.shape[1]
MEMORY_DIM = NODE_DIM
test_record = []
global_edge_feats = graph_feature[1]
args.n_nodes = graph_feature[0].shape[0] + 1
args.n_edges = graph_feature[1].shape[0] + 1

tgn = TGN(edge_features=global_edge_feats, device=device, # neighbor_finder=train_ngh_finder, node_features=node_feats, 
            n_layers=NUM_LAYER,n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            node_dimension = NODE_DIM, time_dimension = TIME_DIM, memory_dimension=MEMORY_DIM,
            embedding_module_type=args.embedding_module, 
            message_function=args.message_function, 
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            args=args)
tgn.update4node_num(args.n_nodes, device)

snapshot_list = list()
rscore, rpresent = TimeRecord(model_name="TGAT"), TimeRecord(model_name="TGAT")
rscore.get_dataset(DATA)
rpresent.get_dataset(DATA)
rscore.set_up_logger(name="time_logger")
rpresent.set_up_logger()
rpresent.record_start()
logger = rpresent.score_log_handler

for i in range(len(round_list)):
  temporal_very_start = time.time()
  full_data, train_data, val_data, nn_val, test_data, nn_test, n_nodes, n_edges = round_list[i]
  # args.n_nodes = n_nodes +1
  args.n_edges = n_edges +1

  edge_feats = full_data.edge_feat
  node_feats = full_data.node_feat
  node_feat_dims = full_data.node_feat.shape[1]

  if args.ignore_node_feats:
    print('>>> Ignore node features')
    node_feats = None
    node_feat_dims = 0

  if edge_feats is None or args.ignore_edge_feats: 
    print('>>> Ignore edge features')
    edge_feats = np.zeros((args.n_edges, 1))
    edge_feat_dims = 1

  train_ngh_finder = get_neighbor_finder(train_data)
  full_ngh_finder = get_neighbor_finder(full_data)
  test_ngh_finder = get_neighbor_finder(test_data)

  train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
  val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
  test_rand_sampler = RandEdgeSampler(test_data.sources, test_data.destinations, seed=2)

  nn_val_rand_sampler = RandEdgeSampler(nn_val.sources, nn_val.destinations, seed=1)
  nn_test_rand_sampler = RandEdgeSampler(nn_test.sources, nn_test.destinations, seed=3)

  # tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_feats, edge_features=edge_feats, device=device,
  #           n_layers=NUM_LAYER,n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
  #           node_dimension = NODE_DIM, time_dimension = TIME_DIM, memory_dimension=MEMORY_DIM,
  #           embedding_module_type=args.embedding_module, 
  #           message_function=args.message_function, 
  #           aggregator_type=args.aggregator,
  #           memory_updater_type=args.memory_updater,
  #           n_neighbors=NUM_NEIGHBORS,
  #           use_destination_embedding_in_message=args.use_destination_embedding_in_message,
  #           use_source_embedding_in_message=args.use_source_embedding_in_message,
  #           args=args)
  tgn.update4temporal_graph(train_ngh_finder, edge_feats)
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)
  early_stopper = EarlyStopMonitor(max_round=args.patience)
  t_total_epoch_train=0
  t_total_epoch_val=0
  t_total_epoch_test=0
  t_total_tppr=0
  stop_epoch=-1

  train_tppr_time=[]
  train_tppr_filled = False
  test_tppr_filled=False
  train_tppr_backup, val_tppr_backup = None, None
  score_recorder = list()

  for epoch in range(NUM_EPOCH):
    print(f"the start of epoch {epoch}")
    logger.info(f"the start of epoch {epoch}")
    t_epoch_train_start = time.time()
    tgn.reset_timer()
    train_data = train_data
    val_data = val_data
    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance/BATCH_SIZE)

    train_ap=[]
    train_acc=[]
    train_auc=[]
    train_loss=[]

    tgn.memory.__init_memory__()
    if args.tppr_strategy=='streaming':
      tgn.embedding_module.reset_tppr()
    tgn.set_neighbor_finder(train_ngh_finder)


    # model training
    for batch_idx in range(0, num_batch):
      start_idx = batch_idx * BATCH_SIZE
      end_idx = min(num_instance, start_idx + BATCH_SIZE)
      sample_inds=np.array(list(range(start_idx,end_idx)))
      sources_batch, destinations_batch = train_data.sources[sample_inds],train_data.destinations[sample_inds]
      edge_idxs_batch = train_data.edge_idxs[sample_inds]
      timestamps_batch = train_data.timestamps[sample_inds]
      size = len(sources_batch)
      _, negatives_batch = train_rand_sampler.sample(size)

      with torch.no_grad():
        pos_label = torch.ones(size, dtype=torch.float, device=device)
        neg_label = torch.zeros(size, dtype=torch.float, device=device)

      tgn = tgn.train()
      optimizer.zero_grad()

      pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS, train=True)
      loss = criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())
      with torch.no_grad():
        pos_prob=pos_prob.cpu().numpy() 
        neg_prob=neg_prob.cpu().numpy() 
        pred_score = np.concatenate([pos_prob, neg_prob]) 
        true_label = np.concatenate([np.ones(size), np.zeros(size)])  
        true_binary_label= np.zeros(size)
        pred_binary_label = np.argmax(np.hstack([pos_prob,neg_prob]),axis=1) 
        train_ap.append(average_precision_score(true_label, pred_score))
        train_auc.append(roc_auc_score(true_label, pred_score))
        train_acc.append(accuracy_score(true_binary_label, pred_binary_label))


    epoch_tppr_time = tgn.embedding_module.t_tppr
    train_tppr_time.append(epoch_tppr_time)

    epoch_train_time = time.time() - t_epoch_train_start
    t_total_epoch_train+=epoch_train_time
    train_ap=np.mean(train_ap)
    train_auc=np.mean(train_auc)
    train_acc=np.mean(train_acc)
    train_loss=np.mean(train_loss)

    if (epoch+1) % epoch_tester != 0:
      continue

    # change the tppr finder to validation and test
    if args.tppr_strategy=='streaming':
      tgn.embedding_module.reset_tppr()
      tgn.embedding_module.train_fill_tppr(train_data.sources, train_data.destinations, train_data.timestamps, train_data.edge_idxs, train_tppr_filled)
      train_tppr_filled = True
    tgn.set_neighbor_finder(full_ngh_finder)

    ########################  Model Validation on the Val Dataset #######################
    t_epoch_val_start=time.time()
    print("start **Validation** transductive/inductive testing")
    ### transductive val
    train_memory_backup = tgn.memory.backup_memory()
    if args.tppr_strategy=='streaming':
      train_tppr_backup = tgn.embedding_module.backup_tppr()

    val_ap, val_auc, val_acc = eval_edge_prediction(model=tgn, negative_edge_sampler=val_rand_sampler, data=val_data, n_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE)
    
    
    tgn.memory.restore_memory(train_memory_backup)
    if args.tppr_strategy=='streaming':
      tgn.embedding_module.restore_tppr(train_tppr_backup)

    ### inductive val
    nn_val_ap, nn_val_auc, nn_val_acc = eval_edge_prediction(model=tgn, negative_edge_sampler=nn_val_rand_sampler, data=nn_val, n_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE)
    

    epoch_val_time = time.time() - t_epoch_val_start
    t_total_epoch_val += epoch_val_time
    epoch_id = epoch+1
    print('epoch: {}, tppr: {}, train: {}, val: {}'.format(epoch_id, epoch_tppr_time, epoch_train_time, epoch_val_time))
    print('train auc: {}, train ap: {}, train acc: {}, train loss: {}'.format(train_auc, train_ap, train_acc, train_loss))
    print('val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    print('val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
    print('val acc: {}, new node val acc: {}'.format(val_acc, nn_val_acc))
    
    logger.info('epoch: {}, tppr: {}, train: {}, val: {}'.format(epoch_id, epoch_tppr_time, epoch_train_time, epoch_val_time))
    logger.info('train auc: {}, train ap: {}, train acc: {}, train loss: {}'.format(train_auc, train_ap, train_acc, train_loss))
    logger.info('val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info('val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
    logger.info('val acc: {}, new node val acc: {}'.format(val_acc, nn_val_acc))


    last_best_epoch=early_stopper.best_epoch
    if early_stopper.early_stop_check(val_ap):
      stop_epoch=epoch_id
      tgn.eval()
      break

    ######################  Evaludate Model on the Test Dataset #######################
    t_test_start=time.time()

    print("start **Test** transductive/inductive testing")

    test_n_nodes = test_data.node_feat.shape[0] + 1
    tgn.update4test(test_ngh_finder, test_n_nodes, test_data.edge_feat)
    
    if args.tppr_strategy=='streaming':
      """
      TODO: Truncate the test_data based on length of full_data to (seen data / unseen T-T+1 moment), and provide cache memory or not
      TODO-2: cache the 'train set' of test_data and memory
      t_t1_interval_Data: trancuate the test_data based on length of train_data to avoid depulication of train-test data
      """
      tgn.embedding_module.reset_tppr() # in global view mode, reset_tppr here is not necessary
      (tsrc, tdst, ttsp, teid), t_t1_interval_Data = test_tranucate(train_data.sources.shape[0], test_data)
      tgn.embedding_module.test_fill_tppr(tsrc, tdst, ttsp, teid, test_tppr_filled)
      test_tppr_filled = True
    
    test_train_memory_backup = tgn.memory.backup_memory()
    if args.tppr_strategy=='streaming':
      test_train_tppr_backup = tgn.embedding_module.backup_tppr()
    
    test_ap, test_auc, test_acc = eval_edge_prediction(model=tgn, negative_edge_sampler=test_rand_sampler, \
                                data=t_t1_interval_Data, n_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE)

    tgn.memory.restore_memory(test_train_memory_backup)
    if args.tppr_strategy=='streaming':
      tgn.embedding_module.restore_tppr(test_train_tppr_backup)

    ### inductive test
    nn_test_ap, nn_test_auc, nn_test_acc = eval_edge_prediction(model=tgn, negative_edge_sampler= nn_test_rand_sampler, data=nn_test, n_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE)
    t_test=time.time()-t_test_start

    train_tppr_time=train_tppr_time[1:]
    NUM_EPOCH=stop_epoch if stop_epoch!=-1 else NUM_EPOCH
    print(f'### num_epoch {NUM_EPOCH}, epoch_train {t_total_epoch_train/NUM_EPOCH}, epoch_val {t_total_epoch_val/NUM_EPOCH}, epoch_test {t_test}, train_tppr {np.mean(train_tppr_time)}')
    
    print('Test statistics: Old nodes -- auc: {}, ap: {}, acc: {}'.format(test_auc, test_ap, test_acc))
    print('Test statistics: New nodes -- auc: {}, ap: {}, acc: {}'.format(nn_test_auc, nn_test_ap, nn_test_acc))
    
    logger.info('Test statistics: Old nodes -- auc: {}, ap: {}, acc: {}'.format(test_auc, test_ap, test_acc))
    logger.info('Test statistics: New nodes -- auc: {}, ap: {}, acc: {}'.format(nn_test_auc, nn_test_ap, nn_test_acc))
    
    tgn.restore_test_emb()
    tgn.embedding_module.backup_release()
    
    validation_dict = {
      "val_acc": val_acc,
      "val_ap": val_ap,
      "val_auc": val_auc,
      "test_acc": test_acc,
      "test_ap": test_ap,
      "test_auc": test_auc,
      "test_new_old_acc": nn_test_acc if 'nn_test_acc' in locals() else None,
      "test_new_old_auc": nn_test_auc if 'nn_test_auc' in locals() else None,
      "test_new_old_ap": nn_test_ap
    }
    
    score_recorder.append(validation_dict)
  
  
  rpresent.score_record(temporal_score_=score_recorder, node_size=full_data.node_feat.shape[0], temporal_idx = i, epoch_interval=epoch_tester, mode='i')
  temporal_very_end = time.time() - temporal_very_start
  th, tm, ts = fast_to_hms(temporal_very_end)
  print(f"Running time for {th} hours, {tm} minutes, {ts} seconds")
  logger.info(f"Running time for {th} hours, {tm} minutes, {ts} seconds")
  
  snapshot_list.append(score_recorder)
    
rpresent.record_end()
rscore.record_end()
rscore.fast_processing('i', snapshot_list)
    
very_end = time.time() - very_start
h,m,s = fast_to_hms(very_end)
print(f"Running time for {h} hours, {m} minutes, {s} seconds")
logger.info(f"Running time for {h} hours, {m} minutes, {s} seconds")