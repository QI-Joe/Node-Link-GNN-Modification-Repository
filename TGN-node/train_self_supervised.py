import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
from utils.time_evaluation import TimeRecord

from evaluation.evaluation import eval_edge_prediction, eval_node_classification
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics, get_data_TGAT

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='dblp')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=1, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--snapshot', type=int, default=10, help='how many temporal graphs')
parser.add_argument('--view', type=int, default=5, help="acutally running graphs")
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')


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
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
SNAPSHOT = args.snapshot
VIEW = args.view
epoch_tester = 1


### Extract data for training, validation and testing
temporaloader, full_graph_nodes, full_graph_feat, full_edge_number = get_data_TGAT(DATA,snapshot=SNAPSHOT, views=SNAPSHOT-2)

num_classes = temporaloader[-1][0].labels.max()+1
snapshot_list = list()
rscore, rpresent = TimeRecord(model_name="TGN"), TimeRecord(model_name="TGN")
rscore.get_dataset(DATA)
rpresent.get_dataset(DATA)
rscore.set_up_logger(name="time_logger")
rpresent.set_up_logger()
rpresent.record_start()

def compute_value(x, threshold):
  lists = list()
  for i, v in enumerate(x):
    if (v>threshold).any():
      lists.append(x[i])
  return lists

for i in range(VIEW):
  full_data, train_data, val_data, test_data, n_nodes, n_edges = temporaloader[i]
  # Initialize training neighbor finder to retrieve temporal graph
  train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

  # Initialize validation and test neighbor finder to retrieve temporal graph
  full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

  test_ngh_finder = get_neighbor_finder(test_data, args.uniform)

  # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
  # across different runs
  # NB: in the inductive setting, negatives are sampled only amongst other new nodes
  train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
  val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
  test_rand_sampler = RandEdgeSampler(test_data.sources, test_data.destinations, seed=2)

  # Set device
  device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
  device = torch.device(device_string)
  # device = torch.device("cpu")
  
  # Compute time statistics
  mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)


  node_features, edge_features = full_data.node_feat, full_data.edge_feat

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)

  src_label, dst_label = train_data.labels[train_data.sources], train_data.labels[train_data.destinations]
  print(f"\n\nthe max node idx is {max(full_data.sources.max(), full_data.destinations.max())} at snapshot {i}")
  print(f"max node idx of val data {val_data.node_feat.shape} and max node idx of test data {test_data.node_feat.shape}")
  print(f"checking the intern node memory {tgn.embedding_module.node_features.shape}")
  if tgn.embedding_module.node_features_backup != None:
    print(f"checking the intern node back memory {tgn.embedding_module.node_features_backup.shape}")
  print("\n\n")

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  print('num of training instances: {}'.format(num_instance))
  # print('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  score_recorder = list()
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    print('start {} epoch'.format(epoch))
    for k in range(0, num_batch, args.backprop_every):
      loss = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        size = len(sources_batch)
        _, negatives_batch = train_rand_sampler.sample(size)

        # with torch.no_grad():
        #   pos_label = torch.ones(size, dtype=torch.float, device=device)
        #   neg_label = torch.zeros(size, dtype=torch.float, device=device)

        tgn = tgn.train()
        src_emb, dst_emb, _ = tgn.compute_temporal_embeddings(source_nodes=sources_batch, destination_nodes=destinations_batch,\
                                                           negative_nodes=destinations_batch, edge_times=timestamps_batch, \
                                                            edge_idxs=edge_idxs_batch, n_neighbors=NUM_NEIGHBORS)
        # pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
        #                                                     timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
        src_batch_label, dst_batch_label = torch.from_numpy(src_label[start_idx: end_idx]).to(device), torch.from_numpy(dst_label[start_idx: end_idx]).to(device)
        loss += criterion(src_emb, src_batch_label) + criterion(dst_emb, dst_batch_label)

      loss /= args.backprop_every

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      if USE_MEMORY:
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    # Validation uses the full graph
    if (epoch+1) % epoch_tester !=0:
      continue

    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      train_memory_backup = tgn.memory.backup_memory()


    max_node = max(val_data.sources.max(), val_data.destinations.max())
    max_node_full = max(full_data.sources.max(), full_data.destinations.max())
    internal_tolearance = tgn.embedding_module.node_features.shape[0]
    assert max_node<full_data.node_feat.shape[0] or max_node<val_data.node_feat.shape[0], f"max node idx {max_node} larger than node features {full_data.node_feat.shape[0]}"
    assert  max_node_full<full_data.node_feat.shape[0] or max_node_full<internal_tolearance, f"max current full node snapshot idx {max_node_full} larger than given node features {internal_tolearance, full_data.node_feat.shape[0]}"
    assert full_data.edge_idxs.max() >= val_data.edge_idxs.max(), f"val data edge idx larger than full data, check it out"

    if i >= 3: 
      print("wearehere")
    try: 
      val_metrics = eval_node_classification(tgn=tgn,
                                           num_cls=num_classes,
                                          batch_size=200,
                                          data=val_data,
                                          n_neighbors=NUM_NEIGHBORS)
    except RuntimeError as error:
      print("\nThe max node idx is", max_node, "Current model node embedding is", tgn.embedding_module.node_features.shape)
      print("\nCurrent full data max node idx is", full_data.node_feat.shape)
      print(f"\n\nthe max node idx is {max_node_full} at snapshot {i}")
      print(f"max node idx of val data {val_data.node_feat.shape} and max node idx of test data {test_data.node_feat.shape}")
      print(f"checking the intern node memory {tgn.embedding_module.node_features.shape}")
    train_losses.append(np.mean(m_loss))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    print('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    print('Epoch mean loss: {:.4f}'.format(np.mean(m_loss)))
    print('val acc: {:.4f}, val precision: {:.4f}'.format(val_metrics["accuracy"], val_metrics["precision"]))

    # Early stopping
    if early_stopper.early_stop_check(val_metrics["precision"]):
      print('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      print(f'Loading the best model at epoch {early_stopper.best_epoch}')
      # best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      # tgn.load_state_dict(torch.load(best_model_path))
      print(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      break
    # tgn.update4test(test_ngh_finder, test_data.node_feat, test_data.edge_feat)

    # test_metrics = eval_node_classification(tgn=tgn,
    #                                            num_cls=num_classes,
    #                                             batch_size=200,
    #                                             data=test_data,
    #                                             n_neighbors=NUM_NEIGHBORS)

    # tgn.restore_test_emb()
    # tgn.embedding_module.backup_release()
    # test_metrics["val_acc"] = val_metrics["accuracy"]
    # score_recorder.append(test_metrics)

    # print('Test statistics: {} all nodes -- acc: {:.4f}, prec: {:.4f}, recall: {:.4f}'.format("TGN", \
    #                 test_metrics["accuracy"], test_metrics["precision"], test_metrics["recall"]))


  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen
  # nodes
  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()

  ### Test
  tgn.update4test(test_ngh_finder, test_data.node_feat, test_data.edge_feat)
  test_metrics = eval_node_classification(tgn=tgn,
                                               num_cls=num_classes,
                                                batch_size=200,
                                                data=test_data,
                                                n_neighbors=NUM_NEIGHBORS)
  test_metrics["val_acc"] = val_metrics["accuracy"]
  score_recorder.append(test_metrics)

  rpresent.score_record(temporal_score_=score_recorder, node_size=full_data.n_unique_nodes, temporal_idx=i, epoch_interval=epoch_tester)

rpresent.record_end()
rscore.record_end()
rscore.fast_processing(snapshot_list)



