import numpy as np
import random
import pandas as pd
from my_dataloader import Temporal_Splitting, Dynamic_Dataloader, Temporal_Dataloader, data_load
import copy
import torch

class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels, \
               hash_table: dict[int, int], node_feat: np.ndarray = None, edge_feat: np.ndarray = None):
      self.sources = sources
      self.destinations = destinations
      self.timestamps = timestamps
      self.edge_idxs = edge_idxs
      self.labels = labels
      self.n_interactions = len(sources)
      self.unique_nodes = set(sources) | set(destinations)
      self.n_unique_nodes = len(self.unique_nodes)
      self.tbatch = None
      self.n_batch = 0
      self.node_feat = node_feat
      self.edge_feat = edge_feat
      self.hash_table = hash_table

def quantile_(threshold: float, timestamps: torch.Tensor) -> tuple[torch.Tensor]:
  full_length = timestamps.shape[0]
  val_idx = int(threshold*full_length)

  if not isinstance(timestamps, torch.Tensor):
     timestamps = torch.from_numpy(timestamps)
  train_mask = torch.zeros_like(timestamps, dtype=bool)
  train_mask[:val_idx] = True

  val_mask = torch.zeros_like(timestamps, dtype=bool)
  val_mask[val_idx:] = True

  return train_mask, val_mask

def to_TPPR_Data(graph: Temporal_Dataloader) -> Data:
    nodes = graph.x
    edge_idx = np.arange(graph.edge_index.shape[1])
    timestamp = graph.edge_attr
    src, dest = graph.edge_index[0, :], graph.edge_index[1, :]
    labels = graph.y

    hash_dataframe = copy.deepcopy(graph.my_n_id.node.loc[:, ["index", "node"]].values.T)
    hash_table: dict[int, int] = {node: idx for idx, node in zip(*hash_dataframe)}
    
    if np.any(graph.edge_attr != None):
        edge_attr = graph.edge_attr
    if np.any(graph.pos != None):
        pos = graph.pos
        pos = pos.numpy() if isinstance(pos, torch.Tensor) else pos
    else:
        pos = graph.x

    edge_feat, node_feat = graph.edge_pos, graph.node_pos
    TPPR_data = Data(sources= src, destinations=dest, timestamps=timestamp, edge_idxs = edge_idx, \
                     labels=labels, hash_table=hash_table, node_feat=node_feat, edge_feat=edge_feat)

    return TPPR_data

def get_data_TGAT(dataset_name, snapshot: int, views: int) -> tuple[list[Data|int], int, np.ndarray, int]:
    r"""
    this function is used to convert the node features to the correct format
    e.g. sample node dataset is in the format of [node_id, edge_idx, timestamp, features] with correspoding
    shape [(n, ), (m,2), (m,), (m,d)]. be cautious on transformation method
    """
    graph, idx_list = data_load(dataset_name, emb_size=64)
    if snapshot<=3: 
        graph.edge_attr = np.arange(graph.edge_index.shape[1])
        graph_list = [graph]
    else:
        graph_list = Temporal_Splitting(graph).temporal_splitting(time_mode='view', snapshot=snapshot, views=views)
    
    graph_num_node, graph_feat, edge_number = max(graph.x)+1, copy.deepcopy(graph.pos), graph.edge_index.shape[1]

    TPPR_list: list[list[Data]] = []
    lenth = len(graph_list)
    single_graph = False
    if lenth < 2: 
        lenth = 2
        single_graph = True

    for idxs in range(0, lenth-1):
        # covert Temproal_graph object to Data object
        items = graph_list[idxs]
        items.edge_attr = items.edge_attr # .numpy()
        # items.pos = items.pos.numpy()
        items.y = np.array(items.y)

        t_labels = items.y
        full_data = to_TPPR_Data(items)
        timestamp = full_data.timestamps
        train_mask, val_mask = quantile_(threshold=0.85, timestamps=timestamp)

        hash_dataframe = copy.deepcopy(items.my_n_id.node.loc[:, ["index", "node"]].values.T)
        hash_table: dict[int, int] = {node: idx for idx, node in zip(*hash_dataframe)}

        train_data = Data(full_data.sources[train_mask], full_data.destinations[train_mask], full_data.timestamps[train_mask],\
                        full_data.edge_idxs[train_mask], t_labels, hash_table = hash_table, node_feat=full_data.node_feat)
        
        val_data = Data(full_data.sources[val_mask], full_data.destinations[val_mask], full_data.timestamps[val_mask],\
                        full_data.edge_idxs[val_mask], t_labels, hash_table = hash_table, node_feat=full_data.node_feat)
        
        if single_graph or idxs == lenth-1:
            test_data = val_data
        else:
            test = graph_list[idxs+1]
            test_data = to_TPPR_Data(test)
        node_num = items.num_nodes
        node_edges = items.num_edges

        TPPR_list.append([full_data, train_data, val_data, test_data, node_num, node_edges])


    return TPPR_list, graph_num_node, graph_feat, edge_number

def get_data_node_classification(dataset_name, use_validation=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  random.seed(2020)

  train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
  test_mask = timestamps > test_time
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name)) 
    
  if randomize_features:
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(2020)

  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)

  # Compute nodes which appear at test time
  test_node_set = set(sources[timestamps > val_time]).union(
    set(destinations[timestamps > val_time]))
  # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
  # their edges from training
  new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

  # Mask saying for each source and destination whether they are new test nodes
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  # Mask which is true for edges with both destination and source not being new test nodes (because
  # we want to remove all edges involving any new test node)
  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  # define the new nodes sets for testing inductiveness of the model
  train_node_set = set(train_data.sources).union(train_data.destinations)
  assert len(train_node_set & new_test_node_set) == 0
  new_node_set = node_set - train_node_set

  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time

  if different_new_nodes_between_val_and_test:
    n_new_nodes = len(new_test_node_set) // 2
    val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
    test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

    edge_contains_new_val_node_mask = np.array(
      [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
    edge_contains_new_test_node_mask = np.array(
      [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


  else:
    edge_contains_new_node_mask = np.array(
      [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  # validation and test with all edges
  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  # validation and test with edges that at least has one new node (not in training set)
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])

  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])

  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
  print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
  print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
    len(new_test_node_set)))

  return node_features, edge_features, full_data, train_data, val_data, test_data, \
         new_node_val_data, new_node_test_data


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
