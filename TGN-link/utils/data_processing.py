import numpy as np
import random
import pandas as pd
from utils.my_dataloader import Temporal_Splitting, Dynamic_Dataloader, Temporal_Dataloader, data_load
import copy
import torch
from typing import Union, Optional, Any
from torch import Tensor
from numpy import ndarray

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
      
      self.target_node: Union[set|None] = None

  def set_up_features(self, node_feat, edge_feat):
    self.node_feat = node_feat
    self.edge_feat = edge_feat

  def inductive_back_propogation(self, node_idx: list, single_graph: bool, t_hash_table: Union[dict | None] = None):
    """
    Expected to clear the node index get establish the mask mainly for non-visible data node \n

    :Attention: meaning of node_idx is different(reversed) when single_graph is different!!!

    :param node_idx when single_graph is True -- is the node that uniquness to the given data object,
    :param node_idx when single_graph is False -- it represent node that should be removed in given node set!!!

    :return self.target_node -- whatever how node_idx and single_graph changed, it always present node to be Uniquness
    to the given Data object
    """
    batch_nodes = np.array(sorted(self.unique_nodes))
    if single_graph:
      # self.target_node_mask = np.isin(batch_nodes, sorted(node_idx))
      self.target_node = self.unique_nodes & set(node_idx)
    else:
      t_transfer_map = np.vectorize(t_hash_table.get)
      t1_transfer_map = np.vectorize(self.hash_table.get)

      seen_nodes = t_transfer_map(node_idx)
      test_seen_nodes = t1_transfer_map(batch_nodes)
      
      # test_node represent node idx that has NOT been seen in validation and train data
      test_node = set(test_seen_nodes) - set(seen_nodes)
      reverse_test_hashtable = {v:k for k, v in self.hash_table.items()}
      t1_back_transfer = np.vectorize(reverse_test_hashtable.get)
      self.target_node = t1_back_transfer(sorted(test_node))

      # self.target_node = self.unique_nodes - set(t_test_node)

    src_mask = np.isin(self.sources, sorted(self.target_node))
    dst_mask = np.isin(self.destinations, sorted(self.target_node))

    new_test_mask = src_mask*dst_mask
    if src_mask.sum()==0 or dst_mask.sum() == 0:
      new_test_mask = src_mask | dst_mask
      src_uniq, src_freq = np.unique(src_mask, return_counts=True)
      dst_uniq, dst_freq = np.unique(dst_mask, return_counts=True)
      print("New Old mode activated, considering one side of edge full of old, seen data")
      print("Source node uniquness {}, frequency {}, destination node uniquness {}, frequency".format(src_uniq, src_freq, dst_uniq, dst_freq))

    return self.inductive_test(new_test_mask)

  def call_for_inductive_nodes(self, val_data: 'Data', test_data: 'Data', single_graph: bool):
    validation_node: set = val_data.unique_nodes
    test_node: set = test_data.unique_nodes
    train_node = self.unique_nodes

    common_share = validation_node & test_node & train_node
    train_val_share = validation_node & train_node
    train_test_share = train_node & test_node
    val_test_share = validation_node & test_node

    expected_val = list(validation_node - (common_share | train_val_share))
    new_val = val_data.inductive_back_propogation(expected_val, single_graph=True)
    new_test= None

    if single_graph:
      expected_test = list(test_node - (train_test_share | common_share | val_test_share))
      new_test = test_data.inductive_back_propogation(expected_test, single_graph = single_graph)
    else:
      t_times_common_data = list(train_test_share | common_share | val_test_share)
      t_times_hash_table = val_data.hash_table
      new_test = test_data.inductive_back_propogation(t_times_common_data, single_graph, t_times_hash_table)

    assert len(set(expected_val) & train_node) == 0, "train_node data is exposed to validation set"
    if single_graph:
      assert len(set(expected_test) & train_node & set(expected_val)) == 0, "train node and val data has interacted with test data"

    return new_val, new_test

  # For edge inductive test and single graph
  def edge_mask(self, data: 'Data', test_element: set):
    test_element = sorted(test_element)
    src_mask = ~np.isin(data.sources, test_element)
    dst_mask = ~np.isin(data.destinations, test_element)
    return src_mask & dst_mask

  def cover_the_edges(self, val_data: 'Data', test_data: 'Data', single_graph: bool = True):
    """
    delete both edges and nodes appeared in train_data to make pure inductive val_data \n
    also, delete both edges and nodes appeared in train_data and val_data to make pure inductive test_data
    """
    valid_node = val_data.unique_nodes
    test_node = test_data.unique_nodes
    train_node = self.unique_nodes

    # common_share = valid_node & test_node & train_node
    train_val_share = valid_node & train_node
    train_test_share = train_node & test_node
    val_test_share = valid_node & test_node

    node_2be_removed_val = train_val_share
    node_2be_removed_test = val_test_share | train_test_share

    val_data.edge_propagate_back(self.edge_mask(val_data, node_2be_removed_val))
    test_data.edge_propagate_back(self.edge_mask(test_data, node_2be_removed_test))

    return

  def inductive_test(self, edge_mask: np.ndarray) -> 'Data':
    self.inductive_edge_mask = edge_mask
    src, dst = self.sources[edge_mask], self.destinations[edge_mask]
    tsp, edge_idx = self.timestamps[edge_mask], self.edge_idxs[edge_mask]
    y, hash_table = self.labels, self.hash_table

    return Data(sources=src, destinations=dst, timestamps=tsp, \
                edge_idxs=edge_idx, labels=y, hash_table=hash_table, node_feat=self.node_feat, edge_feat=self.edge_feat)


  def edge_propagate_back(self, edge_mask: np.ndarray):
    """
    keep the edge mask as permanent variable and modify edge \n
    maintain edges to inductive edge mask
    """
    self.inductive_edge_mask = edge_mask
    self.sources = self.sources[self.inductive_edge_mask]
    self.destinations = self.destinations[self.inductive_edge_mask]
    self.timestamps = self.timestamps[self.inductive_edge_mask]

  def sample(self,ratio):
    data_size=self.n_interactions
    sample_size=int(ratio*data_size)
    sample_inds=random.sample(range(data_size),sample_size)
    sample_inds=np.sort(sample_inds)
    sources=self.sources[sample_inds]
    destination=self.destinations[sample_inds]
    timestamps=self.timestamps[sample_inds]
    edge_idxs=self.edge_idxs[sample_inds]
    labels=self.labels[sample_inds]
    return Data(sources,destination,timestamps,edge_idxs,labels)

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
    
    """
    :param hash_table, should be a matching list, now here it is refreshed idx : origin idx,
    """
    hash_table: dict[int, int] = {idx: node for idx, node in zip(*hash_dataframe)}
    
    if np.any(graph.pos != None):
        node_pos, edge_pos = graph.pos
        def pos2numpy(pos: Optional[Tensor | ndarray] | Any) -> Optional[ndarray|Any]:
           return pos.numpy() if isinstance(pos, Tensor) else pos
        pos = (pos2numpy(node_pos), pos2numpy(edge_pos))
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

        # hash_dataframe = copy.deepcopy(items.my_n_id.node.loc[:, ["index", "node"]].values.T)
        """
        :feature -- TGN-Link will refresh node idx in each temporal graph, so here should use new node idx to match origin node
        Why? Becuase in Zebra-node it maintain a global tppr matrix and embedding output, thus in function
        :func - Temporal_Splitting(graph).temporal_spliting(*param) it wont call temporal_splitting(*param) to rebuild node idx
        So, in Zebra-node it maintain a original node : new node logic. 
        in TGN-link, it will re-build node idx for each temproal node, so it should be new node idx : original node idx!!
        """
        # hash_table: dict[int, int] = {node: idx for idx, node in zip(*hash_dataframe)} # should be idx : node !!
        hash_table = full_data.hash_table

        edge_train_feat = full_data.edge_feat[train_mask]
        edge_val_feat = full_data.edge_feat[val_mask]

        train_data = Data(full_data.sources[train_mask], full_data.destinations[train_mask], full_data.timestamps[train_mask],\
                        full_data.edge_idxs[train_mask], t_labels, hash_table = hash_table, node_feat=full_data.node_feat, edge_feat=edge_train_feat)
        
        val_data = Data(full_data.sources[val_mask], full_data.destinations[val_mask], full_data.timestamps[val_mask],\
                        full_data.edge_idxs[val_mask], t_labels, hash_table = hash_table, node_feat=full_data.node_feat, edge_feat=edge_val_feat)
        
        if single_graph or idxs == lenth-1:
            test_data = val_data
        else:
            test = graph_list[idxs+1]
            test_data = to_TPPR_Data(test)
            
        nn_val, nn_test = train_data.call_for_inductive_nodes(val_data, test_data, single_graph)
        
        node_num = items.num_nodes
        node_edges = items.num_edges

        TPPR_list.append([full_data, train_data, val_data, nn_val, test_data, nn_test, node_num, node_edges])


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
