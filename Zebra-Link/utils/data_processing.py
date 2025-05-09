from math import radians
import numpy as np
import random
import pandas as pd
import os
from utils.my_dataloader import data_load, Temporal_Splitting, Temporal_Dataloader
import torch
import copy
from typing import Union, Optional, Any
from numpy import ndarray
from torch import Tensor

class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels, \
               hash_table: dict[int, int], full_feat: Optional[tuple | ndarray] = None):
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
        self.node_feat, self.edge_feat = full_feat
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

  def call_for_inductive_nodes(self, val_data: 'Data', test_data: 'Data', single_graph: bool, test_val_equal: bool=False):
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
    if test_val_equal:
      return new_val, copy.deepcopy(new_val)

    if single_graph:
      expected_test = list(test_node - (train_test_share | common_share | val_test_share))
      new_test = test_data.inductive_back_propogation(expected_test, single_graph = single_graph)
    else:
      t_times_common_data = list(validation_node | train_node)
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
                edge_idxs=edge_idx, labels=y, hash_table=hash_table, full_feat=(self.node_feat, self.edge_feat))


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

def to_TPPR_Data(graph: Temporal_Dataloader) -> Data:
    edge_idx = np.arange(graph.edge_index.shape[1])
    timestamp = graph.edge_attr
    src, dest = graph.edge_index[0, :], graph.edge_index[1, :]
    labels = graph.y

    hash_dataframe = copy.deepcopy(graph.my_n_id.node.loc[:, ["index", "node"]].values.T)
    hash_table: dict[int, int] = {idx: node for idx, node in zip(*hash_dataframe)}
    
    if np.any(graph.pos != None):
        node_pos, edge_pos = graph.pos
        def pos2numpy(pos: Optional[Tensor | ndarray] | Any) -> Optional[ndarray|Any]:
           return pos.numpy() if isinstance(pos, Tensor) else pos
        pos = (pos2numpy(node_pos), pos2numpy(edge_pos))
    else:
        pos = graph.x

    TPPR_data = Data(sources= src, destinations=dest, timestamps=timestamp, edge_idxs = edge_idx, labels=labels, hash_table=hash_table, full_feat=pos)

    return TPPR_data

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

def quantile_static(val: float, test: float, timestamps: torch.Tensor) -> tuple[torch.Tensor]:
  full_length = timestamps.shape[0]
  val_idx = int(val*full_length)
  test_idx = int(test*full_length)

  if not isinstance(timestamps, torch.Tensor):
     timestamps = torch.from_numpy(timestamps)
  train_mask = torch.zeros_like(timestamps, dtype=bool)
  train_mask[:val_idx] = True

  val_mask = torch.zeros_like(timestamps, dtype=bool)
  val_mask[val_idx:test_idx] = True

  test_mask = torch.zeros_like(timestamps, dtype=bool)
  test_mask[test_idx:] = True

  return train_mask, val_mask, test_mask

def get_data_TPPR(dataset_name, snapshot: int, views: int):
    r"""
    this function is used to convert the node features to the correct format\n
    e.g. sample node dataset is in the format of [node_id, edge_idx, timestamp, features] with correspoding\n
    shape [(n, ), (m,2), (m,), (m,d)]. be cautious on transformation method\n
    
    2025.4.5 TPPR and data_load method will not support TGB-Series data anymore
    """
    graph, idx_list = data_load(dataset_name, emb_size=64)
    if snapshot<=3: 
        graph.edge_attr = np.arange(graph.edge_index.shape[1])
        graph_list = [Temporal_Dataloader(nodes=graph.x, edge_index=graph.edge_index, \
                                          edge_attr=graph.edge_attr, y=graph.y, pos=graph.pos)]
    else:
        graph_list = Temporal_Splitting(graph).temporal_splitting(time_mode = 'view', snapshot=snapshot, views = views)
    graph_num_node, graph_feat, edge_number = max(graph.x), copy.deepcopy(graph.pos), graph.edge_index.shape[1]

    TPPR_list: list[list[Data]] = []
    lenth = len(graph_list)
    single_graph = False
    if lenth < 2: 
        single_graph = True
        
    test_val_equal = False
    for idxs in range(0, lenth):
        # covert Temproal_graph object to Data object
        items = graph_list[idxs]
        items.edge_attr = items.edge_attr # .numpy()
        # items.pos = items.pos.numpy()
        items.y = np.array(items.y)

        t_labels = items.y
        full_data = to_TPPR_Data(items)
        timestamp = full_data.timestamps
        train_mask, val_mask = quantile_(threshold=0.80, timestamps=timestamp)

        if single_graph:
          train_mask, val_mask, test_mask = quantile_static(val=0.1, test=0.2,timestamps=timestamp)

        # hash_dataframe = copy.deepcopy(items.my_n_id.node.loc[:, ["index", "node"]].values.T)
        """
        :feature -- Zebra-Link will refresh node idx in each temporal graph, so here should use new node idx to match origin node
        Why? Becuase in Zebra-node it maintain a global tppr matrix and embedding output, thus in function
        :func - Temporal_Splitting(graph).temporal_spliting(*param) it wont call temporal_splitting(*param) to rebuild node idx
        So, in Zebra-node it maintain a 'Dict[original node : new node]' logic. 
        in Zebra-link, it will re-build node idx for each temproal node, so it should be Dict[new node idx : original node] idx!!
        """
        # hash_table: dict[int, int] = {idx: node for idx, node in zip(*hash_dataframe)}

        full_feat = (full_data.node_feat, full_data.edge_feat)
        train_data = Data(full_data.sources[train_mask], full_data.destinations[train_mask], full_data.timestamps[train_mask],\
                        full_data.edge_idxs[train_mask], t_labels, hash_table = full_data.hash_table, full_feat=full_feat)
        
        val_data = Data(full_data.sources[val_mask], full_data.destinations[val_mask], full_data.timestamps[val_mask],\
                        full_data.edge_idxs[val_mask], t_labels, hash_table = full_data.hash_table, full_feat=full_feat)
        
        if single_graph:
            test_data = Data(full_data.sources[test_mask], full_data.destinations[test_mask], full_data.timestamps[test_mask],\
                        full_data.edge_idxs[test_mask], t_labels, hash_table = full_data.hash_table, full_feat=full_feat)
        elif idxs == lenth-1:
            test_data = copy.deepcopy(val_data)
            test_val_equal=True
        else:
            test = graph_list[idxs+1]
            test_data = to_TPPR_Data(test)
        print(idxs, test_val_equal, full_data.sources.shape, "\n\n")
        
        nn_val, nn_test = train_data.call_for_inductive_nodes(val_data, test_data, True, test_val_equal)

        node_num = items.num_nodes
        node_edges = items.num_edges

        TPPR_list.append([full_data, train_data, val_data, nn_val, test_data, nn_test, node_num, node_edges])

    return TPPR_list, graph_num_node, graph_feat, edge_number


def batch_processor(data_label: dict[dict[int, np.ndarray]], data: Data)->list[tuple[np.ndarray]]:
  time_stamp = data.timestamps
  unique_ts = np.unique(time_stamp)
  idx_list = np.arange(time_stamp.shape[0])
  time_keys = list(data_label.keys())

  last_idx = 0
  batch_list: list[tuple] = list()
  for ts in unique_ts:
    time_mask = time_stamp==ts
    time_mask[:last_idx] = False
    if ts not in time_keys:
      # print(ts)
      if time_mask.sum() != 0:
        last_idx = idx_list[time_mask][-1]
      continue

    temp_dict = data_label[ts]
    keys = np.array(list(temp_dict.keys()))
    values = np.array(list(temp_dict.values()))

    unique_time_nodes = set(data.sources[time_mask]) | set(data.destinations[time_mask])
    # if len(unique_time_nodes) != len(keys):
    #   print(f"At time {ts}; Under the same timetable the unique node {len(unique_time_nodes)} size and {len(set(keys))} isnt matched")
    #   print(f"The different unique nodes are {unique_time_nodes - set(keys)} \n")

    sort_idx = np.argsort(keys)
    sort_key = keys[sort_idx]
    values = values[sort_idx, :]
    last_idx = idx_list[time_mask][-1]

    backprop_mask = np.isin(np.array(sorted(unique_time_nodes)), sort_key)

    batch_list.append((backprop_mask, values, time_mask))
  return batch_list
  

def TGB_load(train, val, test, node_feat):

  def single_transform(_data):
    _edge_idx = np.arange(_data.src.shape[0])
    transform_data = Data(_data.src.numpy(), _data.dst.numpy(), _data.t.numpy(), \
                      _edge_idx, _data.y.numpy(), hash_table=None)
    transform_data.set_up_features(node_feat, _data.msg.numpy())
    return transform_data
  
  train_data = single_transform(train)
  val_data = single_transform(val)
  test_data = single_transform(test)

  return train_data, val_data, test_data

def batch_processor(data_label: dict[dict[int, np.ndarray]], data: Data)->list[tuple[np.ndarray]]:
  time_stamp = data.timestamps
  unique_ts = np.unique(time_stamp)
  idx_list = np.arange(time_stamp.shape[0])
  time_keys = list(data_label.keys())

  last_idx = 0
  batch_list: list[tuple] = list()
  for ts in unique_ts:
    time_mask = time_stamp==ts
    time_mask[:last_idx] = False
    if ts not in time_keys:
      print(ts)
      last_idx = idx_list[time_mask][-1]
      continue

    temp_dict = data_label[ts]
    keys = np.array(list(temp_dict.keys()))
    values = np.array(list(temp_dict.values()))

    unique_time_nodes = set(data.sources[time_mask]) | set(data.destinations[time_mask])
    if len(unique_time_nodes) != len(keys):
      print(f"At time {ts}; Under the same timetable the unique node {len(unique_time_nodes)} size and {len(set(keys))} isnt matched")
      print(f"The different unique nodes are {unique_time_nodes - set(keys)} \n")

    sort_idx = np.argsort(keys)
    sort_key = keys[sort_idx]
    values = values[sort_idx, :]
    last_idx = idx_list[time_mask][-1]

    backprop_mask = np.isin(np.array(sorted(unique_time_nodes)), sort_key)

    batch_list.append((backprop_mask, values, time_mask))
  return batch_list


def test_tranucate(train_length, test_data: Data):
  """
  :return partial of sources, destinations and timestamps of test_data \n
  where cut-offed by train_edge_length
  :from author -- this tranucate really indicating that Zebra is perfectly fit for global view of graph, it is very necessary
  to maintain a global tppr dictionary for efficiency in Zebra
  """
  src, dst, tsp, eid = test_data.sources, test_data.destinations, test_data.timestamps, test_data.edge_idxs
  tppr_tranucate_before = copy.deepcopy((src[:train_length], dst[:train_length], tsp[:train_length], eid[:train_length]))
  tppr_tranucate_after_data_object = Data(src[train_length:], dst[train_length:], tsp[train_length:], eid[train_length:], labels=test_data.labels, hash_table=test_data.hash_table, full_feat=(test_data.node_feat, test_data.edge_feat))
  return tppr_tranucate_before, tppr_tranucate_after_data_object

# path = "data/mooc/ml_mooc.npy"
# edge = np.load(path)
def load_feat(d):
    node_feats = None
    if os.path.exists('../data/{}/ml_{}_node.npy'.format(d,d)):
        node_feats = np.load('../data/{}/ml_{}_node.npy'.format(d,d)) 

    edge_feats = None
    if os.path.exists('../data/{}/ml_{}.npy'.format(d,d)):
        edge_feats = np.load('../data/{}/ml_{}.npy'.format(d,d))
    return node_feats, edge_feats


############## load a batch of training data ##############
def get_data(dataset_name):
  graph_df = pd.read_csv('data/{}/ml_{}.csv'.format(dataset_name,dataset_name))

  #edge_features = np.load('../data/{}/ml_{}.npy'.format(dataset_name,dataset_name))
  #node_features = np.load('../data/{}/ml_{}_node.npy'.format(dataset_name,dataset_name)) 
  #node_features, edge_features = load_feat(dataset_name)

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))
  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values
  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
  
  # ensure we get the same graph
  random.seed(2020)
  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)
  n_edges = len(sources)

  test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
  new_test_node_set = set(random.sample(sorted(test_node_set), int(0.1 * n_total_unique_nodes)))
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])
  train_node_set = set(train_data.sources).union(train_data.destinations)
  assert len(train_node_set & new_test_node_set) == 0


  # * the val set can indeed contain the new test node
  new_node_set = node_set - train_node_set
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)

  test_mask = timestamps > test_time
  edge_contains_new_node_mask = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
  new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
  new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])
  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])
  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])


  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,full_data.n_unique_nodes))
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
  print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

  return full_data, train_data, val_data, test_data, \
         new_node_val_data, new_node_test_data, n_total_unique_nodes, n_edges

