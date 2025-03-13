# @pre-announce: this version of dataloader will served for linking data loading, fundmental 
# loading method will be modified to suit for application

import pandas as pd
import numpy as np
import torch
import random
import math
from torch_geometric.data import Data
from torch import Tensor
import copy
import os
from torch_geometric.loader import NeighborLoader
from typing import Any, Union
from multipledispatch import dispatch
import os.path as osp
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T
from datetime import datetime
import itertools
from collections import defaultdict
import enum
from sklearn.preprocessing import scale

MOOC, Mooc_extra = "Temporal_Dataset/act-mooc/act-mooc/", ["mooc_action_features", "mooc_action_labels", "mooc_actions"]
MATHOVERFLOW, MathOverflow_extra = "Temporal_Dataset/mathoverflow/", ["sx-mathoverflow-a2q", "sx-mathoverflow-c2a", "sx-mathoverflow-c2q", "sx-mathoverflow"]
OVERFLOW = r"Temporal_Dataset/"
STATIC = ["mathoverflow", "dblp", "askubuntu", "stackoverflow"]

class NodeIdxMatching(object):

    """
    Not that appliable on train_mask or node_mask, at least in CLDG train_mask should be aggreated within NeigborLoader
    as seed node, while it is computed manually to present "positive pair"
    """

    def __init__(self, is_df: bool, df_node: pd.DataFrame = None, nodes: np.ndarray = [], label: np.ndarray=[]) -> None:
        """
        self.node: param; pd.Dataframe has 2 columns,\n
        'node': means node index in orginal entire graph\n
        'label': corresponding label
        """
        super(NodeIdxMatching, self).__init__()
        self.is_df = is_df
        if is_df: 
            if not df_node: raise ValueError("df_node is required")
            self.node = df_node
        else:
            if not isinstance(nodes, (np.ndarray, list, torch.Tensor)): 
                nodes = list(nodes)
            self.nodes = self.to_numpy(nodes)
            self.node: pd.DataFrame = pd.DataFrame({"node": nodes, "label": label}).reset_index()

    def to_numpy(self, nodes: Union[torch.Tensor, np.array]):
        if isinstance(nodes, torch.Tensor):
            if nodes.device == "cuda:0":
                nodes = nodes.cpu().numpy()
            else: 
                nodes = nodes.numpy()
        return nodes

    def idx2node(self, indices: Union[np.array, torch.Tensor]) -> np.ndarray:
        indices = self.to_numpy(indices)
        node = self.node.node.iloc[indices]
        return node.values
    
    def node2idx(self, node_indices: Union[np.array, torch.Tensor] = None) -> np.ndarray:
        if node_indices is None:
            return np.array(self.node.index)
        node_indices = self.to_numpy(node_indices)
        indices = self.node.node[self.node.node.isin(node_indices)].index
        return indices.values
    
    def edge_restore(self, edges: torch.Tensor, to_tensor: bool = False) -> Union[torch.Tensor, np.array]:
        edges = self.to_numpy(edges)
        df_edges = pd.Series(edges.T).apply(lambda x: x.map(self.node.node))
        if to_tensor: 
            df_edges = torch.tensor(df_edges.values.T)
        return df_edges.values
    
    def get_label_by_node(self, node_indices: Union[torch.Tensor, list[int], np.ndarray]) -> list:
        node_indices = self.to_numpy(node_indices)
        idx_mask: pd.Series = self.node.node[self.node.node.isin(node_indices)].index
        labels: list = self.node.label[idx_mask].values.tolist()
        return labels
    
    def get_label_by_idx(self, idx: Union[torch.Tensor, list[int], np.ndarray]) -> list:
        idx = self.to_numpy(idx)
        return self.node.label[idx].tolist()
    
    def sample_idx(self, node_indices: Union[torch.Tensor, list[int], np.ndarray]) -> torch.Tensor:
        node_indices = self.to_numpy(node_indices)
        idx_mask: pd.Series = self.node.node[self.node.node.isin(node_indices)].index
        return list(idx_mask.values)
    
    def matrix_edge_replacement(self, src: Union[pd.DataFrame|torch.Tensor])->np.ndarray:
        nodes = self.node["node"].values
        match_list = self.node[["node", "index"]].values

        max_size = max(nodes) + 1
        space_array = np.zeros((max_size,), dtype=np.int32)
        idx = match_list[:, 0]
        values = match_list[:, 1]

        space_array[idx] = values


        given_input = copy.deepcopy(src.numpy().T)

        col1, col2 = given_input[:, 0], given_input[:, 1]
        replace_col1 = space_array[col1]
        replced_col2 = space_array[col2]
        replaced_given = np.vstack((replace_col1, replced_col2)) # [2, n]
        return replaced_given


class Temporal_Dataloader(Data):
    """
    an overrided class of Data, to store splitted temporal data and reset their index 
    which support for fast local idx and global idx mapping/fetching
    """
    def __init__(self, nodes: Union[list, np.ndarray], edge_index: np.ndarray,\
                  edge_attr: Union[list|np.ndarray], y: list,\
                    pos: tuple[torch.Tensor]) -> None:
        
        super(Temporal_Dataloader, self).__init__(x = nodes, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)
        self.x = nodes
        self.edge_index = edge_index
        self.ori_edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y

        self.node_pos, self.edge_pos = pos
        self.general_pos = None
        self.my_n_id = NodeIdxMatching(False, nodes=self.x, label=self.y)
        self.idx2node = self.my_n_id.node
        self.layer2_n_id: pd.DataFrame = None

        self.src_timedifference_sequence: np.ndarray[float] = []
        self.src_previous_destid_sequence: np.ndarray[float] =[]
        self.dest_timedifference_sequence: np.ndarray[float] = []

        self.src_num: int = -1
        self.dest_num: int = -1

    def get_node(self):
        return self.my_n_id.node2idx(None)
    
    def get_edge_index(self):
        return self.my_n_id.matrix_edge_replacement(self.edge_index)
    
    def get_Temporalgraph(self):
        self.x = self.get_node()
        self.edge_index = self.get_edge_index()
        return self

    def up_report(self):
        reverse: pd.DataFrame = pd.DataFrame(self.idx2node.index, index=self.idx2node.values)
        return reverse.to_dict()


class Dynamic_Dataloader(object):
    """
    a class to store a group of temporal dataset, calling with updated event running
    return a temporal data every time
    """
    def __init__(self, data: list[Data], graph: Data) -> None:
        super(Dynamic_Dataloader, self).__init__()
        self.data = data
        self.graph = graph
        self.num_classes = int(self.graph.y.max().item() + 1)
        self.len = len(data)

        self.num_nodes = self.graph.x.shape[0]
        self.num_edges = self.graph.edge_index.shape[-1]
        self.temporal = len(data)

        self.temporal_event = None

    def __getitem__(self, idx)-> Temporal_Dataloader:
        return self.data[idx]
    
    def get_temporal(self) -> Union[Data|Temporal_Dataloader|None]:
        if not self.temporal_event:
            self.update_event()
        return self.temporal_event
    
    def get_T1graph(self, timestamp: int) -> Union[Data|Temporal_Dataloader|None]:
        if timestamp>=self.len-1:
            return self.data[self.len-1]
        if len(self.data) <= 1:
            if self.data.is_empty(): return self.graph
            return self.data[0]
        return self.data[timestamp+1]

    def update_event(self, timestamp: int = -1):
        if timestamp>=self.len-1:
            return
        self.temporal_event = self.data[timestamp+1]


class Temporal_Splitting(object):

    def __init__(self, graph: Data) -> None:
        
        super(Temporal_Splitting, self).__init__()
        self.graph = graph 

        if self.graph.edge_attr == None:
            self.graph.edge_attr = np.arange(self.graph.edge_index.size(1))

        self.n_id = NodeIdxMatching(False, nodes=self.graph.x, label=self.graph.y)
        self.temporal_list: list[Temporal_Dataloader] = []
        self.set_mapping: dict = None
    
    @dispatch(int, bool)
    def __getitem__(self, idx: int, is_node:bool = False):
        if is_node:
            return self.tracing_dict[idx]
        return self.temporal_list[idx]
    
    @dispatch(int, int)
    def __getitem__(self, list_idx: int, idx: int):
        return self.temporal_list[list_idx][idx]

    def constrct_tracing_dict(self, temporal_list: list[Temporal_Dataloader]) -> None:
        tracing_dict = {}
        for idx, temporal in enumerate(temporal_list):
            temporal_dict: dict[int: int] = temporal.up_report()
            tracing_dict = {**tracing_dict, **{key: [val, idx] for key, val in temporal_dict.items()}}
        self.tracing_dict = tracing_dict
        return

    def sampling_layer(self, snapshots: int, views: int, span: float, strategy: str="sequential"):
        T = []
        if strategy == 'random':
            T = [random.uniform(0, span * (snapshots - 1) / snapshots) for _ in range(views)]
        elif strategy == 'low_overlap':
            if (0.75 * views + 0.25) > snapshots:
                return "The number of sampled views exceeds the maximum value of the current policy."
            start = random.uniform(0, span - (0.75 * views + 0.25) * span /  snapshots)
            T = [start + (0.75 * i * span) / snapshots for i in range(views)]
        elif strategy == 'high_overlap':
            if (0.25 * views + 0.75) > snapshots:
                return "The number of sampled views exceeds the maximum value of the current policy."
            start = random.uniform(0, span - (0.25 * views + 0.75) * span /  snapshots)
            T = [start + (0.25 * i * span) / snapshots for i in range(views)]
        elif strategy == "sequential":
            T = [span * i / (snapshots-1) for i in range(1, snapshots)]
            if views > snapshots:
                return "The number of sampled views exceeds the maximum value of the current policy."
            T = random.sample(T, views)
        T= sorted(T)
        if T[0] == float(0):
            T.pop(0)
        return T

    def sampling_layer_by_time(self, span, duration: int = 30):
        """
        span :param; entire timestamp, expected in Unix timestamp such as 1254192988
        duration: param; how many days
        """
        Times = [datetime.fromtimestamp(stamp).strftime("%Y-%m-%d") for stamp in span]
        start_time = Times[0]

        T_duration: list[int] = []

        for idx, tim in enumerate(span):
            if Times[idx] - start_time >=duration:
                T_duration.append(tim)
                start_time = Times[idx]
        
        return T_duration

    def edge_special_generating(self, graph: Union[Data | Temporal_Dataloader], temporal_idx: int):
        numpy_edges = graph.edge_index.numpy() if isinstance(graph.edge_index, Tensor) else graph.edge_index
        time_series = graph.edge_attr.numpy() if isinstance(graph.edge_attr, Tensor) else graph.edge_attr
        
        src, dest = numpy_edges[0, :], numpy_edges[1,:]
        dest_num = len(np.unique(dest))
        src_num = len(np.unique(src))

        item_current_timestamp = defaultdict(float)
        item_timedifference_sequence: list[float] = []
        for idx, des in enumerate(dest):
            timestamp = time_series[idx]
            item_timedifference_sequence.append(timestamp - item_current_timestamp[idx])
            item_current_timestamp[des] = timestamp

        user_time_diiference_sequence: list[float] = []
        user_current_timestamp = defaultdict(float)
        user_previsou_itemid_sequence: list[float] = []
        user_latest_itemid = defaultdict(lambda: dest_num)

        for idx, user in enumerate(src):
            timestamp = time_series[idx] # record current time
            user_time_diiference_sequence.append(timestamp-user_current_timestamp[user]) # fetch the lastest time stamp
            user_current_timestamp[user] = timestamp # update the lastest timestamp
            """
            user_latest_itemid record the latest interact destination nodes, here, 
            it is suspected that item2id, and user2id is used to reindex the node id;
            given library already reindexed node so here no necessary of user2id or item2id
            """
            user_previsou_itemid_sequence.append(user_latest_itemid[user])
            user_latest_itemid[user]=dest[idx]
    
        user_time_diiference_sequence = scale(np.array(user_time_diiference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

        """
        not on GPU first, may need to convert back to numpy for calculation first
        """
        graph.src_timedifference_sequence = user_time_diiference_sequence
        graph.src_previous_destid_sequence = np.array(user_previsou_itemid_sequence)
        graph.dest_timedifference_sequence = item_timedifference_sequence

        graph.src_num = src_num
        graph.dest_num = dest_num

        return graph


    def temporal_splitting(self, time_mode: str, **kwargs) -> list[Data]:
        """
        currently only suitable for CLDG dataset, to present flexibilty of function Data\n
        log 12.3:\n
        Temporal and Any Dynamic data loader will no longer compatable with static graph
        here we assume edge_attr means time_series in default
        """
        edge_index = self.graph.edge_index
        edge_attr = self.graph.edge_attr
        pos = self.graph.pos

        max_time = max(edge_attr)
        temporal_subgraphs = []

        T: list = []

        span = (max(edge_attr) - min(edge_attr)).item()
        snapshot, views = kwargs["snapshot"], kwargs["views"]
        T = self.sampling_layer(snapshot, views, span)

        for idx, start in enumerate(T):
            if start<0.01: continue

            sample_time = start

            end = min(start + span / snapshot, max_time)
            sample_time = (edge_attr <= end) # returns an bool value

            sampled_edges = edge_index[:, sample_time]
            sampled_nodes = torch.unique(sampled_edges) # orignal/gobal node index

            y = self.graph.y[sample_time]
            subpos = pos[sample_time]

            temporal_subgraph = Temporal_Dataloader(nodes=sampled_nodes, edge_index=sampled_edges, \
                edge_attr=edge_attr[sample_time], y=y, pos=subpos).get_Temporalgraph()
            
            # JODIE Data format fitting and y from node-id to edge-id
            temporal_subgraph = self.edge_special_generating(graph = temporal_subgraph, temporal_idx=idx)
            # temporal_subgraph.y = temporal_subgraph.y[temporal_subgraph.edge_index[0]]
            
            temporal_subgraphs.append(temporal_subgraph)

        return temporal_subgraphs


def time_encoding(timestamp: torch.Tensor, emb_size: int = 64):
    
    timestamps = torch.tensor(timestamp, dtype=torch.float32).unsqueeze(1)
    max_time = timestamps.max() if timestamps.numel() > 0 else 1.0  # Avoid division by zero
    div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
    
    te = torch.zeros(len(timestamps), emb_size)
    te[:, 0::2] = torch.sin(timestamps / max_time * div_term)
    te[:, 1::2] = torch.cos(timestamps / max_time * div_term)
    
    return te

def position_encoding(max_len, emb_size)->torch.Tensor:
    pe = torch.zeros(max_len, emb_size)
    position = torch.arange(0, max_len).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def load_mooc_interact(path: str = None, dataset: str = "mooc", *wargs) -> tuple[pd.DataFrame, np.ndarray]:
    mooc_head = ['user_id', 'item_id', 'timestamp', 'state_label', 'f1', 'f2', 'f3', 'f4']

    edges = pd.read_csv(os.path.join("./data/", dataset, '{}.csv'.format(dataset)), sep=',', names=mooc_head, header=None, skiprows=1)
    new_edge, feature = edges.iloc[:, :-4], edges.iloc[:, -4:].values

    return new_edge, feature

def load_mathoverflow_interact(path: str = MATHOVERFLOW, *wargs) -> pd.DataFrame:
    edges = pd.read_csv(os.path.join(path, "sx-mathoverflow"+".txt"), sep=' ', names=['src', 'dst', 'time'])
    label = pd.read_csv(os.path.join(path, "node2label"+".txt"), sep=' ', names=['node', 'label'])
    return edges, label

def get_combination(labels: list[int]) -> dict:
    """
    :param labels: list of unique labels, for overflow it is fixed as [1,2,3]
    :return: a dictionary that stores all possible combination of labels, usually is 6
    """
    unqiue_node = len(labels)

    combination: dict = {}
    outer_ptr = 0
    for i in range(1, unqiue_node+1):
        pairs = itertools.combinations(labels, i)
        for pair in pairs:
            combination[pair] = outer_ptr
            outer_ptr += 1
    return combination

def load_static_overflow(prefix: str, path: str=None, *wargs) -> tuple[Data, NodeIdxMatching]:
    dataset = "sx-"+prefix
    path = OVERFLOW + prefix + r"/static"
    edges = pd.read_csv(os.path.join(path, dataset+".txt"), sep=' ', names=['src', 'dst', 'time'])
    label = pd.read_csv(os.path.join(path, "node2label.txt"), sep=' ', names=['node', 'label'])
    return edges, label

def load_dynamic_overflow(prefix: str, path: str=None, *wargs) -> tuple[pd.DataFrame, dict]:
    dataset = prefix
    path = OVERFLOW + prefix + r"/dynamic"
    labels: list = [1,2,3]
    edges = pd.read_csv(os.path.join(path, dataset+".txt"), sep=' ', names=['src', 'dst', 'time', 'appearance'])
    combination_dict = get_combination(labels)
    
    return edges, combination_dict

def dynamic_label(edges: pd.DataFrame, combination_dict: dict) -> pd.DataFrame:
    """
    Very slow when facing large dataset. Recommend to use function matrix_dynamic_label
    """
    unique_node = edges[["src", "dst"]].stack().unique()
    node_label: list[tuple[int, int]] = []
    for node in unique_node:
        appearance = edges[(edges.src == node) | (edges.dst == node)].apprarance.values
        appearance = tuple(set(appearance))
        node_label.append((node, combination_dict[appearance]))
    return pd.DataFrame(node_label, columns=["node", "label"])

def load_static_dataset(path: str = None, dataset: str = "mathoverflow", fea_dim: int = 64, *wargs) -> tuple[Temporal_Dataloader, NodeIdxMatching]:
    """
    Now this txt file only limited to loading data in from mathoverflow datasets
    path: (path, last three words of dataset) -> (str, str) e.g. ('data/mathoverflow/sx-mathoverflow-a2q.txt', 'a2q')
    node Idx of mathoverflow is not consistent!
    """
    if dataset == "mooc":
        edges, edge_feature = load_mooc_interact() if not path else load_mooc_interact(path)
    else: 
        raise NotImplemented("Method not implmented.")

    edge_index = edges.loc[:, ["user_id", "item_id"]].values.T
    x = np.unique(edge_index.values.flatten())
    labels = edges.state_label.values 
    start_time = edges.time.min()
    edges.time = edges.time.apply(lambda x: x - start_time)
    time: np.ndarray = edges.time.values
    
    time_pos = time_encoding(timestamp=time).numpy()
    pos = np.vstack((edge_feature, time_pos))

    graph = Data(x=x, edge_index=edge_index, edge_attr=time, y=labels, pos = pos)
    idxloader = NodeIdxMatching(False, nodes=x, label=labels)

    """
    :param x -- numpy x, in shape of (num_nodes, )
    :param edge_index -- numpy edge index, in shpae of (num_edges, 2)
    :param time -- numpy time series, in shape of (num_edges,)
    :param labels -- numpy state label of edges, in shape of (num_edges, )
    :param pos -- combination of edge_feature and edge_time_feature, in shape of (num_edges, dim_edge_fea + dim_edge_time)
    """
    return graph, idxloader



def load_example():
    return "node_feat", "node_label", "edge_index", "train_indices", "val_indices", "test_indices"

def data_load(dataset: str, **wargs) -> tuple[Temporal_Dataloader, Union[NodeIdxMatching|dict]]:
    if dataset in STATIC:
        return load_static_dataset(dataset=dataset, **wargs)
    raise ValueError("Dataset not found")

    
def to_cuda(graph: Union[Data, Temporal_Dataloader], device:str = "cuda:0"):
    device = torch.device(device)
    if not isinstance(graph.x, torch.Tensor):
        graph.x = torch.tensor(graph.x).to(device)
    if not isinstance(graph.edge_index, torch.Tensor):
        graph.edge_index = torch.tensor(graph.edge_index).to(device)
    if not isinstance(graph.edge_attr, torch.Tensor):
        graph.edge_attr = torch.tensor(graph.edge_attr).to(device)
    if not isinstance(graph.y, torch.Tensor) or graph.y.device != device:
        graph.y = torch.tensor(graph.y).to(device)
    
    graph.general_pos = graph.node_pos.to(device)
    graph.general_pos = graph.edge_pos.to(device)

    return graph

def str2bool(given: str)->bool:
    if given in ["True", "true", 1]:
        return True
    return False