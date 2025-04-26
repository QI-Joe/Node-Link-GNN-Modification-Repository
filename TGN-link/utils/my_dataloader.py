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
from typing import Any, Union, Tuple, Optional
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
STATIC = ["mathoverflow", "dblp", "askubuntu", "stackoverflow", "mooc"]
DYNAMIC = ["mathoverflow", "askubuntu", "stackoverflow"]
EMB_SIZE = 64

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
            if len(label) > len(nodes):
                label = np.arange(len(nodes))
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
    
    def edge_replacement(self, df_edge: pd.DataFrame):
        if not isinstance(df_edge, pd.Series):
            df_edge = self.to_numpy(df_edge)
            df_edge = pd.DataFrame(df_edge.T)
        transfor_platform = self.node.node
        transfor_platform = pd.Series(transfor_platform.index, index= transfor_platform.values)
        # given function "map" and data series, iterate through col is the fastest way
        df_edge = df_edge.apply(lambda x: x.map(transfor_platform))
        return df_edge.values.T
    
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


        given_input = copy.deepcopy(src.T)

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
        self.kept_train_mask = None
        self.kept_val_mask = None

        self.node_pos, self.edge_pos = pos
        self.my_n_id = NodeIdxMatching(False, nodes=self.x, label=self.y)
        self.idx2node = self.my_n_id.node
        self.layer2_n_id: pd.DataFrame = None

        self.src_timedifference_sequence: np.ndarray[float] = []
        self.src_previous_destid_sequence: np.ndarray[float] =[]
        self.dest_timedifference_sequence: np.ndarray[float] = []

        self.src_num: int = -1
        self.dest_num: int = -1
    
    def test_fast_sparse_build(self, key: np.ndarray, value: np.ndarray) -> torch.Tensor:
        r"""
        Without considering idx matching, assume that tppr_node idx is consistent with current node idx
        """
        return [[src, dst, value[src, idx]] \
                    for src in range(key.shape[0]) \
                    for idx, dst in enumerate(key[src]) \
                        if dst>0 or value[src, idx] > 0]

    def reverse_idx(self, key: list[list[int]], value: list[list[float]])->list[list[int]]:
        """
        reverse the node index in key back to new sorted node idx
        key:param, a 2-layer nested list stores node grabbed from TPPR list
        """
        indices_sparse_container: list[list[int, int, float]] = []
        for idx, idx_ in enumerate(key):
            weight = value[idx]
            for nid, w in zip(idx_, weight):
                # nid: original node idx need to be replaced
                # w: weight of the node
                if w>0:
                    mask = (self.my_n_id.node.node == nid)
                    # during the tppr updating, becasue this is a progressive incresasing process
                    # current snapshot may not contain the feedback NEIGHBOR node, 
                    # so we need to check if the node is in the current snapshot
                    if mask.sum() == 0:
                        continue
                    new_nid = self.my_n_id.node[mask].values[0][0]
                    indices_sparse_container.append([idx, new_nid, w])
        return indices_sparse_container

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
    
    def establish_two_layer_idx_matching(self, large_idx: NodeIdxMatching) -> None:
        if large_idx.node.shape[0] < self.my_n_id.node.shape[0]: # <= or < ?
            raise ValueError("Input object should owns a parent graph of current stored temporal graph")
        self.layer2_n_id = self.my_n_id.node.merge(large_idx.node, on="node", suffixes=["_subTemporal", "_Temporal"])
        return

    def single_layer_mask(self, mask, sub_nodes: torch.Tensor, match_list: pd.DataFrame, large_graph_length = None):
        large_graph_length = self.pos.size(0)
        node_idx = sub_nodes[mask]
        large_graph_idx = match_list.index_Temporal[match_list.index_subTemporal.isin(node_idx)].values

        new_mask = torch.full((large_graph_length, ), False, dtype=bool)
        new_mask[large_graph_idx]=True
        return new_mask

    def mask_adjustment_two_layer_idx(self):
        """
        temporal_data: Temporal_Dataloader object, marked as Data for type declaration
        temporal_data.x: consistent sequence node index of sub temporal graph, from 0 to n
        temporal_data.pos: if dataset is Mathoverflow, then this is default node features compute from pos_encoding
        otherwise it is defualt node features. in RoLAND model pos.size(0) >> x.size(0)
        temporal_data.idx2node: node matching dataframe, based on original node idx to match corresponding index 
        of different layer

        :Feburary 11th--Depcrecated considering RoLAND doest need this
        this function aims to solve unmatched node features and sub tmeporal graph size. In RoLAND model input is set
        as total number of nodes; to keep on training the model input must has size n x features; where n is total number of nodes
        
        """
        mask_train: torch.Tensor = self.train_mask
        mask_val: torch.Tensor = self.val_mask
        nodes = self.x
        match_list: pd.DataFrame = self.layer2_n_id
        self.kept_train_mask, self.kept_val_mask = self.train_mask, self.val_mask
        self.train_mask = self.single_layer_mask(mask_train, nodes, match_list)
        self.val_mask = self.single_layer_mask(mask_val, nodes, match_list)
        return self


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

        self.graph.x = np.array(self.graph.x)
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

    class Label(enum.Enum):
        c1 = 1
        c2 = 2
        c3 = 3

    def __init__(self, graph: Data) -> None:
        
        super(Temporal_Splitting, self).__init__()
        self.graph = graph 

        if self.graph.edge_attr.any() == None:
            self.graph.edge_attr = np.arange(self.graph.edge_index.shape[0])

        self.n_id = NodeIdxMatching(False, nodes=self.graph.x, label=self.graph.y)
        self.temporal_list: list[Temporal_Dataloader] = []
        self.set_mapping: dict = None
        self.combination = label_match([1, 2, 3])
    
    @dispatch(int, bool)
    def __getitem__(self, idx: int, is_node:bool = False):
        if is_node:
            return self.tracing_dict[idx]
        return self.temporal_list[idx]
    
    @dispatch(int, int)
    def __getitem__(self, list_idx: int, idx: int):
        return self.temporal_list[list_idx][idx]

    def get_map(self, set_input):
        set_input_frozenset = frozenset(set_input)
        return self.set_mapping.get(set_input_frozenset, None)

    def set_map(self, c1, c2, c3):
        set_mapping = {
            frozenset(c1): self.Label.c1.value,
            frozenset(c2): self.Label.c2.value,
            frozenset(c3): self.Label.c3.value
        }
        self.set_mapping = set_mapping

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

        if time_mode == "time":
            span = edge_attr.cpu().numpy()
            duration = kwargs["duration"]
            T = self.sampling_layer_by_time(span = span, duration=duration)
        elif time_mode == "view":
            span = (max(edge_attr) - min(edge_attr)).item()
            snapshot, views = kwargs["snapshot"], kwargs["views"]
            T = self.sampling_layer(snapshot, views, span)

        for idx, start in enumerate(T):
            if start<0.01: continue

            sample_time = start

            end = min(start + span / snapshot, max_time)
            sample_time = (edge_attr <= end) # returns an bool value

            sampled_edges = edge_index[:, sample_time]
            sampled_nodes = np.unique(sampled_edges) # orignal/gobal node index

            y = self.graph.y[sample_time] # in edge prediction task, y should be the same length of edges
            
            nodepos, edgepos = pos
            subpos = (nodepos[self.n_id.sample_idx(sampled_nodes)], edgepos[sample_time])

            temporal_subgraph = Temporal_Dataloader(nodes=sampled_nodes, edge_index=sampled_edges, \
                edge_attr=edge_attr[sample_time], y=y, pos=subpos).get_Temporalgraph()
            
            temporal_subgraphs.append(temporal_subgraph)

        return temporal_subgraphs

def label_match(labels: list):
    label_len = len(labels)

    combination = {}
    outer_ptr = 0
    for ptr in range(1, label_len+1):
        val = itertools.combinations(labels, ptr)
        for v in val:
            combination[v] = outer_ptr
            outer_ptr += 1
    return combination

def time_encoding(timestamp: torch.Tensor, emb_size: int = EMB_SIZE):
    
    timestamps = torch.tensor(timestamp, dtype=torch.float32).unsqueeze(1)
    max_time = timestamps.max() if timestamps.numel() > 0 else 1.0  # Avoid division by zero
    div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
    
    te = torch.zeros(len(timestamps), emb_size)
    te[:, 0::2] = torch.sin(timestamps / max_time * div_term)
    te[:, 1::2] = torch.cos(timestamps / max_time * div_term)
    
    return te

def position_encoding(max_len, emb_size: int = EMB_SIZE)->torch.Tensor:
    pe = torch.zeros(max_len, emb_size)
    position = torch.arange(0, max_len).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def load_dblp_interact(path: str = None, dataset: str = "dblp", *wargs) -> pd.DataFrame:
    edges = pd.read_csv(os.path.join("/mnt/d/CodingArea/Python/Depreciated_data/CLDG/Data/CLDG-datasets/", dataset, '{}.txt'.format(dataset)), sep=' ', names=['src', 'dst', 'time'])
    label = pd.read_csv(os.path.join('/mnt/d/CodingArea/Python/Depreciated_data/CLDG/Data/CLDG-datasets/', dataset, 'node2label.txt'), sep=' ', names=['node', 'label'])

    return edges, label

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

def load_mooc(path:str=None) -> Tuple[pd.DataFrame]:
    feat = pd.read_csv(os.path.join(path, "mooc_action_features.tsv"), sep = '\t')
    general = pd.read_csv(os.path.join(path, "mooc_actions.tsv"), sep = '\t')
    edge_label = pd.read_csv(os.path.join(path, "mooc_action_labels.tsv"), sep = '\t')
    return general, feat, edge_label

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

def get_dataset(path, name: str):
    assert name.lower() in [val.lower() for val in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']]
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name == 'Coauthor-CS'.lower():
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy'.lower():
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS'.lower():
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers'.lower():
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo'.lower():
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())

def load_standard(dataset: str, *wargs) -> tuple[Data, NodeIdxMatching]:
    
    path = osp.expanduser('~/datasets')
    path = osp.join(path, dataset)
    dataset = get_dataset(path, dataset)
    return dataset

def edge_load_mooc(dataset:str):
    auto_path = r"../../TestProejct/Temporal_Dataset/act-mooc/act-mooc"
    edge, feat, label = load_mooc(auto_path)
    # for edge, its column idx is listed as ["ACTIONID", "USERID", "TARGETID", "TIMESTAMP"]
    edge = edge.values
    edge_idx, src2dst, timestamp = edge[:, 0], edge[:, 1:3].T, edge[:, 3]
    
    print(src2dst.dtype, src2dst.shape)
    src2dst = src2dst.astype(np.int64)
    
    edge_pos = feat.iloc[:, 1:].values
    y = label.iloc[:, 1].values
    
    node = np.unique(src2dst).astype(np.int64)
    max_node = int(np.max(node)) + 1
    if dataset == "mooc":
        node = np.unique(src2dst[0])
    node_pos = position_encoding(max_node).numpy()
    # edge_pos = time_encoding(timestamp)
    
    pos = (node_pos, edge_pos)
    graph = Data(x = node, edge_index=src2dst, edge_attr=timestamp, y = y, pos = pos)
    return graph

def load_static_dataset(path: str = None, dataset: str = "mathoverflow", fea_dim: int = 64, *wargs) -> tuple[Temporal_Dataloader, NodeIdxMatching]:
    """
    Now this txt file only limited to loading data in from mathoverflow datasets
    path: (path, last three words of dataset) -> (str, str) e.g. ('data/mathoverflow/sx-mathoverflow-a2q.txt', 'a2q')
    node Idx of mathoverflow is not consistent!
    """
    if dataset[-8:] == "overflow" or dataset == "askubuntu":
        edges, label = load_static_overflow(dataset) if not path else load_static_overflow(dataset, path)
    elif dataset == "dblp":
        edges, label = load_dblp_interact() if not path else load_dblp_interact(path)
    elif dataset == "mooc":
        return edge_load_mooc(dataset), None

    x = label.node.to_numpy()
    nodes = position_encoding(x.max()+1, fea_dim)[x].numpy()
    labels = label.label.to_numpy()

    edge_index = edges.loc[:, ["src", "dst"]].values.T
    start_time = edges.time.min()
    edges.time = edges.time.apply(lambda x: x - start_time)
    time = edges.time.values
    
    time_pos = time_encoding(timestamp=time).numpy()
    pos = (nodes, time_pos)

    graph = Data(x=x, edge_index=edge_index, edge_attr=time, y=labels, pos = pos)
    # neighborloader = NeighborLoader(graph, num_neighbors=[10, 10], batch_size =2048, shuffle = False)
    idxloader = NodeIdxMatching(False, nodes=x, label=labels)
    return graph, idxloader

def load_tsv(path: list[tuple[str]], *wargs) -> tuple[pd.DataFrame]:
    """
    Note this function only for loading data in act-mooc dataset
    """
    dfs:dict = {p[1]: pd.read_csv(p[0], sep='\t') for p in path}

    label = dfs["mooc_action_labels"]
    action_features = dfs["mooc_action_features"]
    actions = dfs["mooc_actions"]
    return label, action_features, actions

def load_example():
    return "node_feat", "node_label", "edge_index", "train_indices", "val_indices", "test_indices"

def data_load(dataset: str, emb_size: int, **wargs) -> tuple[Temporal_Dataloader, Union[NodeIdxMatching|dict]]:
    global EMB_SIZE
    EMB_SIZE = emb_size
    dataset = dataset.lower()
    if dataset in STATIC:
        return load_static_dataset(dataset=dataset, **wargs)
    elif dataset in ["cora", "citeseer", "wikics"] :
        graph = load_standard(dataset, **wargs)[0]
        fake_time = graph.num_edges
        edge_attr = np.arange(fake_time)
        graph.edge_attr = edge_attr
        
        node_pos = graph.x.numpy()
        edge_pos = position_encoding(graph.num_edges, emb_size=emb_size)

        graph.pos=(node_pos, edge_pos)
        nodes = [i for i in range(graph.x.shape[0])]

        graph.x = nodes
        return graph, NodeIdxMatching(False, nodes=nodes, label=graph.y.numpy())
    raise ValueError("Dataset not found")

def t2t1_node_alignment(nodes, t_graph: Temporal_Dataloader, t1_graph: Temporal_Dataloader) -> list[int]:
    raise NotImplementedError("need to implemented for next step")

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
    
    graph.node_pos = graph.node_pos.to(device)
    graph.edge_pos = graph.edge_pos.to(device)

    return graph

def str2bool(given: str)->bool:
    if given in ["True", "true", 1]:
        return True
    return False