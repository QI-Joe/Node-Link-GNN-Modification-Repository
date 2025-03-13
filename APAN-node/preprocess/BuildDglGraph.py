import dgl
import torch
import argparse
import pandas as pd
import numpy as np
from dgl.data.utils import save_graphs
from my_dataloader import Temporal_Dataloader, Temporal_Splitting, Dynamic_Dataloader, data_load

import dgl.function as fn

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('-d', '--data', type=str, choices=["cora", "dblp", "alipay"], help='Dataset name (eg. wikipedia or reddit)',
                        default='alipay')
parser.add_argument('--new_node_count', action='store_true',
                        help='count how many nodes are not in training set')  
parser.add_argument('--snapshot', default=10, type=int, help="number of snapshot in tmeporal graph NN")  
args = parser.parse_args()
args.new_node_count = True

snapshot = args.snapshot
dataset = args.data

graph, idxloader = data_load(dataset=dataset)
graph_dataloader = Temporal_Splitting(graph=graph).temporal_splitting(time_mode="view", snapshot=snapshot, view = snapshot-2)
dataloader = Dynamic_Dataloader(data = graph_dataloader, graph = graph)

graph_df = pd.read_csv('./data/{}.csv'.format(args.data))
edge_features = np.load('./data/{}.npy'.format(args.data))
nfeat_dim = edge_features.shape[1]

pyg_data: Temporal_Dataloader = dataloader.get_temporal()

src = torch.tensor(pyg_data.edge_index[0])
dst = torch.tensor(pyg_data.edge_index[1])
label = torch.tensor(pyg_data.y, dtype=torch.float32)
timestamp = torch.tensor(pyg_data.edge_attr, dtype=torch.float32)
edge_feat = pyg_data.edge_pos.type(torch.float32)

g = dgl.graph((torch.cat([src,dst]), torch.cat([dst,src])))
len_event = src.shape[0]

g.edata['label'] = label.repeat(2).squeeze()
g.edata['timestamp'] = timestamp.repeat(2).squeeze()
g.edata['feat'] = edge_feat.repeat(2,1).squeeze()

print(g)
save_graphs(f"./data/{args.data}.bin", g)

if args.new_node_count:
    origin_num_edges = g.num_edges()//2
    train_eid = torch.arange(0, int(0.7 * origin_num_edges))
    un_train_eid = torch.arange(int(0.7 * origin_num_edges), origin_num_edges)

    train_g = dgl.graph(g.find_edges(train_eid))
    val_n_test_g = dgl.compact_graphs(dgl.graph(g.find_edges(un_train_eid)))

    print(f'total nodes: {g.num_nodes()}, training nodes: {train_g.num_nodes()}, val_n_test nodes: {val_n_test_g.num_nodes()}')
    old_nodes = val_n_test_g.num_nodes()-g.num_nodes()+train_g.num_nodes()
    print(f'old nodes in val_n_test: {old_nodes} ({round((old_nodes)*100/val_n_test_g.num_nodes(),4)}%)')
    new_nodes = g.num_nodes()-train_g.num_nodes()
    print(f'new nodes in val_n_test: {new_nodes} ({round((new_nodes)*100/val_n_test_g.num_nodes(),4)}%)')
