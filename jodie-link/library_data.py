'''
This is a supporting library for the loading the data.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

from __future__ import division
import numpy as np
import random
import sys
import operator
import copy
from collections import defaultdict
import numpy as np
import random
from sklearn.preprocessing import scale
from my_dataloader import Temporal_Dataloader

# LOAD THE NETWORK
def load_network(args, time_scaling=True):
    '''
    This function loads the input network.

    The network should be in the following format:
    One line per interaction/edge.
    Each line should be: user, item, timestamp, state label, array of features.
    Timestamp should be in cardinal format (not in datetime).
    State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
    Feature list can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions. 
    '''

    network = args.network
    datapath = args.datapath

    user_sequence = []
    item_sequence = []
    label_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = []

    print("\n\n**** Loading %s network from file: %s ****" % (network, datapath))
    f = open(datapath,"r")
    f.readline()
    for cnt, l in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list 
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        if start_timestamp is None:
            start_timestamp = float(ls[2])
        timestamp_sequence.append(float(ls[2]) - start_timestamp) 
        y_true_labels.append(int(ls[3])) # label = 1 at state change, 0 otherwise
        feature_sequence.append(list(map(float,ls[4:])))
    f.close()

    user_sequence = np.array(user_sequence) 
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    print("Formating item sequence")
    nodeid = 0
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    print("Formating user sequence")
    nodeid = 0
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: num_items)
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    print("*** Network loading completed ***\n\n")
    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, \
        feature_sequence, \
        y_true_labels]

def t2t1_node_alignment(t_nodes: set, t: Temporal_Dataloader, t1: Temporal_Dataloader):
    t_list = t.my_n_id.node.values
    t1_list = t1.my_n_id.node.values

    t2t1 = t_list[np.isin(t_list[:, 0], list(t_nodes)), 1].tolist()
    t1_extra = list(set(t1_list[:,1]) - set(t_list[:,1]))

    new_nodes = sorted(t2t1+t1_extra) # here the node is original nodes
    resort_nodes = t1_list[np.isin(t1_list[:,1], new_nodes), 0].tolist() # here we match the original nodes back to new idxed node
    
    t1_src = np.isin(t1.edge_index[0], resort_nodes)
    t1_dst = np.isin(t1.edge_index[1], resort_nodes)

    return t1_src|t1_dst, ~t1_src|~t1_dst

def split_edges_for_link_prediction(data: Temporal_Dataloader, t1_data: Temporal_Dataloader, train_ratio: float = 0.85, random_seed: int = 2025):
    """
    Splits the edges of data into train and val sets by train_ratio (e.g., 0.85 : 0.15).
    Returns two dictionaries of arrays, for train and val, each containing:
    user_sequence_id, item_sequence_id, timestamp_sequence, feature_sequence,
    user_timediffs_sequence, item_timediffs_sequence, user_previous_itemid_sequence, y_true
    The arrays remain in the same shape/order as JODIE typically expects.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Original full edge sequences
    full_user_seq  = data.edge_index[0]
    full_item_seq  = data.edge_index[1]
    full_timestamps= data.edge_attr
    full_features  = data.edge_pos if data.edge_pos is not None else None
    full_user_tdiff= data.src_timedifference_sequence
    full_item_tdiff= data.dest_timedifference_sequence
    full_prev_item = data.src_previous_destid_sequence
    full_y         = data.y[full_user_seq]  # (y[user_sequence_id])

    total_events   = full_user_seq.shape[0]
    indices        = np.arange(total_events)
    # Shuffle or sort by time – choose whichever is consistent with your training
    # For a simple random split:
    np.random.shuffle(indices)

    train_size   = int(total_events * train_ratio)
    train_idx    = indices[:train_size]
    val_idx      = indices[train_size:]

    src_unique_nodes = np.unique(full_user_seq)
    dst_unique_nodes = np.unique(full_item_seq)
    # Inductive validation set creation
    # Identify unique users and items in the training set
    train_users = set(full_user_seq[train_idx])
    train_items = set(full_item_seq[train_idx])
    
    src_mask_node_set = set(random.sample(sorted(train_users), int(0.1 * src_unique_nodes.shape[0])))
    src_inductive_nodes = list(src_mask_node_set)
    
    dst_mask_node_set = set(random.sample(sorted(train_items), int(0.1 * dst_unique_nodes.shape[0])))
    dst_inductive_nodes = list(dst_mask_node_set)

    
    val_user = full_user_seq[val_idx]
    val_item = full_item_seq[val_idx]

    # Filter validation indices to include only edges with new users or items
    users_val_nn_idx_mask = np.isin(val_user, src_inductive_nodes)
    items_val_nn_idx_mask = np.isin(val_item, dst_inductive_nodes)
    
    val_nn_mask = users_val_nn_idx_mask | items_val_nn_idx_mask
    if users_val_nn_idx_mask.sum() == 0 or items_val_nn_idx_mask.sum() == 0:
        val_nn_mask = users_val_nn_idx_mask | items_val_nn_idx_mask
        print("Conduct in New-Old inductive test")

    val_nn_idx = val_idx[val_nn_mask]

    train_user_array, train_item_array = full_user_seq[train_idx], full_item_seq[train_idx]

    user_train_idx_mask = ~np.isin(train_user_array, src_inductive_nodes)
    item_train_idx_mask = ~np.isin(train_item_array, dst_inductive_nodes)
    
    train_idx = train_idx[user_train_idx_mask*item_train_idx_mask]


    nn_test_mask, old_test = t2t1_node_alignment(set(src_inductive_nodes)|set(dst_inductive_nodes), data, t1_data)

    # Helper to slice if the array is not None
    def safe_slice(arr, idx):
        return arr[idx] if arr is not None else None

    # Prepare dictionary for train portion
    train_dict = {
        "user_sequence_id"          : full_user_seq[train_idx],
        "item_sequence_id"          : full_item_seq[train_idx],
        "timestamp_sequence"        : full_timestamps[train_idx],
        "feature_sequence"          : safe_slice(full_features, train_idx),
        "user_timediffs_sequence"   : full_user_tdiff[train_idx],
        "item_timediffs_sequence"   : full_item_tdiff[train_idx],
        "user_previous_itemid_seq"  : full_prev_item[train_idx],
        "y_true"                    : full_y[train_idx]
    }

    # Prepare dictionary for val portion
    val_dict = {
        "user_sequence_id"          : full_user_seq[val_idx],
        "item_sequence_id"          : full_item_seq[val_idx],
        "timestamp_sequence"        : full_timestamps[val_idx],
        "feature_sequence"          : safe_slice(full_features, val_idx),
        "user_timediffs_sequence"   : full_user_tdiff[val_idx],
        "item_timediffs_sequence"   : full_item_tdiff[val_idx],
        "user_previous_itemid_seq"  : full_prev_item[val_idx],
        "y_true"                    : full_y[val_idx]
    }
    

    val_nn_dict = {
        "user_sequence_id"          : full_user_seq[val_nn_idx],
        "item_sequence_id"          : full_item_seq[val_nn_idx],
        "timestamp_sequence"        : full_timestamps[val_nn_idx],
        "feature_sequence"          : safe_slice(full_features, val_nn_idx),
        "user_timediffs_sequence"   : full_user_tdiff[val_nn_idx],
        "item_timediffs_sequence"   : full_item_tdiff[val_nn_idx],
        "user_previous_itemid_seq"  : full_prev_item[val_nn_idx],
        "y_true"                    : full_y[val_nn_idx]
    }
    
    test_dict = {
        "user_sequence_id"          : t1_data.edge_index[0],
        "item_sequence_id"          : t1_data.edge_index[1],
        "timestamp_sequence"        : t1_data.edge_attr,
        "feature_sequence"          : safe_slice(t1_data.edge_pos, slice(None)),
        "user_timediffs_sequence"   : t1_data.src_timedifference_sequence,
        "item_timediffs_sequence"   : t1_data.dest_timedifference_sequence,
        "user_previous_itemid_seq"  : t1_data.src_previous_destid_sequence,
        "y_true"                    : t1_data.y
    }
    
    test_nn_dict = {
        "user_sequence_id"          : t1_data.edge_index[0][nn_test_mask],
        "item_sequence_id"          : t1_data.edge_index[1][nn_test_mask],
        "timestamp_sequence"        : t1_data.edge_attr[nn_test_mask],
        "feature_sequence"          : safe_slice(t1_data.edge_pos, nn_test_mask),
        "user_timediffs_sequence"   : t1_data.src_timedifference_sequence[nn_test_mask],
        "item_timediffs_sequence"   : t1_data.dest_timedifference_sequence[nn_test_mask],
        "user_previous_itemid_seq"  : t1_data.src_previous_destid_sequence[nn_test_mask],
        "y_true"                    : t1_data.y[nn_test_mask]
    }
    return train_dict, val_dict, val_nn_dict, test_dict, test_nn_dict
