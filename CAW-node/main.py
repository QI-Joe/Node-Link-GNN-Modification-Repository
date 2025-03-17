import pandas as pd
from log import *
from eval import *
from utils import *
from train import *
#import numba
from module import CAWN
from graph import NeighborFinder
import resource
from my_dataloader import to_cuda, Temporal_Dataloader, Temporal_Splitting, Dynamic_Dataloader, data_load, t2t1_node_alignment
import copy

args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
GPU = args.gpu
USE_TIME = args.time
ATTN_AGG_METHOD = args.attn_agg_method
ATTN_MODE = args.attn_mode
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
POS_ENC = args.pos_enc
POS_DIM = args.pos_dim
WALK_POOL = args.walk_pool
WALK_N_HEAD = args.walk_n_head
WALK_MUTUAL = args.walk_mutual if WALK_POOL == 'attn' else False
TOLERANCE = args.tolerance
CPU_CORES = args.cpu_cores
NGH_CACHE = args.ngh_cache
VERBOSITY = args.verbosity
AGG = args.agg
SEED = args.seed
SNAPSHOT = args.snapshot
VIEW = SNAPSHOT - 2
assert(CPU_CORES >= -1)
set_random_seed(SEED)

# Load data and sanity check
graph, idxloader = data_load(dataset=DATA)
idx_list = Temporal_Splitting(graph=graph).temporal_splitting(time_mode="view", snapshot = SNAPSHOT, views = VIEW)
temporaloader = Dynamic_Dataloader(idx_list, graph)

num_cls = graph.y.max()+1
for sp in range(VIEW):
    temporalgraph = temporaloader.get_temporal()
    t1_temporal = temporaloader.get_T1graph(sp)

    src_l = temporalgraph.edge_index[0]
    dst_l = temporalgraph.edge_index[1]
    e_idx_l = np.arange(len(src_l))
    label_l = temporalgraph.y
    ts_l = temporalgraph.edge_attr

    e_feat = temporalgraph.edge_pos
    n_feat = temporalgraph.node_pos

    max_idx = max(src_l.max(), dst_l.max())+1
    assert(np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx or (not math.isclose(1, args.data_usage)))  # all nodes except node 0 should appear and be compactly indexed
    assert(n_feat.shape[0] == max_idx or (not math.isclose(1, args.data_usage)))  # the nodes need to map one-to-one to the node feat matrix

    # split and pack the data by generating valid train/val/test mask according to the "mode"
    val_time = int(temporalgraph.edge_attr.shape[0]*0.85)
    # val_time, test_time = list(np.quantile(temporalgraph.edge_attr.numpy(), [0.85, 0.90]))
    
    if args.mode == 't':
        print('Transductive training...')
        valid_train_flag = e_idx_l<val_time
        valid_val_flag = e_idx_l > val_time
        valid_test_flag = np.ones(t1_temporal.edge_attr.shape)

    else:
        assert(args.mode == 'i')
        print('Inductive training...')
        # pick some nodes to mask (i.e. reserved for testing) for inductive setting
        total_node_set = set(np.unique(temporalgraph.edge_index))
        num_total_unique_nodes = len(total_node_set)
        mask_node_set = set(random.sample(sorted(set(src_l[e_idx_l > val_time]).union(set(dst_l[e_idx_l > val_time]))), int(0.1 * num_total_unique_nodes)))
        mask_node_set = list(mask_node_set)
        
        # align the node from t to t1
        t2t1_node: list[int] = t2t1_node_alignment(mask_node_set, temporalgraph, t1_temporal)

        mask_src_flag = np.isin(src_l, mask_node_set).astype(np.int8)
        mask_dst_flag = np.isin(dst_l, mask_node_set).astype(np.int8)

        none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
        valid_train_flag = (e_idx_l <= val_time) * (none_mask_node_flag > 0.5)
        valid_val_flag = (e_idx_l > val_time) * (none_mask_node_flag > 0.5)  # both train and val edges can not contain any masked nodes
        
        mask_src_flag_t1 = np.isin(t1_temporal.edge_index[0], t2t1_node).astype(np.int8)
        mask_dst_flag_t1 = np.isin(t1_temporal.edge_index[1], t2t1_node).astype(np.int8)
        t1_none_mask_node_flag = (1 - mask_src_flag_t1) * (1 - mask_dst_flag_t1)
        valid_test_flag = (none_mask_node_flag < 0.5)  # test edges must contain at least one masked node
        
        # for inductive learning, temporal graph should not be considered...
        valid_test_new_new_flag = mask_src_flag * mask_dst_flag 
        # new_flag: node intersection in mask_src_flag and mask_dst_flag 
        valid_test_new_old_flag = (valid_test_flag.astype(int) - valid_test_new_new_flag.astype(int)).astype(bool)
        # old_flag: node union - node intersection in mask_src_flag and mask_dst_flag
        print('Sampled {} nodes (10 %) which are masked in training and reserved for testing...'.format(len(mask_node_set)))

    # split data according to the mask
    src_label, edge_label = label_l[src_l], label_l[dst_l]
    train_src_l, train_dst_l, train_ts_l, train_e_idx_l = src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag]
    train_label_l = (src_label[valid_train_flag], edge_label[valid_train_flag])
    
    val_src_l, val_dst_l, val_ts_l, val_e_idx_l = src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag]
    val_label_l = (src_label[valid_val_flag], edge_label[valid_val_flag])

    test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = t1_temporal.edge_index[0], \
    t1_temporal.edge_index[1], t1_temporal.edge_attr, np.arange(t1_temporal.edge_index.shape[1]), t1_temporal.y
    
    # test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], label_l[valid_test_flag]
    
    if args.mode == 'i':
        # need to be modified
        test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l, test_label_new_new_l = src_l[valid_test_new_new_flag], dst_l[valid_test_new_new_flag], ts_l[valid_test_new_new_flag], e_idx_l[valid_test_new_new_flag], label_l[valid_test_new_new_flag]
        test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l, test_label_new_old_l = src_l[valid_test_new_old_flag], dst_l[valid_test_new_old_flag], ts_l[valid_test_new_old_flag], e_idx_l[valid_test_new_old_flag], label_l[valid_test_new_old_flag]
    
    train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
    val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
    train_val_data = (train_data, val_data)

    # create two neighbor finders to handle graph extraction.
    # for transductive mode all phases use full_ngh_finder, for inductive node train/val phases use the partial one
    # while test phase still always uses the full one
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(full_adj_list, bias=args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)
    partial_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        partial_adj_list[src].append((dst, eidx, ts))
        partial_adj_list[dst].append((src, eidx, ts))
    for src, dst, eidx, ts in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l):
        partial_adj_list[src].append((dst, eidx, ts))
        partial_adj_list[dst].append((src, eidx, ts))
    partial_ngh_finder = NeighborFinder(partial_adj_list, bias=args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)
    ngh_finders = partial_ngh_finder, full_ngh_finder

    # create random samplers to generate train/val/test instances
    train_rand_sampler = RandEdgeSampler((train_src_l, ), (train_dst_l, ))
    val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l))
    test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
    rand_samplers = train_rand_sampler, val_rand_sampler

    # multiprocessing memory setting
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (200*args.bs, rlimit[1]))

    # model initialization
    device = torch.device('cuda:{}'.format(GPU))
    cawn = CAWN(n_feat, e_feat, agg=AGG,
                num_layers=NUM_LAYER, use_time=USE_TIME, attn_agg_method=ATTN_AGG_METHOD, attn_mode=ATTN_MODE,
                n_head=ATTN_NUM_HEADS, drop_out=DROP_OUT, pos_dim=POS_DIM, pos_enc=POS_ENC, num_classes = num_cls,
                num_neighbors=NUM_NEIGHBORS, walk_n_head=WALK_N_HEAD, walk_mutual=WALK_MUTUAL, walk_linear_out=args.walk_linear_out, walk_pool=args.walk_pool,
                cpu_cores=CPU_CORES, verbosity=VERBOSITY, get_checkpoint_path=None)
    cawn.to(device)
    optimizer = torch.optim.Adam(cawn.parameters(), lr=LEARNING_RATE)
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

    # start train and val phases
    train_val(train_val_data, cawn, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, ngh_finders, rand_samplers, None)

# final testing
cawn.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l)
print('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc, test_ap))
test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1]*6
if args.mode == 'i':
    test_new_new_acc, test_new_new_ap, test_new_new_f1, test_new_new_auc = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_label_new_new_l, test_e_idx_new_new_l)
    print('Test statistics: {} new-new nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_new_acc, test_new_new_auc, test_new_new_ap))
    test_new_old_acc, test_new_old_ap, test_new_old_f1, test_new_old_auc = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_label_new_old_l, test_e_idx_new_old_l)
    print('Test statistics: {} new-old nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_old_acc, test_new_old_auc, test_new_old_ap))

# save model
# logger.info('Saving CAWN model ...')
# torch.save(cawn.state_dict(), best_model_path)
# logger.info('CAWN model saved')

# save one line result
save_oneline_result('log/', args, [test_acc, test_auc, test_ap, test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc])
# save walk_encodings_scores
# checkpoint_dir = '/'.join(cawn.get_checkpoint_path(0).split('/')[:-1])
# cawn.save_walk_encodings_scores(checkpoint_dir)
