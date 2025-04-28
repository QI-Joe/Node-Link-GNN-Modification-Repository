from log import *
from eval import *
from utils import *
from module import CAWN
from graph import NeighborFinder
import resource
from my_dataloader import Temporal_Splitting, Dynamic_Dataloader, data_load, t2t1_node_alignment
from sklearn.metrics import roc_auc_score
import copy
from time_evaluation import TimeRecord
from torch import Tensor

args, sys_argv = get_args()
rscore, rpresent = TimeRecord(model_name="CAW"), TimeRecord(model_name="CAW")

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
epoch_tester: int = 10
assert(CPU_CORES >= -1)
set_random_seed(SEED)

# Load data and sanity check
graph, idxloader = data_load(dataset=DATA)
idx_list = Temporal_Splitting(graph=graph).temporal_splitting(time_mode="view", snapshot = SNAPSHOT, views = VIEW)
temporaloader = Dynamic_Dataloader(idx_list, graph)
snapshot_list = list()

edge_dim = graph.pos[1].shape[1]
time_dim = node_dim = graph.pos[0].shape[1]

rscore.get_dataset(DATA)
rpresent.get_dataset(DATA)
rscore.set_up_logger(name="time_logger")
rpresent.set_up_logger()
rpresent.record_start()

num_cls = graph.y.max()+1

cawn = CAWN(agg=AGG, node_dim=node_dim, edge_dim=edge_dim,
            num_layers=NUM_LAYER, use_time=USE_TIME, attn_agg_method=ATTN_AGG_METHOD, attn_mode=ATTN_MODE,
            n_head=ATTN_NUM_HEADS, drop_out=DROP_OUT, pos_dim=POS_DIM, pos_enc=POS_ENC, num_classes = num_cls,
            num_neighbors=NUM_NEIGHBORS, walk_n_head=WALK_N_HEAD, walk_mutual=WALK_MUTUAL, walk_linear_out=args.walk_linear_out, walk_pool=args.walk_pool,
            cpu_cores=CPU_CORES, verbosity=VERBOSITY, get_checkpoint_path=None)

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
    
    max_idx = max(src_l.max(), dst_l.max()) + 1
    assert(np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx or (not math.isclose(1, args.data_usage)))  # all nodes except node 0 should appear and be compactly indexed
    assert(n_feat.shape[0] == max_idx or (not math.isclose(1, args.data_usage)))  # the nodes need to map one-to-one to the node feat matrix

    # split and pack the data by generating valid train/val/test mask according to the "mode"
    val_time = int(temporalgraph.edge_attr.shape[0]*0.85)

    t1_new_new_mask = None
    if args.mode == 't':
        print('Transductive training...')
        valid_train_flag = e_idx_l<val_time
        valid_val_flag = e_idx_l > val_time
        valid_test_flag = np.ones(t1_temporal.edge_attr.shape)

    else:
        assert(args.mode == 'i'), f"mode input is wrong, expect i, t get {args.mode}"
        print('Inductive training...')
        # pick some nodes to mask (i.e. reserved for testing) for inductive setting
        total_node_set = set(np.unique(temporalgraph.edge_index))
        num_total_unique_nodes = len(total_node_set)
        mask_node_set = set(random.sample(sorted(set(src_l[e_idx_l > val_time]).union(set(dst_l[e_idx_l > val_time]))), int(0.1 * num_total_unique_nodes)))
        mask_node_set = list(mask_node_set)
        
        # align the node from t to t1
        t1_new_new_mask, t1_pure_old_mask, total_num_nodes = t2t1_node_alignment(mask_node_set, temporalgraph, t1_temporal)

        mask_src_flag = np.isin(src_l, mask_node_set).astype(np.int8)
        mask_dst_flag = np.isin(dst_l, mask_node_set).astype(np.int8)

        none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
        valid_train_flag = (e_idx_l <= val_time) * (none_mask_node_flag > 0.5)
        valid_val_flag = (e_idx_l > val_time) * (none_mask_node_flag > 0.5)  # both train and val edges can not contain any masked nodes
        
        t1_new_mask = (t1_pure_old_mask < 0.5)  # test edges must contain at least one masked node
        
        # new_flag: node intersection in mask_src_flag and mask_dst_flag 
        valid_test_new_old_flag = (t1_new_mask.astype(int) - t1_new_new_mask.astype(int)).astype(bool)
        # old_flag: node union - node intersection in mask_src_flag and mask_dst_flag
        print('Sampled {} nodes (10 %) and all new nodes {} which are masked in training and reserved for testing'.format(len(mask_node_set), total_num_nodes))
        tforder, ttf = np.unique(t1_new_mask, return_counts=True)
        t1tf = np.unique(t1_new_new_mask, return_counts=True)[1]
        print('All the edges pending in order of {} for validation are {} new-new pure inducitve and {} new-old inductive learning'.format(tforder, ttf, t1tf))

    # split data according to the mask
    src_label, edge_label = label_l[src_l], label_l[dst_l]
    train_src_l, train_dst_l, train_ts_l, train_e_idx_l = src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag]
    train_label_l = (src_label[valid_train_flag], edge_label[valid_train_flag])
    
    val_src_l, val_dst_l, val_ts_l, val_e_idx_l = src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag]
    val_label_l = (src_label[valid_val_flag], edge_label[valid_val_flag])

    test_src_l, test_dst_l, test_ts_l, test_e_idx_l = t1_temporal.edge_index[0], \
    t1_temporal.edge_index[1], t1_temporal.edge_attr, np.arange(t1_temporal.edge_index.shape[1])
    test_label_l = (t1_temporal.y[t1_temporal.edge_index[0]], t1_temporal.y[t1_temporal.edge_index[1]])
    test_n_feat, test_e_feat = t1_temporal.node_pos, t1_temporal.edge_pos

    if args.mode == 'i':
        test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l, test_label_new_new_l = test_src_l[t1_new_new_mask], test_dst_l[t1_new_new_mask], test_ts_l[t1_new_new_mask], test_e_idx_l[t1_new_new_mask], test_label_l[t1_new_new_mask]
        test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l, test_label_new_old_l = test_src_l[valid_test_new_old_flag], test_dst_l[valid_test_new_old_flag], test_ts_l[valid_test_new_old_flag], test_e_idx_l[valid_test_new_old_flag], test_label_l[valid_test_new_old_flag]

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
    
    max_test_idx = max(test_src_l.max(), test_dst_l.max()) + 1
    full_test_adj = [[] for _ in range(max_test_idx+1)]
    for esrc, edst, eidx, ets in zip(test_src_l, test_dst_l, test_e_idx_l, test_ts_l):
        full_test_adj[esrc].append((edst, eidx,ets))
        full_test_adj[edst].append((esrc, eidx, ets))
    full_test_ngh_finder = NeighborFinder(full_test_adj, bias = args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)

    ngh_finders = partial_ngh_finder, full_ngh_finder

    # create random samplers to generate train/val/test instances
    train_rand_sampler = RandEdgeSampler((train_src_l, ), (train_dst_l, ))
    val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l))
    test_rand_sampler = RandEdgeSampler((test_src_l,), (test_dst_l,))

    # multiprocessing memory setting
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (200*args.bs, rlimit[1]))

    # model initialization
    device = torch.device('cuda:{}'.format(GPU))

    
    cawn.temproal_update(ngh_finder=full_ngh_finder, n_feat=n_feat, e_feat=e_feat)
    cawn.to(device)
    optimizer = torch.optim.Adam(cawn.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

    mode = args.mode
    if mode == 't':  # transductive
        cawn.update_ngh_finder(full_ngh_finder)
    elif mode == 'i':  # inductive
        cawn.update_ngh_finder(partial_ngh_finder)
    else:
        raise ValueError('training mode {} not found.'.format(mode))
    
    device = cawn.n_feat_th.data.device
    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / BATCH_SIZE)
    print('num of training instances: {}'.format(num_instance))
    print('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    
    score_record = list()
    for epoch in range(NUM_EPOCH):
        rpresent.epoch_record()
        acc, ap, m_loss, auc = [], [], [], []
        
        np.random.shuffle(idx_list)  # shuffle the training samples for every epoch
        print('start {} epoch'.format(epoch))
        for k in range(num_batch):
            # generate training mini-batch
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            src_l_cut, dst_l_cut = train_src_l[batch_idx], train_dst_l[batch_idx]
            ts_l_cut = train_ts_l[batch_idx]
            e_l_cut = train_e_idx_l[batch_idx]
            # label_l_cut = train_label_l[batch_idx]  # currently useless since we are not predicting edge labels
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

            # feed in the data and learn from error
            optimizer.zero_grad()
            cawn.train()
            pos_prob, neg_prob = cawn.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut)   # the core training code
            pos_label = torch.ones(size, dtype=torch.float, device=device, requires_grad=False)
            neg_label = torch.zeros(size, dtype=torch.float, device=device, requires_grad=False)
            loss: Tensor = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
            loss.backward()
            optimizer.step()

            # collect training results
            if (k+1) % (num_batch//2)==0 & (epoch+1) % epoch_tester ==0:
                with torch.no_grad():
                    cawn.eval()
                    pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
                    pred_label = pred_score > 0.5
                    true_label = np.concatenate([np.ones(size), np.zeros(size)])
                    acc.append((pred_label == true_label).mean())
                    ap.append(average_precision_score(true_label, pred_score))
                    # f1.append(f1_score(true_label, pred_label))
                    m_loss.append(loss.item())
                    auc.append(roc_auc_score(true_label, pred_score))
        
        rpresent.epoch_end(BATCH_SIZE)
        if (epoch+1) % epoch_tester == 0:
            # validation phase use all information
            val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for {} nodes'.format(mode), cawn, val_rand_sampler, val_src_l,
                                                            val_dst_l, val_ts_l, val_label_l, val_e_idx_l)
            print('epoch: {}:'.format(epoch))
            print('epoch mean loss: {}'.format(np.mean(m_loss)))
            print('train acc: {}, val acc: {}'.format(np.mean(acc), val_acc))
            print('train auc: {}, val auc: {}'.format(np.mean(auc), val_auc))
            print('train ap: {}, val ap: {}'.format(np.mean(ap), val_ap))

            # early stop check and checkpoint saving
            if early_stopper.early_stop_check(val_ap):
                print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
                print(f'Loading the best model at epoch {early_stopper.best_epoch}')
                print(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                break

            # final testing
            cawn.test_emb_update(full_test_ngh_finder, test_n_feat, test_e_feat)  # remember that testing phase should always use the full neighbor finder
            test_acc, test_ap, test_f1, test_auc = eval_one_epoch(cawn, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_e_idx_l)
            print('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc, test_ap))
            test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1]*6

            validation_dict = {
                "val_acc": val_acc,
                "val_f1": val_f1,
                "test_acc": test_acc,
                "precision": test_ap,  # Assuming test_ap is used as precision
                "test_roc_auc": test_auc,    # Assuming test_auc is used as recall
                "f1": test_f1  # Assuming test_new_new_ap is used as F1
            }

            if args.mode == 'i':
                test_new_new_acc, test_new_new_ap, test_new_new_f1, test_new_new_auc = eval_one_epoch(cawn, test_rand_sampler, test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l)
                print('Test statistics: {} new-new nodes -- acc: {}, auc: {}, ap: {}'.format(mode, test_new_new_acc, test_new_new_auc,test_new_new_ap ))
                test_new_old_acc, test_new_old_ap, test_new_old_f1, test_new_old_auc = eval_one_epoch(cawn, test_rand_sampler, test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l)
                print('Test statistics: {} new-old nodes -- acc: {}, auc: {}, ap: {}'.format(mode, test_new_old_acc, test_new_old_auc, test_new_old_ap))

                validation_dict = {**validation_dict, **{"test_new_new_acc": test_new_new_acc, "test_new_new_auc": test_new_new_auc, "test_new_new_f1": test_new_new_f1}, **{"test_new_old_acc": test_new_old_acc, "test_new_old_auc": test_new_old_auc, "test_new_old_f1": test_new_old_f1}}
            score_record.append(validation_dict)
            cawn.train_val_emb_restore()

    """
    TODO: Modify time_evaluation.py to fit for new metrics testing result
    """

    temporaloader.update_event(sp)
    rpresent.score_record(temporal_score_=score_record, node_size=temporalgraph.num_nodes, temporal_idx=sp, epoch_interval=epoch_tester, mode=mode)
    snapshot_list.append(score_record)

rpresent.record_end()
rscore.record_end()
rscore.fast_processing(mode, snapshot_list)
