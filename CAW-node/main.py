import pandas as pd
from log import *
from eval import *
from utils import *
from train import *
#import numba
from module import CAWN
from graph import NeighborFinder
import resource
from my_dataloader import Temporal_Dataloader, Temporal_Splitting, Dynamic_Dataloader, data_load, t2t1_node_alignment
import copy
from time_evaluation import TimeRecord

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

rscore.get_dataset(DATA)
rpresent.get_dataset(DATA)
rscore.set_up_logger(name="time_logger")
rpresent.set_up_logger()
rpresent.record_start()

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

    test_src_l, test_dst_l, test_ts_l, test_e_idx_l = t1_temporal.edge_index[0], \
    t1_temporal.edge_index[1], t1_temporal.edge_attr, np.arange(t1_temporal.edge_index.shape[1])
    test_label_l = (t1_temporal.y[t1_temporal.edge_index[0]], t1_temporal.y[t1_temporal.edge_index[1]])
    
    # test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], label_l[valid_test_flag]
    
    if args.mode == 'i':
        # need to be modified
        test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l, test_label_new_new_l = src_l[valid_test_new_new_flag], dst_l[valid_test_new_new_flag], ts_l[valid_test_new_new_flag], e_idx_l[valid_test_new_new_flag], label_l[valid_test_new_new_flag]
        test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l, test_label_new_old_l = src_l[valid_test_new_old_flag], dst_l[valid_test_new_old_flag], ts_l[valid_test_new_old_flag], e_idx_l[valid_test_new_old_flag], label_l[valid_test_new_old_flag]
    
    # train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
    # val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
    # train_val_data = (train_data, val_data)

    # test_data = test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l

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
    test_rand_sampler = RandEdgeSampler((test_src_l, ), (test_dst_l, ))
    # rand_samplers = train_rand_sampler, val_rand_sampler

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
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

    # start train and val phases
    # srecord = train_val(train_val_data=train_val_data, model=cawn, mode=args.mode, bs=BATCH_SIZE, epochs=NUM_EPOCH, criterion=criterion, \
    #           optimizer=optimizer, early_stopper=early_stopper, num_cls=num_cls, ngh_finders=ngh_finders, rand_samplers=rand_samplers, logger=rpresent)
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
        trian_acc_src, train_acc_dst, m_loss = [], [], []
        
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
            src_label = torch.from_numpy(train_label_l[0][batch_idx]).to(device)  # currently useless since we are not predicting edge labels
            dst_label = torch.from_numpy(train_label_l[1][batch_idx]).to(device)
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

            # feed in the data and learn from error
            optimizer.zero_grad()
            cawn.train()
            src_emb, dst_emb = cawn.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut)   # the core training code
            
            src_pred, dst_pred = cawn.projection(src_emb), cawn.projection(dst_emb)
            loss: Tensor = criterion(src_pred, src_label) + criterion(dst_pred, dst_label)
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())

            # collect training results
            if (k+1) % (num_batch//2)==0 & (epoch+1) % epoch_tester ==0:
                with torch.no_grad():
                    cawn.eval()
                    pred_src = src_pred.detach().argmax(dim=-1).cpu().numpy()
                    pred_dest = dst_pred.detach().argmax(dim=-1).cpu().numpy()

                    src_label, dst_label = src_label.cpu().numpy(), dst_label.cpu().numpy()

                    src_train_acc = accuracy_score(src_label, pred_src)
                    dst_train_acc = accuracy_score(dst_label, pred_dest)

                    trian_acc_src.append(src_train_acc)
                    train_acc_dst.append(dst_train_acc)
                    print(f"Epoch {epoch} - Batch {k}: Source Accuracy: {src_train_acc:.4f}, Destination Accuracy: {dst_train_acc:.4f}, mean loss {np.mean(m_loss):.4f}")
        
        rpresent.epoch_end(BATCH_SIZE)
        # validation phase use all information
        if (epoch+1) % epoch_tester ==0:
            val_src, val_dst = eval_one_epoch(num_cls, cawn, val_rand_sampler, val_src_l,
                                                            val_dst_l, val_ts_l, val_label_l, val_e_idx_l)
            print('epoch: {}:'.format(epoch))
            print('epoch mean loss: {:.4f}'.format(np.mean(m_loss)))
            print('Src train acc: {:.4f}, Src val acc: {:.4f}'.format(np.mean(trian_acc_src), val_src["accuracy"]))
            print('Dst train acc: {:.4f}, Dst val acc: {:.4f}'.format(np.mean(train_acc_dst), val_dst["accuracy"]))
            print('Src val ap: {:.4f}, Dst val ap: {:.4f}'.format(val_src["prec"], val_dst["prec"]))


            val_ap = (val_src["prec"]+val_dst["prec"]) / 2
            # final testing
            cawn.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
            test_src, test_dst = eval_one_epoch(num_cls, cawn, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l)
            
            test_src["train_acc"], test_dst["train_acc"] = np.mean(trian_acc_src), np.mean(train_acc_dst)
            test_src["val_acc"], test_dst["val_acc"] = val_src["accuracy"], val_dst["accuracy"]
            
            test_acc = (test_src["accuracy"] + test_dst["accuracy"]) / 2
            test_ap = (test_src["precision"] + test_dst["precision"]) / 2
            test_recall = (test_src["recall"] + test_dst["recall"]) / 2
            print('Test statistics: {} all nodes -- acc: {:.4f}, prec: {:.4f}, recall: {:.4f}'.format(args.mode, test_acc, test_ap, test_recall))            
            score_record.extend([test_src, test_dst])

            # early stop check and checkpoint saving
            if early_stopper.early_stop_check(val_ap):
                print('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                print(f'Loading the best model at epoch {early_stopper.best_epoch}')
                print(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                break

        temporaloader.update_event(sp)
        rpresent.score_record(temporal_score_=score_record, node_size=temporalgraph.num_nodes, temporal_idx=sp, epoch_interval=epoch_tester)
        snapshot_list.append(score_record)

rpresent.record_end()
rscore.record_end()
rscore.fast_processing(snapshot_list)

# save one line result
# save_oneline_result('log/', args, [test_acc, test_auc, test_ap, test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc])
# save walk_encodings_scores
# checkpoint_dir = '/'.join(cawn.get_checkpoint_path(0).split('/')[:-1])
# cawn.save_walk_encodings_scores(checkpoint_dir)
