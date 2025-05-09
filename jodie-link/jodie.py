'''
This code trains the JODIE model for the given dataset. 
The task is: interaction prediction.

How to run: 
$ python jodie.py --network reddit --model jodie --epochs 50

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

import time

from library_data import *
import library_models as lib
from library_models import *
from my_dataloader import Temporal_Dataloader, Dynamic_Dataloader, Temporal_Splitting, data_load, to_cuda
from torch_geometric.transforms import RandomNodeSplit
from evaluate_node_classification import Simple_Regression
from torch import Tensor
from time_evaluation import TimeRecord
import argparse
from library_data import split_edges_for_link_prediction
from library_models import eval_one_epoch
from library_models import RandEdgeSampler

def get_time_diffs(graph: Temporal_Dataloader) -> Tensor:
    freq_dict, length = {}, graph.edge_index.shape[1]
    src_sequence_id = graph.edge_index[0]
    dest_sequence_id = graph.edge_index[1]
    timestamp_list = graph.edge_attr.cpu()
    for nidx in range(length):
        u = src_sequence_id[nidx]
        i = dest_sequence_id[nidx]
        freq_dict[u] = nidx
        freq_dict[i] = nidx
    freq_dict: dict[int, int] = dict(sorted(freq_dict.items(), key=lambda x: x[0]))
    time_idx = list(freq_dict.values())
    train_time_diffs = timestamp_list[time_idx]
    train_time_diffs -= torch.full(train_time_diffs.size(), torch.min(train_time_diffs))
    return train_time_diffs.type(torch.float32)

recsd = TimeRecord(model_name="Jodie")
recsd_score = TimeRecord(model_name="Jodie")
# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', required=False, default="mooc", help='Name of the network/dataset')
parser.add_argument('--model', default="jodie", help='Model name to save output in file')
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=20, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=64, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument('--state_change', default=True, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.')
parser.add_argument('--snapshot', default=60, type=int, help='used for decide how many snapshots there would be')
args = parser.parse_args()

if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# SET GPU
if args.gpu == -1:
    args.gpu = select_free_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# SET BATCHING TIMESPAN
'''
Timespan is the frequency at which the batches are created and the JODIE model is trained. 
As the data arrives in a temporal order, the interactions within a timespan are added into batches (using the T-batch algorithm). 
The batches are then used to train JODIE. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
Longer timespan leads to less frequent model updates. 
'''
epoch_interval = 1
recsd.get_dataset(args.network)
recsd_score.get_dataset(args.network)
recsd.set_up_logger()
recsd_score.set_up_logger(name="time_logger")
recsd.record_start()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graph, idxloader = data_load(dataset=args.network)
graph_dataloader = Temporal_Splitting(graph=graph).temporal_splitting(time_mode="view", snapshot = args.snapshot, views = args.snapshot-2)
temporaloader = Dynamic_Dataloader(data = graph_dataloader, graph=graph)

num_features = graph.pos[1].shape[1]

model = JODIE(args, num_features)
snapshot_list = []
for snapshot in range(args.snapshot):
    recsd.temporal_record()
    data = temporaloader.get_temporal()
    t1_data = temporaloader.get_T1graph(snapshot)
    num_users = data.src_num
    num_items = data.dest_num

    num_interactions = data.num_edges
    unique_y, y_freq = np.unique(data.y, return_counts=True)
    y_distro = [y_freq[i] / data.y.shape[0] for i in range(y_freq.shape[0])]
    y_distro = [0.5, 0.5]
    y_distro_cuda = torch.tensor(y_distro, dtype=torch.float32).to(device)
    crossEntropyLoss = nn.CrossEntropyLoss(weight=y_distro_cuda)
    MSELoss = nn.MSELoss()

    timespan = data.edge_attr[-1].item() - data.edge_attr[0].item()
    tbatch = timespan / 500

    train, valid, valid_nn, test, test_nn = split_edges_for_link_prediction(data, t1_data)

    all_node = data.num_nodes
    model.reset_prediction(all_node, all_node)
    model = model.to(device)

    val_rand_sampler = RandEdgeSampler(valid["user_sequence_id"], valid["item_sequence_id"])
    valnn_rand_sampler = RandEdgeSampler(valid_nn["user_sequence_id"], valid_nn["item_sequence_id"])
    test_rand_sampler = RandEdgeSampler(test["user_sequence_id"], test["item_sequence_id"])
    testnn_rand_sampler = RandEdgeSampler(test_nn["user_sequence_id"], test_nn["item_sequence_id"])

    user_embeddings = torch.tensor(data.node_pos).to(device)
    item_embeddings = torch.tensor(data.node_pos).to(device)
    item_embedding_static = Variable(torch.eye(all_node).cuda()) # one-hot vectors for static embeddings
    user_embedding_static = Variable(torch.eye(all_node).cuda()) # one-hot vectors for static embeddings 

    # INITIALIZE MODEL
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    user_sequence_id = np.copy(train["user_sequence_id"])
    item_sequence_id = np.copy(train["item_sequence_id"])
    feature_sequence = train["feature_sequence"]
    user_timediffs_sequence = train["user_timediffs_sequence"]
    item_timediffs_sequence = train["item_timediffs_sequence"]
    user_previous_itemid_sequence = train["user_previous_itemid_seq"]
    y_true = train["y_true"]
    timestamp_sequence = train["timestamp_sequence"]

    # RUN THE JODIE MODEL
    print("*** Training the JODIE model for %d epochs ***" % args.epochs)

    # variables to help using tbatch cache between epochs
    is_first_epoch = True
    cached_tbatches_user = {}
    cached_tbatches_item = {}
    cached_tbatches_interactionids = {}
    cached_tbatches_feature = {}
    cached_tbatches_user_timediffs = {}
    cached_tbatches_item_timediffs = {}
    cached_tbatches_previous_item = {}

    train_reg: torch.nn.modules = None
    val_reg: torch.nn.modules = None
    test_reg: torch.nn.modules = None

    train_end_idx = train["item_sequence_id"].shape[0]
    score_record: list[dict[str, float]] = []
    for ep in range(args.epochs):
        recsd.epoch_record()
        print('Epoch %d of %d' % (ep, args.epochs))

        epoch_start_time = time.time()
        # INITIALIZE EMBEDDING TRAJECTORY STORAGE
        user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
        item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())

        optimizer.zero_grad()
        reinitialize_tbatches()
        total_loss, loss, total_interaction_count = 0, 0, 0

        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False
        
        j_interval = 0
        # TRAIN TILL THE END OF TRAINING INTERACTION IDX
        for j in range(train_end_idx):

            if is_first_epoch:
                # READ INTERACTION J
                userid = user_sequence_id[j]
                itemid = item_sequence_id[j]
                feature = torch.tensor(feature_sequence[j], dtype=torch.float32).to(device)
                user_timediff = user_timediffs_sequence[j]
                item_timediff = item_timediffs_sequence[j]

                # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
                tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
                lib.tbatchid_user[userid] = tbatch_to_insert
                lib.tbatchid_item[itemid] = tbatch_to_insert

                lib.current_tbatches_user[tbatch_to_insert].append(userid)
                lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                lib.current_tbatches_interactionids[tbatch_to_insert].append(j) # edge id
                lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

            timestamp = timestamp_sequence[j].item()
            if tbatch_start_time is None:
                tbatch_start_time = timestamp

            # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
            if (j - j_interval) > tbatch:
                j_interval = j # RESET START TIME FOR THE NEXT TBATCHES

                # ITERATE OVER ALL T-BATCHES
                if not is_first_epoch:
                    lib.current_tbatches_user = cached_tbatches_user[timestamp]
                    lib.current_tbatches_item = cached_tbatches_item[timestamp]
                    lib.current_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                    lib.current_tbatches_feature = cached_tbatches_feature[timestamp]
                    lib.current_tbatches_user_timediffs = cached_tbatches_user_timediffs[timestamp]
                    lib.current_tbatches_item_timediffs = cached_tbatches_item_timediffs[timestamp]
                    lib.current_tbatches_previous_item = cached_tbatches_previous_item[timestamp]


                for i in range(len(lib.current_tbatches_user)):
                    # print('Processed %d of %d T-batches ' % (i, len(lib.current_tbatches_user)))
                    
                    total_interaction_count += len(lib.current_tbatches_interactionids[i])

                    # LOAD THE CURRENT TBATCH
                    if is_first_epoch:
                        lib.current_tbatches_user[i] = torch.LongTensor(lib.current_tbatches_user[i]).cuda()
                        lib.current_tbatches_item[i] = torch.LongTensor(lib.current_tbatches_item[i]).cuda()
                        lib.current_tbatches_interactionids[i] = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                        temp_features = torch.concat(lib.current_tbatches_feature[i], dim=0).cuda()
                        val = temp_features.size(0)
                        lib.current_tbatches_feature[i] = temp_features.reshape(val//num_features, num_features)


                        lib.current_tbatches_user_timediffs[i] = torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()
                        lib.current_tbatches_item_timediffs[i] = torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()
                        lib.current_tbatches_previous_item[i] = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()

                    tbatch_userids = lib.current_tbatches_user[i] # Recall "lib.current_tbatches_user[i]" has unique elements
                    tbatch_itemids = lib.current_tbatches_item[i] # Recall "lib.current_tbatches_item[i]" has unique elements
                    tbatch_interactionids = lib.current_tbatches_interactionids[i]
                    feature_tensor = Variable(lib.current_tbatches_feature[i]) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                    user_timediffs_tensor = Variable(lib.current_tbatches_user_timediffs[i]).unsqueeze(1)
                    item_timediffs_tensor = Variable(lib.current_tbatches_item_timediffs[i]).unsqueeze(1)
                    tbatch_itemids_previous = lib.current_tbatches_previous_item[i]
                    item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]

                    # PROJECT USER EMBEDDING TO CURRENT TIME
                    user_embedding_input = user_embeddings[tbatch_userids,:]
                    user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                    user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous,:], user_embedding_static[tbatch_userids,:]], dim=1)

                    # PREDICT NEXT ITEM EMBEDDING                            
                    predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                    # CALCULATE PREDICTION LOSS
                    item_embedding_input = item_embeddings[tbatch_itemids,:]
                    loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]], dim=1).detach())

                    # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                    user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                    item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

                    item_embeddings[tbatch_itemids,:] = item_embedding_output
                    user_embeddings[tbatch_userids,:] = user_embedding_output  

                    user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                    item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output

                    # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                    loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                    loss += MSELoss(user_embedding_output, user_embedding_input.detach())

                    # CALCULATE STATE CHANGE LOSS
                    if args.state_change:
                        loss += calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_timeseries, y_true, crossEntropyLoss) 

                # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # RESET LOSS FOR NEXT T-BATCH
                loss = 0
                item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                user_embeddings.detach_()
                item_embeddings_timeseries.detach_() 
                user_embeddings_timeseries.detach_()
            
                # REINITIALIZE
                if is_first_epoch:
                    cached_tbatches_user[timestamp] = lib.current_tbatches_user
                    cached_tbatches_item[timestamp] = lib.current_tbatches_item
                    cached_tbatches_interactionids[timestamp] = lib.current_tbatches_interactionids
                    cached_tbatches_feature[timestamp] = lib.current_tbatches_feature
                    cached_tbatches_user_timediffs[timestamp] = lib.current_tbatches_user_timediffs
                    cached_tbatches_item_timediffs[timestamp] = lib.current_tbatches_item_timediffs
                    cached_tbatches_previous_item[timestamp] = lib.current_tbatches_previous_item
                    
                    reinitialize_tbatches()
                    tbatch_to_insert = -1
        
        recsd.epoch_end(batch_size=train_end_idx)
        is_first_epoch = False # as first epoch ends here
        print("Last epoch took {} minutes".format((time.time()-epoch_start_time)/60))
        # END OF ONE EPOCH 
        print("\n\nTotal loss in this epoch = %f" % (total_loss))
        # item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
        # user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)

        # 别save model了，直接开测，node prediction任务测试比link prediction更直接一些
        if (ep+1) % epoch_interval == 0:
            model.eval()
            # Binary validation procedure
            val_acc, val_f1, val_roc, val_ap = eval_one_epoch(dataset=valid, model=model, sampler=val_rand_sampler, device = device, user_emb=user_embeddings, item_emb = item_embeddings)
            nn_val_acc, nn_val_f1, nn_val_roc, nn_val_ap = eval_one_epoch(dataset=valid_nn, model=model, sampler=valnn_rand_sampler, device=device, user_emb=user_embeddings, item_emb=item_embeddings)
            
            t1_src_emb, t1_item_emb = torch.tensor(t1_data.edge_pos).to(device), torch.tensor(t1_data.edge_pos).to(device)
            nn_test_acc, nn_test_f1, nn_test_roc, nn_test_ap = eval_one_epoch(dataset=test_nn, model=model, sampler=test_rand_sampler, device=device, user_emb=t1_src_emb, item_emb=t1_item_emb)
            test_acc, test_f1, test_roc, test_ap = eval_one_epoch(dataset=test, model=model, sampler=testnn_rand_sampler, device=device, user_emb=t1_src_emb, item_emb=t1_item_emb)

            score_record.append({
                "epoch": ep + 1,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_roc": val_roc,
                "val_ap": val_ap,
                "nn_val_acc": nn_val_acc,
                "nn_val_f1": nn_val_f1,
                "nn_val_roc": nn_val_roc,
                "nn_val_ap": nn_val_ap,
                "nn_test_acc": nn_test_acc,
                "nn_test_f1": nn_test_f1,
                "nn_test_roc": nn_test_roc,
                "nn_test_ap": nn_test_ap,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_roc": test_roc,
                "test_ap": test_ap
            })
    
    # record in of data and metrics
    temporaloader.update_event(timestamp=snapshot)
    recsd.temporal_end(data.num_nodes)
    recsd.score_record(temporal_score_=score_record, node_size=data.num_nodes, temporal_idx=snapshot, epoch_interval=epoch_interval)
    snapshot_list.append(score_record)
    
recsd.record_end()
recsd_score.record_end()
recsd_score.fast_processing(snapshot_list)


