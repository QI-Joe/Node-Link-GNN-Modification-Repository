'''
This is a supporting library with the code of the model.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
from collections import defaultdict
import os
import gpustat
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

PATH = "./"

# try:
#     # get_ipython
#     trange = tnrange
#     tqdm = tqdm_notebook
# except NameError:
#     pass

total_reinitialization_count = 0

# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


# THE JODIE MODULE
class JODIE(nn.Module):
    def __init__(self, args, num_features):
        super(JODIE,self).__init__()

        print("*** Initializing the JODIE model ***")
        self.modelname = args.model
        self.embedding_dim = args.embedding_dim
        # self.num_users = num_users
        # self.num_items = num_items
        # self.user_static_embedding_size = num_users
        # self.item_static_embedding_size = num_items

        print("Initializing user and item embeddings")
        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))

        rnn_input_size_items = rnn_input_size_users = self.embedding_dim + 1 + num_features

        print("Initializing user and item RNNs")
        self.item_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)
        self.user_rnn = nn.RNNCell(rnn_input_size_items, self.embedding_dim)

        print("Initializing linear layers")
        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        # self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size + self.embedding_dim * 2, self.item_static_embedding_size + self.embedding_dim)
        self.embedding_layer = NormalLinear(1, self.embedding_dim)
        print("*** JODIE initialization complete ***\n\n")
        
    def reset_prediction(self, num_src: int, num_dest: int):
        """
        TODO: Remember to list why this function is needed
        """
        self.num_users, self.num_items = num_src, num_dest
        self.user_static_embedding_size = num_src
        self.item_static_embedding_size = num_dest
        self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size + self.embedding_dim * 2, self.item_static_embedding_size + self.embedding_dim)
        
    def forward(self, user_embeddings, item_embeddings, timediffs=None, features=None, select=None):
        if select == 'item_update':
            input1 = torch.cat([user_embeddings, timediffs, features], dim=1)
            item_embedding_output = self.item_rnn(input1, item_embeddings)
            return F.normalize(item_embedding_output)

        elif select == 'user_update':
            input2 = torch.cat([item_embeddings, timediffs, features], dim=1)
            user_embedding_output = self.user_rnn(input2, user_embeddings)
            return F.normalize(user_embedding_output)

        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
            #user_projected_embedding = torch.cat([input3, item_embeddings], dim=1)
            return user_projected_embedding

    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings

    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        #x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)

class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

def eval_one_epoch(dataset: dict[np.ndarray], model: JODIE, sampler: RandEdgeSampler, \
    device, user_emb, item_emb):
    """
    """
    BATCH = 200
    # prob_socre_list, prob_label_list = [], []
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    full_length = dataset["user_sequence_id"].shape[0]
    for j in range(BATCH, full_length, BATCH):
        batch_end = min(j, full_length)
        batch_idx = np.arange(j - BATCH, batch_end)
        size = batch_idx.shape[0]
        
        userid = dataset["user_sequence_id"][batch_idx]
        itemid = dataset["item_sequence_id"][batch_idx]
        feature = dataset["feature_sequence"][batch_idx]
        user_timediff = dataset["user_timediffs_sequence"][batch_idx]
        item_timediff = dataset["item_timediffs_sequence"][batch_idx]
        # y_true = dataset["y_true"][j]
        # itemid_previous = dataset["user_previous_itemid_seq"][j]

        src_fake, dst_fake = sampler.sample(size)
        tbatch_userids_fake = torch.LongTensor(src_fake).to(device)
        tbatch_itemids_fake = torch.LongTensor(dst_fake).to(device)
        fake_user_embedding_input = user_emb[tbatch_userids_fake, :]
        fake_item_embedding_input = item_emb[tbatch_itemids_fake, :]

        tbatch_userids = torch.LongTensor(userid).to(device)
        tbatch_itemids = torch.LongTensor(itemid).to(device)
        user_embedding_input = user_emb[tbatch_userids, :]
        item_embedding_input = item_emb[tbatch_itemids, :]
        
        feature_tensor = torch.Tensor(feature).unsqueeze(0).to(device)
        user_timediffs_tensor = torch.Tensor(user_timediff).unsqueeze(1).to(device)
        item_timediffs_tensor = torch.Tensor(item_timediff).unsqueeze(1).to(device)
        # itemid_embedding_previous = item_emb[torch.LongTensor([itemid_previous])]
        
        with torch.no_grad():
            user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='project') 
            fake_user_embeding_output = model.forward(fake_user_embedding_input, fake_item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='project') 
            # item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update') 

        # item_emb[itemid,:] = item_embedding_output.squeeze(0) 
        # user_emb[userid,:] = user_embedding_output.squeeze(0) 

        # eventually = torch.concat([user_embedding_output, fake_user_embeding_output], dim=0)
        eventually = user_embedding_output
        with torch.no_grad():
            prob = model.predict_label(eventually)
        prob_score = prob.cpu().numpy().sum(axis=1)
        pred_label = prob.argmax(dim=1).cpu().numpy()
        
        true_label = dataset["y_true"][batch_idx]
        # true_label = np.concatenate([np.ones(size), np.zeros(size)])
        # prob_socre_list.append(prob_score)
        # prob_label_list.append(pred_label)

        val_acc.append(accuracy_score(true_label, pred_label))
        val_f1.append(f1_score(true_label, pred_label, average='weighted'))
        val_auc.append(roc_auc_score(true_label, prob_score))
        val_ap.append(average_precision_score(true_label, prob_score))

    print(f"Validation ROC AUC: {np.mean(val_auc):.4f}, Average Precision: {np.mean(val_ap):.4f}")
    return np.mean(val_acc), np.mean(val_f1), np.mean(val_auc), np.mean(val_ap)

# INITIALIZE T-BATCH VARIABLES
def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_item
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs, current_tbatches_user_timediffs_next

    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)

    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count +=1


# CALCULATE LOSS FOR THE PREDICTED USER STATE 
def calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_time_series, y_true, loss_function):
    # PREDCIT THE LABEL FROM THE USER DYNAMIC EMBEDDINGS
    prob = model.predict_label(user_embeddings_time_series[tbatch_interactionids,:])
    y = Variable(torch.LongTensor(y_true).cuda()[tbatch_interactionids])
    
    loss = loss_function(prob, y)

    return loss


# SAVE TRAINED MODEL TO DISK
def save_model(model, optimizer, args, epoch, user_embeddings, item_embeddings, train_end_idx, user_embeddings_time_series=None, item_embeddings_time_series=None, path=PATH):
    print("*** Saving embeddings and model ***")
    state = {
            'user_embeddings': user_embeddings.data.cpu().numpy(),
            'item_embeddings': item_embeddings.data.cpu().numpy(),
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'train_end_idx': train_end_idx
            }

    if user_embeddings_time_series is not None:
        state['user_embeddings_time_series'] = user_embeddings_time_series.data.cpu().numpy()
        state['item_embeddings_time_series'] = item_embeddings_time_series.data.cpu().numpy()

    directory = os.path.join(path, 'saved_models/%s' % args.network)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, "checkpoint.%s.ep%d.tp%.1f.pth.tar" % (args.model, epoch, args.train_proportion))
    torch.save(state, filename)
    print("*** Saved embeddings and model to file: %s ***\n\n" % filename)


# LOAD PREVIOUSLY TRAINED AND SAVED MODEL
def load_model(model, optimizer, args, epoch):
    modelname = args.model
    filename = PATH + "saved_models/%s/checkpoint.%s.ep%d.tp%.1f.pth.tar" % (args.network, modelname, epoch, args.train_proportion)
    checkpoint = torch.load(filename)
    print("Loading saved embeddings and model: %s" % filename)
    args.start_epoch = checkpoint['epoch']
    user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']).cuda())
    item_embeddings = Variable(torch.from_numpy(checkpoint['item_embeddings']).cuda())
    try:
        train_end_idx = checkpoint['train_end_idx'] 
    except KeyError:
        train_end_idx = None

    try:
        user_embeddings_time_series = Variable(torch.from_numpy(checkpoint['user_embeddings_time_series']).cuda())
        item_embeddings_time_series = Variable(torch.from_numpy(checkpoint['item_embeddings_time_series']).cuda())
    except:
        user_embeddings_time_series = None
        item_embeddings_time_series = None

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return [model, optimizer, user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, train_end_idx]


# SET USER AND ITEM EMBEDDINGS TO THE END OF THE TRAINING PERIOD 
def set_embeddings_training_end(user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, user_data_id, item_data_id, train_end_idx):
    userid2lastidx = {}
    for cnt, userid in enumerate(user_data_id[:train_end_idx]):
        userid2lastidx[userid] = cnt
    itemid2lastidx = {}
    for cnt, itemid in enumerate(item_data_id[:train_end_idx]):
        itemid2lastidx[itemid] = cnt

    try:
        embedding_dim = user_embeddings_time_series.size(1)
    except:
        embedding_dim = user_embeddings_time_series.shape[1]
    for userid in userid2lastidx:
        user_embeddings[userid, :embedding_dim] = user_embeddings_time_series[userid2lastidx[userid]]
    for itemid in itemid2lastidx:
        item_embeddings[itemid, :embedding_dim] = item_embeddings_time_series[itemid2lastidx[itemid]]

    user_embeddings.detach_()
    item_embeddings.detach_()


# SELECT THE GPU WITH MOST FREE MEMORY TO SCHEDULE JOB 
def select_free_gpu():
    mem = []
    gpus = list(set(range(torch.cuda.device_count()))) # list(set(X)) is done to shuffle the array
    for i in gpus:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        mem.append(gpu_stats.jsonify()["gpus"][i]["memory.used"])
    return str(gpus[np.argmin(mem)])

