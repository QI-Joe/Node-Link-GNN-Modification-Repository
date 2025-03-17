import torch
import numpy as np
from tqdm import tqdm
import math
from eval import *
from module import CAWN
from torch import Tensor


def train_val(train_val_data: tuple[tuple[np.ndarray]], model: CAWN, mode, bs, epochs, \
              criterion: torch.nn.CrossEntropyLoss, optimizer: Adam, \
              early_stopper, ngh_finders, rand_samplers, logger):
    # unpack the data, prepare for the training
    train_data, val_data = train_val_data
    
    train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = train_data
    val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = val_data
    
    train_rand_sampler, val_rand_sampler = rand_samplers
    partial_ngh_finder, full_ngh_finder = ngh_finders

    if mode == 't':  # transductive
        model.update_ngh_finder(full_ngh_finder)
    elif mode == 'i':  # inductive
        model.update_ngh_finder(partial_ngh_finder)
    else:
        raise ValueError('training mode {} not found.'.format(mode))
    device = model.n_feat_th.data.device
    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / bs)
    print('num of training instances: {}'.format(num_instance))
    print('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    for epoch in range(epochs):
        trian_acc_src, train_acc_dst, m_loss = [], [], []
        
        np.random.shuffle(idx_list)  # shuffle the training samples for every epoch
        print('start {} epoch'.format(epoch))
        for k in range(num_batch):
            # generate training mini-batch
            s_idx = k * bs
            e_idx = min(num_instance - 1, s_idx + bs)
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
            model.train()
            src_emb, dst_emb = model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut)   # the core training code
            
            src_pred, dst_pred = model.projection(src_emb), model.projection(dst_emb)
            loss: Tensor = criterion(src_pred, src_label) + criterion(dst_pred, dst_label)
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())

            # collect training results
            if (k+1) % (num_batch//2)==0 & (epoch+1) % 5 ==0:
                with torch.no_grad():
                    model.eval()
                    pred_src = src_pred.detach().argmax(dim=-1).cpu().numpy()
                    pred_dest = dst_pred.detach().argmax(dim=-1).cpu().numpy()

                    src_label, dst_label = src_label.cpu().numpy(), dst_label.cpu().numpy()

                    src_train_acc = accuracy_score(src_label, pred_src)
                    dst_train_acc = accuracy_score(dst_label, pred_dest)

                    trian_acc_src.append(src_train_acc)
                    train_acc_dst.append(dst_train_acc)
                    print(f"Epoch {epoch} - Batch {k}: Source Accuracy: {src_train_acc:.4f}, Destination Accuracy: {dst_train_acc:.4f}, mean loss {np.mean(m_loss):.4f}")

        # validation phase use all information
        if (epoch+1) % 2 ==0:
            val_src, val_dst = eval_one_epoch(10, model, val_rand_sampler, val_src_l,
                                                            val_dst_l, val_ts_l, val_label_l, val_e_idx_l)
            print('epoch: {}:'.format(epoch))
            print('epoch mean loss: {:.4f}'.format(np.mean(m_loss)))
            print('Src train acc: {:.4f}, Src val acc: {:.4f}'.format(np.mean(trian_acc_src), val_src["accuracy"]))
            print('Dst train acc: {:.4f}, Dst val acc: {:.4f}'.format(np.mean(train_acc_dst), val_dst["accuracy"]))
            print('Src val ap: {:.4f}, Dst val ap: {:.4f}'.format(val_src["prec"], val_dst["prec"]))


            val_ap = (val_src["prec"]+val_dst["prec"]) / 2
            # early stop check and checkpoint saving
            if early_stopper.early_stop_check(val_ap):
                print('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                print(f'Loading the best model at epoch {early_stopper.best_epoch}')
                print(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                break
            import sys
            sys.exit()


