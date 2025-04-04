import dgl.data
from dgl.data.utils import load_graphs
import dgl
import torch
import time
import psutil
import numpy as np
from model import Msg2Mail, Encoder, Decoder
from dataloader import dataloader, frauder_sampler
from utils import set_logger, EarlyStopMonitor, get_current_ts, get_args, set_random_seeds, GradualWarmupScheduler
# from eval import eval_epoch
from pathlib import Path
import argparse
from preprocess.my_dataloader import Temporal_Dataloader, Temporal_Splitting, Dynamic_Dataloader, data_load
from torch_geometric.data import Data
from typing import Union


def args_process():
    parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
    parser.add_argument('--snapshot', default=20, type=int, help="number of snapshot in tmeporal graph NN")  
    args = parser.parse_args()
    args.new_node_count = True
    return args

def pyg2dgl(graph: Union[Temporal_Dataloader | Data]) -> dgl.data:
    src = torch.tensor(graph.edge_index[0])
    dst = torch.tensor(graph.edge_index[1])
    label = torch.tensor(graph.y, dtype=torch.float32)
    timestamp = graph.edge_attr.type(torch.float32)
    edge_feat = graph.edge_pos.type(torch.float32)

    g = dgl.graph((torch.cat([src,dst]), torch.cat([dst,src])))
    g.ndata['label'] = label # .repeat(2).squeeze()
    src_label = label[src]
    dest_label = label[dst]
    elabel = torch.concat([src_label, dest_label], dim=0)
    g.edata['label'] = elabel
    g.edata['timestamp'] = timestamp.repeat(2).squeeze()
    g.edata['feat'] = edge_feat.repeat(2,1).squeeze()
    return g


def train(args, logger):

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    snapshot = args.snapshot
    dataset = args.data

    graph, idxloader = data_load(dataset=dataset)
    graph_dataloader = Temporal_Splitting(graph=graph).temporal_splitting(time_mode="view", snapshot=snapshot, views = snapshot-2)
    temporaloader = Dynamic_Dataloader(data = graph_dataloader, graph = graph)

    for snum in range(snapshot-2):
        pyg_data: Temporal_Dataloader = temporaloader.get_temporal()
        pygt1: Temporal_Dataloader = temporaloader.get_T1graph(snum)

        # convert from PyG dataloader to dgl
        g = pyg2dgl(pyg_data)
        g1 = pyg2dgl(pygt1)

        print(g)
        efeat_dim = g.edata['feat'].shape[1]
        nfeat_dim = efeat_dim


        train_loader, val_loader, test_loader, num_val_samples, num_test_samples = dataloader(args, g, g1=g1)


        encoder = Encoder(args, nfeat_dim, n_head=args.n_head, dropout=args.dropout).to(device)
        decoder = Decoder(args, nfeat_dim).to(device)
        msg2mail = Msg2Mail(args, nfeat_dim)
        fraud_sampler = frauder_sampler(g)

        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        if args.warmup:
            scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_lr)
            optimizer.zero_grad()
            optimizer.step()
        loss_fcn = torch.nn.BCEWithLogitsLoss()

        loss_fcn = loss_fcn.to(device)

        early_stopper = EarlyStopMonitor(logger=logger, max_round=args.patience, higher_better=True)

        # if args.pretrain:
        #     logger.info(f'Loading the linkpred pretrained attention based encoder model')
        #     encoder.load_state_dict(torch.load(Pretrained_MODEL_PATH+get_pretrain_model_name('Encoder')))

        for epoch in range(args.n_epoch):
            # reset node state
            g.ndata['mail'] = torch.zeros((g.num_nodes(), args.n_mail, nfeat_dim+2), dtype=torch.float32) 
            g.ndata['feat'] = torch.zeros((g.num_nodes(), nfeat_dim), dtype=torch.float32) # init as zero, people can init it using others.
            g.ndata['last_update'] = torch.zeros((g.num_nodes()), dtype=torch.float32) 
            encoder.train()
            decoder.train()
            start_epoch = time.time()
            m_loss = []
            for batch_idx, (input_nodes, pos_graph, neg_graph, blocks, frontier, current_ts) in enumerate(train_loader):
                

                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device) if neg_graph is not None else None
                

                if not args.no_time or not args.no_pos:
                    current_ts, pos_ts, num_pos_nodes = get_current_ts(args, pos_graph, neg_graph)
                    pos_graph.ndata['ts'] = current_ts
                else:
                    current_ts, pos_ts, num_pos_nodes = None, None, None
                
                reverse_graph = dgl.add_reverse_edges(neg_graph) if neg_graph is not None else None
                emb, _ = encoder.forward(dgl.add_reverse_edges(pos_graph), reverse_graph, num_pos_nodes)
                if batch_idx != 0:
                    if 'LP' not in args.tasks and args.balance:
                        neg_graph = fraud_sampler.sample_fraud_event(g, args.bs//5, current_ts.max().cpu()).to(device)
                    logits, labels = decoder.forward(emb, pos_graph, neg_graph)

                    loss = loss_fcn(logits, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    m_loss.append(loss.item())


                # MSG Passing
                with torch.no_grad():
                    mail = msg2mail.gen_mail(args, emb, input_nodes, pos_graph, frontier, 'train')

                    if not args.no_time:
                        g.ndata['last_update'][pos_graph.ndata[dgl.NID][:num_pos_nodes]] = pos_ts.to('cpu')
                    g.ndata['feat'][pos_graph.ndata[dgl.NID]] = emb.to('cpu')
                    g.ndata['mail'][input_nodes] = mail
                if batch_idx % 100 == 1:
                    gpu_mem = torch.cuda.max_memory_allocated() / 1.074e9 if torch.cuda.is_available() and args.gpu >= 0 else 0
                    torch.cuda.empty_cache()
                    mem_perc = psutil.virtual_memory().percent
                    cpu_perc = psutil.cpu_percent(interval=None)
                    output_string = f'Epoch {epoch} | Step {batch_idx}/{len(train_loader)} | CPU {cpu_perc:.1f}% | Sys Mem {mem_perc:.1f}% | GPU Mem {gpu_mem:.4f}GB '
                    
                    output_string += f'| {args.tasks} Loss {np.mean(m_loss):.4f}'

                    logger.info(output_string)

            total_epoch_time = time.time() - start_epoch
            logger.info(' training epoch: {} took {:.4f}s'.format(epoch, total_epoch_time))
            val_ap, val_auc, val_acc, val_loss = eval_epoch(args, logger, g, val_loader, encoder, decoder, msg2mail, loss_fcn, device, num_val_samples)
            logger.info('Val {} Task | ap: {:.4f} | auc: {:.4f} | acc: {:.4f} | Loss: {:.4f}'.format(args.tasks, val_ap, val_auc, val_acc, val_loss))

            if args.warmup:
                scheduler_warmup.step(epoch)
            else:
                scheduler_lr.step()

            early_stopper_metric = val_ap if 'LP' in args.tasks else val_auc

            if early_stopper.early_stop_check(early_stopper_metric):
                logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                # encoder.load_state_dict(torch.load(MODEL_SAVE_PATH+get_model_name('Encoder')))
                # decoder.load_state_dict(torch.load(MODEL_SAVE_PATH+get_model_name('Decoder')))

                test_result = [early_stopper.best_ap, early_stopper.best_auc, early_stopper.best_acc, early_stopper.best_loss]
                break

            test_ap, test_auc, test_acc, test_loss = eval_epoch(args, logger, g, test_loader, encoder, decoder, msg2mail, loss_fcn, device, num_test_samples)
            logger.info('Test {} Task | ap: {:.4f} | auc: {:.4f} | acc: {:.4f} | Loss: {:.4f}'.format(args.tasks, test_ap, test_auc, test_acc, test_loss))
            test_result = [test_ap, test_auc, test_acc, test_loss]

            if early_stopper.best_epoch == epoch: 
                early_stopper.best_ap = test_ap
                early_stopper.best_auc = test_auc
                early_stopper.best_acc = test_acc
                early_stopper.best_loss = test_loss
                logger.info(f'Saving the best model at epoch {early_stopper.best_epoch}')
                # torch.save(encoder.state_dict(), MODEL_SAVE_PATH+get_model_name('Encoder'))
                # torch.save(decoder.state_dict(), MODEL_SAVE_PATH+get_model_name('Decoder'))

    
if __name__ == '__main__':
    args = get_args()
    set_random_seeds(args.seed)

    train(args, None)