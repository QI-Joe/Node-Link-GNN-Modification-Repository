import math
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from utils import RandEdgeSampler
from typing import Union
from torch.optim import Adam
from module import TGAN

class LogRegression(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LogRegression, self).__init__()
        self.lin = torch.nn.Linear(in_channels, num_classes)
        torch.nn.init.xavier_uniform_(self.lin.weight.data)
        # torch.nn.init.xavier_uniform_(self.lin.weight.data)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        ret = self.lin(x)
        return ret

def Simple_Regression(embedding: torch.Tensor, label: Union[torch.Tensor | np.ndarray], num_classes: int, \
                      num_epochs: int = 1500,  project_model=None, return_model: bool = False) -> tuple[float, float, float, float]:
    
    device = embedding.device
    if not isinstance(label, torch.Tensor):
        label = torch.LongTensor(label).to(device)
    linear_regression = LogRegression(embedding.size(1), num_classes).to(device) if project_model==None else project_model
    f = torch.nn.LogSoftmax(dim=-1)
    optimizer = Adam(linear_regression.parameters(), lr=0.01, weight_decay=1e-4)

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        linear_regression.train()
        optimizer.zero_grad()
        output = linear_regression(embedding)
        loss = loss_fn(f(output), label)

        loss.backward(retain_graph=False)
        optimizer.step()

        # if (epoch+1) % 1000 == 0:
        #     print(f'LogRegression | Epoch {epoch+1}: loss {loss.item():.4f}')

    with torch.no_grad():
        projection = linear_regression(embedding)
        y_true, y_hat = label.cpu().numpy(), projection.argmax(-1).cpu().numpy()
        accuracy, precision, recall, f1 = accuracy_score(y_true, y_hat), \
                                        precision_score(y_true, y_hat, average='macro', zero_division=0), \
                                        recall_score(y_true, y_hat, average='macro'),\
                                        f1_score(y_true, y_hat, average='macro')
        prec_micro, recall_micro, f1_micro = precision_score(y_true, y_hat, average='micro', zero_division=0), \
                                            recall_score(y_true, y_hat, average='micro'),\
                                            f1_score(y_true, y_hat, average='micro')
    if return_model:
        return {"test_acc": accuracy, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, \
            "micro_prec": prec_micro, "micro_recall": recall_micro, "micro_f1": f1_micro}, linear_regression
    
    return {"test_acc": accuracy, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, \
            "micro_prec": prec_micro, "micro_recall": recall_micro, "micro_f1": f1_micro}, None


def dict_merge(d1: dict, d2: dict, k):
    if not d1:
        return d2
    
    for key, val in d2.items():
        d1[key] = (d1[key]*(k-1) + d2[key]) / k
    return d1

src_proj: LogRegression = None
dest_proj: LogRegression = None

def eval_one_epoch(num_classes, tgan: TGAN, sampler: RandEdgeSampler, src, dst, ts, label, num_neighbors: int, interval=None):
    global src_proj, dest_proj
    val_metrics_src, val_metrics_dst = dict(), dict()
    tgan = tgan.eval()
    TEST_BATCH_SIZE = 200
    num_test_instance = len(src)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
    src_lb, dest_lb = label

    interval = num_test_batch//4

    src_collector, dst_collector, label_idx = None, None, 0
    for k in range(num_test_batch):
        s_idx = k * TEST_BATCH_SIZE
        e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
        src_l_cut = src[s_idx:e_idx]
        dst_l_cut = dst[s_idx:e_idx]
        ts_l_cut = ts[s_idx:e_idx]

        if len(src_l_cut)<=5:
            print(s_idx, e_idx)
            continue
        size = len(src_l_cut)
        src_l_fake, dst_l_fake = sampler.sample(size)
        with torch.no_grad():
            src_emb, dest_emb = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, num_neighbors)
            src_emb, dest_emb = src_emb.detach(), dest_emb.detach()

        if src_collector==None and dst_collector==None:
            src_collector = src_emb
            dst_collector = dest_emb
        else:
            src_collector = torch.vstack([src_collector, src_emb])
            dst_collector = torch.vstack([dst_collector, dest_emb])
        label_idx = e_idx
        if interval==0 or (k+1) % interval == 0:
            print(src_collector.shape, label_idx)
            src_metrics, src_proj = Simple_Regression(src_collector, src_lb[:e_idx], num_classes=num_classes, project_model=src_proj, return_model=True)
            dest_metrics, dest_proj = Simple_Regression(dst_collector, dest_lb[:e_idx], num_classes=num_classes, project_model=dest_proj, return_model=True)

            val_metrics_src = dict_merge(val_metrics_src, src_metrics, k)
            val_metrics_dst = dict_merge(val_metrics_dst, dest_metrics, k)
        
    return val_metrics_src, val_metrics_dst