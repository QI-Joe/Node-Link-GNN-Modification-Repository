from dgl.data import DGLDataset
from torch.utils.data import DataLoader
import dgl
from dataloader import MultiLayerTemporalNeighborSampler, TemporalEdgeCollator
import torch

def dataloader(args, g: DGLDataset, g1: DGLDataset):
    origin_num_edges = g.num_edges() // 2
    test_num_edges = g1.num_edges() // 2

    train_eid = torch.arange(0, int(0.7 * origin_num_edges), dtype=torch.int32)
    val_eid = torch.arange(int(0.7 * origin_num_edges), origin_num_edges, dtype=torch.int32)
    test_eid = torch.arange(0, test_num_edges, dtype=torch.int32)

    # reverse_eids = torch.cat([torch.arange(origin_num_edges, 2 * origin_num_edges), torch.arange(0, origin_num_edges)])
    exclude, reverse_eids = None, None

    # will be set as None
    negative_sampler = dgl.dataloading.negative_sampler.Uniform(1) if 'LP' in args.tasks else None

    fanouts = [args.n_degree for _ in range(args.n_layer)]
    sampler = MultiLayerTemporalNeighborSampler(args, fanouts, return_eids=False)
    # sampler = dgl.dataloading.as_edge_prediction_sampler(sampler)
    train_collator = TemporalEdgeCollator(args, g, train_eid, sampler, exclude=exclude, reverse_eids=reverse_eids, negative_sampler=negative_sampler, mode='train')
    
    train_loader = DataLoader(
                        train_collator.dataset, collate_fn=train_collator.collate,
                        batch_size=args.bs, shuffle=False, drop_last=False, num_workers=args.n_worker)
    val_collator = TemporalEdgeCollator(args, g, val_eid, sampler, exclude=exclude, reverse_eids=reverse_eids, negative_sampler=negative_sampler)
    val_loader = DataLoader(
                        val_collator.dataset, collate_fn=val_collator.collate,
                        batch_size=args.bs, shuffle=False, drop_last=False, num_workers=args.n_worker)

    test_collator = TemporalEdgeCollator(args, g1, test_eid, sampler, exclude=exclude, reverse_eids=reverse_eids, negative_sampler=negative_sampler)
    test_loader = DataLoader(
                        test_collator.dataset, collate_fn=test_collator.collate,
                        batch_size=args.bs, shuffle=False, drop_last=False, num_workers=args.n_worker)
    return train_loader, val_loader, test_loader, val_eid.shape[0], test_eid.shape[0]
