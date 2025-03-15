import dgl
import torch
import numpy as np
import random

class TemporalEdgeCollator(dgl.dataloading.EdgeCollator):
    def __init__(self, args, g, eids, graph_sampler, g_sampling=None, exclude=None,
                 reverse_eids=None, reverse_etypes=None, negative_sampler=None, mode='val'):
        super(TemporalEdgeCollator, self).__init__(g, eids, graph_sampler, g_sampling, exclude,
                 reverse_eids, reverse_etypes, negative_sampler)
        
        self.args = args
        self.mode = mode

    def collate(self, items):
        #print('before', self.block_sampler.ts)

        current_ts = self.g.edata['timestamp'][items[-1]]  # only sample edges before last timestamp in a batch
        self.block_sampler.ts = current_ts
        neg_pair_graph = None
        if self.negative_sampler is None:
            input_nodes, pair_graph, blocks = self._collate(items)
        else:
            input_nodes, pair_graph, neg_pair_graph, blocks = self._collate_with_negative_sampling(items)

        for i in range(self.args.n_layer-1):
            self.block_sampler.frontiers[0].add_edges(*self.block_sampler.frontiers[i+1].edges())
        frontier = dgl.reverse(self.block_sampler.frontiers[0])

        return input_nodes, pair_graph, neg_pair_graph, blocks, frontier, current_ts

class MultiLayerTemporalNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, args, fanouts, replace=False, return_eids=False):
        super().__init__(len(fanouts), return_eids)

        self.fanouts = fanouts
        self.replace = replace
        self.ts = 0
        self.args = args
        self.frontiers = [None for _ in range(len(fanouts))]

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]

        g = dgl.in_subgraph(g, seed_nodes)
        g.remove_edges(torch.where(g.edata['timestamp']>self.ts)[0].type(torch.int32))

        if fanout is None:
            frontier = g
            #frontier = dgl.in_subgraph(g, seed_nodes)
        else:
            if self.args.uniform:
                frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
            else:
                frontier = dgl.sampling.select_topk(g, fanout, 'timestamp', seed_nodes)

        self.frontiers[block_id] = frontier
        return frontier
    
    """
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        ""Generate the a list of MFGs given the destination nodes.

        Parameters
        ----------
        g : DGLGraph
            The original graph.
        seed_nodes : Tensor or dict[ntype, Tensor]
            The destination nodes by node type.

            If the graph only has one node type, one can just specify a single tensor
            of node IDs.
        exclude_eids : Tensor or dict[etype, Tensor]
            The edges to exclude from computation dependency.

        Returns
        -------
        list[DGLGraph]
            The MFGs generated for computing the multi-layer GNN output.

        Notes
        -----
        For the concept of frontiers and MFGs, please refer to
        :ref:`User Guide Section 6 <guide-minibatch>` and
        :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
        ""
        blocks = []
        eid_excluder = _create_eid_excluder(exclude_eids, self.output_device)

        # if isinstance(g, DistGraph):
        #     # TODO:(nv-dlasalle) dist graphs may not have an associated graph,
        #     # causing an error when trying to fetch the device, so for now,
        #     # always assume the distributed graph's device is CPU.
        #     graph_device = F.cpu()
        # else:
        #     graph_device = g.device

        for block_id in reversed(range(self.num_layers)):
            seed_nodes_in = seed_nodes
            if isinstance(seed_nodes_in, dict):
                seed_nodes_in = {ntype: nodes.to(graph_device) \
                    for ntype, nodes in seed_nodes_in.items()}
            else:
                seed_nodes_in = seed_nodes_in.to(graph_device)
            frontier = self.sample_frontier(block_id, g, seed_nodes_in)

            if self.output_device is not None:
                frontier = frontier.to(self.output_device)
                if isinstance(seed_nodes, dict):
                    seed_nodes_out = {ntype: nodes.to(self.output_device) \
                        for ntype, nodes in seed_nodes.items()}
                else:
                    seed_nodes_out = seed_nodes.to(self.output_device)
            else:
                seed_nodes_out = seed_nodes

            # Removing edges from the frontier for link prediction training falls
            # into the category of frontier postprocessing
            if eid_excluder is not None:
                eid_excluder(frontier)

            block = transform.to_block(frontier, seed_nodes_out)
            if self.return_eids:
                assign_block_eids(block, frontier)

            seed_nodes = {ntype: block.srcnodes[ntype].data[NID] for ntype in block.srctypes}
            blocks.insert(0, block)
        return blocks
    """
class frauder_sampler():
    def __init__(self, g):
        self.fraud_eid = torch.where(g.edata['label']!=0)[0]
        len_frauder = self.fraud_eid.shape[0] // 2
        self.fraud_eid = self.fraud_eid[:len_frauder]
        self.ts = g.edata['timestamp'][self.fraud_eid]
    def sample_fraud_event(self, g, bs, current_ts):
        idx = (self.ts<current_ts)
        num_fraud = idx.sum().item()
        
        if num_fraud > bs:
            
            idx[random.sample(list(range(num_fraud)), num_fraud-bs)] = False # 只采样一部分fraud event
            
        fraud_eid = self.fraud_eid[idx]
        
        fraud_graph = dgl.edge_subgraph(g, fraud_eid)
        return fraud_graph
