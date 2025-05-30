#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/25
@author yrh

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from logzero import logger

from deepgraphgo.modules import *

__all__ = ['GcnNet']


class GcnNet(nn.Module):
    """
    """

    def __init__(self, *, labels_num, input_size, hidden_size, num_gcn=0, dropout=0.5, residual=True,
                 **kwargs):
        super(GcnNet, self).__init__()
        logger.info(F'GCN: labels_num={labels_num}, input size={input_size}, hidden_size={hidden_size}, '
                    F'num_gcn={num_gcn}, dropout={dropout}, residual={residual}')
        self.labels_num = labels_num
        self.input = nn.EmbeddingBag(input_size, hidden_size, mode='sum', include_last_offset=True)
        self.input_bias = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.update = nn.ModuleList(NodeUpdate(hidden_size, hidden_size, dropout) for _ in range(num_gcn))
        self.output = nn.Linear(hidden_size, self.labels_num)
        self.residual = residual
        self.num_gcn = num_gcn
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input.weight)
        for update in self.update:
            update.reset_parameters()
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, blocks, inputs):
        # blocks: list of dgl.Block, inputs: tuple for EmbeddingBag
        h = self.dropout(F.relu(self.input(*inputs) + self.input_bias))
        for i, (block, update) in enumerate(zip(blocks, self.update)):
            # Message passing for each block/layer
            # Get source and destination node features
            h_src = h
            # Edge weights for 'ppi' and 'self'
            ppi = block.edata['ppi'].unsqueeze(-1)
            if self.residual:
                self_w = block.edata['self'].unsqueeze(-1)
                m_res = h_src[block.edges()[0]] * self_w
                res = torch.zeros((block.num_dst_nodes(), h_src.shape[1]), device=h_src.device)
                res = res.index_add(0, block.edges()[1], m_res)
            else:
                res = None
            ppi_m_out = h_src[block.edges()[0]] * ppi
            ppi_out = torch.zeros((block.num_dst_nodes(), h_src.shape[1]), device=h_src.device)
            ppi_out = ppi_out.index_add(0, block.edges()[1], ppi_m_out)
            # NodeUpdate expects ppi_out and optionally res
            if res is not None:
                h = update(ppi_out, res)
            else:
                h = update(ppi_out)
        return self.output(h)
