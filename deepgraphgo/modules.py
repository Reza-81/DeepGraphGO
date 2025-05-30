#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/25
@author yrh

"""

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['NodeUpdate']


class NodeUpdate(nn.Module):
    """

    """
    def __init__(self, in_f, out_f, dropout):
        super(NodeUpdate, self).__init__()
        self.ppi_linear = nn.Linear(in_f, out_f)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ppi_out, res=None):
        outputs = self.dropout(F.relu(self.ppi_linear(ppi_out)))
        if res is not None:
            outputs = outputs + res
        return outputs

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ppi_linear.weight)
