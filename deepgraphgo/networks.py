#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020/8/25
@author yrh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from logzero import logger

__all__ = ['FeedForwardNet']


class FeedForwardNet(nn.Module):
    """
    """
    def __init__(self, labels_num, input_size, hidden_size, dropout=0.5):
        super(FeedForwardNet, self).__init__()
        logger.info(f'FeedForwardNet: labels_num={labels_num}, input_size={input_size}, hidden_size={hidden_size}, dropout={dropout}')
        self.labels_num = labels_num
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, labels_num)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return self.output(h)