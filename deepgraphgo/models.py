#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020/8/25
@author yrh
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
from logzero import logger
import dgl.dataloading as dgldl

from deepgraphgo.networks import FeedForwardNet
from deepgraphgo.evaluation import fmax, aupr

__all__ = ['Model']


class Model(object):
    """
    """
    def __init__(self, *, model_path: Path, dgl_graph, network_x, embeddings=None, **kwargs):
        self.model = self.network = FeedForwardNet(labels_num=kwargs['labels_num'],
                                                 input_size=network_x.shape[1] + kwargs['embedding_dim'],
                                                 hidden_size=kwargs['hidden_size'],
                                                 dropout=kwargs.get('dropout', 0.5))
        self.dp_network = nn.DataParallel(self.network.cuda())
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = None
        self.dgl_graph, self.network_x, self.batch_size = dgl_graph, network_x, None
        self.embeddings = embeddings.cuda() if embeddings is not None else None

    def get_scores(self, batch_x, input_nodes):
        # Concatenate InterPro features and embeddings
        node_emb = self.embeddings[input_nodes] if self.embeddings is not None else torch.zeros(len(input_nodes), 64).cuda()
        combined_features = torch.cat((batch_x, node_emb), dim=1)
        scores = self.network(combined_features)
        return scores

    def get_optimizer(self, **kwargs):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **kwargs)

    def train_step(self, batch_x, batch_y, input_nodes, update, **kwargs):
        self.model.train()
        scores = self.get_scores(batch_x, input_nodes)
        loss = self.loss_fn(scores, batch_y.cuda())
        loss.backward()
        if update and self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def train(self, train_data, valid_data, loss_params=(), opt_params=(), epochs_num=10, batch_size=40, **kwargs):
        self.get_optimizer(**dict(opt_params))
        self.batch_size = batch_size

        (train_ppi, train_y), (valid_ppi, valid_y) = train_data, valid_data
        best_fmax = 0.0

        # Convert data to tensors and create dataset
        train_features = self.network_x[train_ppi].toarray()
        train_features = torch.from_numpy(train_features).float().cuda()
        train_labels = torch.from_numpy(train_y.toarray()).float()
        train_input_nodes = torch.from_numpy(train_ppi).long()
        train_dataset = TensorDataset(train_features, train_labels, train_input_nodes)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        for epoch_idx in range(epochs_num):
            train_loss = 0.0
            for batch_features, batch_labels, batch_input_nodes in tqdm(train_dataloader, desc=f'Epoch {epoch_idx}', leave=False, dynamic_ncols=True):
                batch_features, batch_labels = batch_features.cuda(), batch_labels.cuda()
                train_loss += self.train_step(batch_features, batch_labels, batch_input_nodes, True)
            best_fmax = self.valid(valid_ppi, valid_y, epoch_idx, train_loss / len(train_ppi), best_fmax)

    def valid(self, valid_ppi, valid_y, epoch_idx, train_loss, best_fmax):
        scores = self.predict(valid_ppi, valid=True)
        (fmax_, t_), aupr_ = fmax(valid_y, scores), aupr(valid_y.toarray().flatten(), scores.flatten())
        logger.info(f'Epoch {epoch_idx}: Loss: {train_loss:.5f} Fmax: {fmax_:.3f} {t_:.2f} AUPR: {aupr_:.3f}')
        if fmax_ > best_fmax:
            best_fmax = fmax_
            self.save_model()
        return best_fmax

    @torch.no_grad()
    def predict_step(self, batch_x, input_nodes):
        self.model.eval()
        return torch.sigmoid(self.get_scores(batch_x, input_nodes)).cpu().numpy()

    def predict(self, test_ppi, batch_size=None, valid=False, **kwargs):
        if batch_size is None:
            batch_size = self.batch_size
        if not valid:
            self.load_model()
        test_features = self.network_x[test_ppi].toarray()
        test_input_nodes = torch.from_numpy(test_ppi).long()
        test_dataset = TensorDataset(torch.from_numpy(test_features).float().cuda(), test_input_nodes)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        scores_list = []
        for batch_features, batch_input_nodes in test_dataloader:
            batch_scores = self.predict_step(batch_features, batch_input_nodes)
            scores_list.append(batch_scores)
        scores = np.vstack(scores_list)
        return scores

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))