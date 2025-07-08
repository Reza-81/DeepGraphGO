#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/8/25
@author yrh

"""

import numpy as np
import torch
import torch.nn as nn
import dgl
from pathlib import Path
from tqdm import tqdm
from logzero import logger
import dgl.dataloading as dgldl

from deepgraphgo.networks import GcnNet
from deepgraphgo.evaluation import fmax, aupr

__all__ = ['Model']


class Model(object):
    """

    """
    def __init__(self, *, model_path: Path, dgl_graph, network_x, **kwargs):
        self.model = self.network = GcnNet(**kwargs)
        self.dp_network = nn.DataParallel(self.network.cuda())
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = None
        self.dgl_graph, self.network_x, self.batch_size = dgl_graph, network_x, None

    def get_scores(self, blocks, batch_x):
        # blocks: list of dgl.Block, batch_x: input features for the first block
        scores = self.network(blocks, batch_x)
        return scores

    def get_optimizer(self, **kwargs):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **kwargs)

    def train_step(self, blocks, batch_x, batch_y, update, **kwargs):
        self.model.train()
        scores = self.get_scores(blocks, batch_x)
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
        ppi_train_idx = np.full(self.network_x.shape[0], -1, dtype=int)
        ppi_train_idx[train_ppi] = np.arange(train_ppi.shape[0])
        best_fmax = 0.0

        sampler = dgldl.NeighborSampler([self.model.num_gcn] * self.model.num_gcn)
        train_nids = torch.from_numpy(train_ppi).long().cuda()
        train_dataloader = dgldl.DataLoader(
            self.dgl_graph, train_nids, sampler,
            batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

        for epoch_idx in range(epochs_num):
            train_loss = 0.0
            for input_nodes, output_nodes, blocks in tqdm(train_dataloader, desc=f'Epoch {epoch_idx}', leave=False, dynamic_ncols=True):
                # batch_x: features for input_nodes
                batch_x = self.network_x[input_nodes.cpu().numpy()]
                batch_x = (
                    torch.from_numpy(batch_x.indices).cuda().long(),
                    torch.from_numpy(batch_x.indptr).cuda().long(),
                    torch.from_numpy(batch_x.data).cuda().float()
                )
                # batch_y: labels for output_nodes
                batch_y = train_y[ppi_train_idx[output_nodes.cpu().numpy()]].toarray()
                batch_y = torch.from_numpy(batch_y)
                train_loss += self.train_step(blocks, batch_x, batch_y, True)
            best_fmax = self.valid(valid_ppi, valid_y, epoch_idx, train_loss / len(train_ppi), best_fmax)

    def valid(self, valid_loader, targets, epoch_idx, train_loss, best_fmax):
        scores = self.predict(valid_loader, valid=True)
        (fmax_, t_), aupr_ = fmax(targets, scores), aupr(targets.toarray().flatten(), scores.flatten())
        logger.info(F'Epoch {epoch_idx}: Loss: {train_loss:.5f} '
                    F'Fmax: {fmax_:.3f} {t_:.2f} AUPR: {aupr_:.3f}')
        if fmax_ > best_fmax:
            best_fmax = fmax_
            self.save_model()
        return best_fmax

    @torch.no_grad()
    def predict_step(self, blocks, batch_x):
        self.model.eval()
        return torch.sigmoid(self.get_scores(blocks, batch_x)).cpu().numpy()

    def predict(self, test_ppi, batch_size=None, valid=False, **kwargs):
        if batch_size is None:
            batch_size = self.batch_size
        if not valid:
            self.load_model()
        sampler = dgldl.NeighborSampler([self.model.num_gcn] * self.model.num_gcn)
        test_nids = torch.from_numpy(np.unique(test_ppi)).long().cuda()
        test_dataloader = dgldl.DataLoader(
            self.dgl_graph, test_nids, sampler,
            batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
        mapping = {x: i for i, x in enumerate(np.unique(test_ppi))}
        test_ppi_idx = np.asarray([mapping[x] for x in test_ppi])
        scores_list = []
        for input_nodes, output_nodes, blocks in test_dataloader:
            batch_x = self.network_x[input_nodes.cpu().numpy()]
            batch_x = (
                torch.from_numpy(batch_x.indices).cuda().long(),
                torch.from_numpy(batch_x.indptr).cuda().long(),
                torch.from_numpy(batch_x.data).cuda().float()
            )
            batch_scores = self.predict_step(blocks, batch_x)
            scores_list.append(batch_scores)
        scores = np.vstack(scores_list)
        return scores[test_ppi_idx]

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
