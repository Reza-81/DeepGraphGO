#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Node2Vec embeddings from the filtered PPI matrix (sparse npz).
Avoids NetworkX for memory efficiency.
"""

import click
import numpy as np
import scipy.sparse as ssp
from gensim.models import Word2Vec
import random
from pathlib import Path
from ruamel.yaml import YAML
from logzero import logger

try:
    from numba import njit
    HAS_NUMBA = True
    logger.info("Numba detected; using JIT for faster walk simulation.")
except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba not installed; walks will be slower. Install with 'pip install numba'.")

@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-f', '--filtered-npz', type=click.Path(exists=True), required=True, help='Path to filtered normalized PPI matrix npz (from preprocessing_1.py).')
def main(data_cnf, filtered_npz):
    yaml = YAML(typ='safe')
    data_cnf = yaml.load(Path(data_cnf))

    logger.info(f'Loading filtered PPI matrix from {filtered_npz}')
    ppi_mat = ssp.load_npz(filtered_npz).tocsr()
    num_nodes = ppi_mat.shape[0]
    logger.info(f'Matrix loaded: shape={ppi_mat.shape}, nnz={ppi_mat.nnz}')

    indptr = ppi_mat.indptr
    indices = ppi_mat.indices
    data = ppi_mat.data

    # Simulate biased random walk (Node2Vec style)
    def simulate_walk(start, length, p, q):
        walk = [start]
        for _ in range(length - 1):
            cur = walk[-1]
            cur_start = indptr[cur]
            cur_end = indptr[cur + 1]
            if cur_start == cur_end:
                break  # No neighbors
            neighbors = indices[cur_start:cur_end]
            weights = data[cur_start:cur_end]
            if len(walk) == 1:
                # Unbiased first step
                sum_w = weights.sum()
                probs = weights / sum_w if sum_w > 0 else np.ones(len(weights)) / len(weights)
            else:
                prev = walk[-2]
                prev_start = indptr[prev]
                prev_end = indptr[prev + 1]
                prev_neighbors = set(indices[prev_start:prev_end])
                alpha = np.zeros(len(neighbors))
                for i, w in enumerate(neighbors):
                    if w == prev:
                        alpha[i] = 1.0 / p
                    elif w in prev_neighbors:
                        alpha[i] = 1.0
                    else:
                        alpha[i] = 1.0 / q
                adj_weights = alpha * weights
                sum_w = adj_weights.sum()
                probs = adj_weights / sum_w if sum_w > 0 else np.ones(len(adj_weights)) / len(adj_weights)
            next_node = random.choices(neighbors, weights=probs, k=1)[0]
            walk.append(next_node)
        return walk

    if HAS_NUMBA:
        # JIT version for speed
        @njit
        def simulate_walk_jit(indptr, indices, data, start, length, p, q):
            walk = np.empty(length, dtype=np.int32)
            walk[0] = start
            walk_len = 1
            for step in range(1, length):
                cur = walk[step - 1]
                cur_start = indptr[cur]
                cur_end = indptr[cur + 1]
                if cur_start == cur_end:
                    break
                neighbors = indices[cur_start:cur_end]
                weights = data[cur_start:cur_end]
                if step == 1:
                    sum_w = np.sum(weights)
                    probs = weights / sum_w if sum_w > 0 else np.ones(len(weights)) / len(weights)
                else:
                    prev = walk[step - 2]
                    prev_start = indptr[prev]
                    prev_end = indptr[prev + 1]
                    alpha = np.zeros(len(neighbors))
                    for i in range(len(neighbors)):
                        w = neighbors[i]
                        if w == prev:
                            alpha[i] = 1.0 / p
                        else:
                            # Check if w in prev neighbors (numba doesn't support sets well, so loop check)
                            is_neighbor = False
                            for j in range(prev_start, prev_end):
                                if indices[j] == w:
                                    is_neighbor = True
                                    break
                            if is_neighbor:
                                alpha[i] = 1.0
                            else:
                                alpha[i] = 1.0 / q
                    adj_weights = alpha * weights
                    sum_w = np.sum(adj_weights)
                    probs = adj_weights / sum_w if sum_w > 0 else np.ones(len(adj_weights)) / len(adj_weights)
                # Numba random choice approximation (cumsum)
                cum_probs = np.cumsum(probs)
                r = random.random() * cum_probs[-1]
                next_idx = np.searchsorted(cum_probs, r)
                walk[step] = neighbors[next_idx]
                walk_len += 1
            return walk[:walk_len]

        simulate_walk = simulate_walk_jit  # Override with JIT

    # Generator for walks (yields str for Word2Vec keys)
    def walk_generator(num_walks, walk_length, p, q):
        for start in range(num_nodes):
            for _ in range(num_walks):
                walk = simulate_walk(indptr, indices, data, start, walk_length, p, q)
                yield [str(node) for node in walk]

    # Fit Word2Vec on generator
    logger.info('Generating embeddings with Word2Vec')
    model = Word2Vec(sentences=walk_generator(num_walks=10, walk_length=80, p=1.0, q=1.0),
                     vector_size=128, window=10, min_count=1, sg=1, workers=4, epochs=1)

    # Extract embeddings
    embeddings = np.zeros((num_nodes, 128), dtype=np.float32)
    for i in range(num_nodes):
        if str(i) in model.wv:
            embeddings[i] = model.wv[str(i)]
        else:
            logger.warning(f'No embedding for node {i}; setting to zero.')

    emb_path = 'data/ppi_node2vec_embeddings.npy'
    np.save(emb_path, embeddings)
    logger.info(f'Embeddings saved to {emb_path}')

if __name__ == '__main__':
    main()