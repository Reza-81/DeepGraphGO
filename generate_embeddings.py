```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Node2Vec embeddings from the filtered PPI matrix (sparse npz) without NetworkX.
Uses on-the-fly random walks with a single-pass Word2Vec to reduce RAM usage.
Includes progress tracking for Word2Vec training and embedding extraction.
"""

import click
import numpy as np
import scipy.sparse as ssp
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import random
from pathlib import Path
from ruamel.yaml import YAML
from logzero import logger
from tqdm import tqdm

try:
    from numba import njit
    HAS_NUMBA = True
    logger.info("Numba detected; using JIT for faster walk simulation.")
except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba not installed; walks will be slower. Install with 'pip install numba'.")

@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-f', '--filtered-npz', type=click.Path(exists=True), required=True, 
              help='Path to filtered normalized PPI matrix npz (from preprocessing_1.py).')
@click.option('--dim', type=int, default=128, help='Embedding dimension.')
@click.option('--walk-length', type=int, default=80, help='Length of each random walk.')
@click.option('--num-walks', type=int, default=10, help='Number of walks per node.')
@click.option('--p', type=float, default=1.0, help='Node2Vec p parameter.')
@click.option('--q', type=float, default=1.0, help='Node2Vec q parameter.')
def main(data_cnf, filtered_npz, dim, walk_length, num_walks, p, q):
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
                sum_w = weights.sum()
                probs = weights / sum_w if sum_w > 0 else np.ones(len(weights)) / len(weights)
            else:
                prev = walk[-2]
                prev_start = indptr[prev]
                prev_end = indptr[prev + 1]
                prev_neighbors = set(indices[prev_start:prev_end])
                alpha = np.zeros(len(neighbors), dtype=np.float32)
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
        return [str(node) for node in walk]

    if HAS_NUMBA:
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
                cum_probs = np.cumsum(probs)
                r = random.random() * cum_probs[-1]
                next_idx = np.searchsorted(cum_probs, r)
                walk[step] = neighbors[next_idx]
                walk_len += 1
            return walk[:walk_len]

        def simulate_walk(start, length, p, q):
            walk = simulate_walk_jit(indptr, indices, data, start, length, p, q)
            return [str(node) for node in walk]

    # Collect walks into a list for single-pass training
    logger.info(f'Generating {num_walks} walks of length {walk_length} per node')
    walks = []
    for start in tqdm(range(num_nodes), desc="Generating walks"):
        for _ in range(num_walks):
            walks.append(simulate_walk(start, walk_length, p, q))

    # Define callback for Word2Vec training progress
    class ProgressCallback(CallbackAny2Vec):
        def __init__(self, total_sentences):
            self.total_sentences = total_sentences
            self.epoch = 0
            self.batch_count = 0
            self.progress_bar = None

        def on_epoch_begin(self, model):
            logger.info(f'Starting Word2Vec epoch {self.epoch + 1}')
            self.progress_bar = tqdm(total=self.total_sentences, desc=f'Word2Vec Epoch {self.epoch + 1}')

        def on_epoch_end(self, model):
            self.progress_bar.close()
            self.epoch += 1
            self.batch_count = 0

        def on_batch_end(self, model):
            self.batch_count += model.batch_words
            self.progress_bar.update(model.batch_words)

    logger.info(f'Generated {len(walks)} walks')
    logger.info(f'Generating embeddings with Word2Vec (dim={dim}, walk_length={walk_length}, num_walks={num_walks})')
    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=10,
        min_count=1,
        sg=1,
        workers=4,
        epochs=1,
        callbacks=[ProgressCallback(len(walks) * dim)]
    )

    logger.info('Extracting embeddings')
    embeddings = np.zeros((num_nodes, dim), dtype=np.float32)
    for i in tqdm(range(num_nodes), desc="Extracting embeddings"):
        node_str = str(i)
        if node_str in model.wv:
            embeddings[i] = model.wv[node_str]
        else:
            logger.warning(f'No embedding for node {i}; setting to zero.')

    emb_path = 'data/ppi_node2vec_embeddings.npy'
    np.save(emb_path, embeddings)
    logger.info(f'Embeddings saved to {emb_path}, shape={embeddings.shape}')

if __name__ == '__main__':
    main()
```