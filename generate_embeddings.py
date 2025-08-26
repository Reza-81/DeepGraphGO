#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Node2Vec embeddings from the filtered PPI matrix (sparse npz).
"""

import click
import numpy as np
import scipy.sparse as ssp
import networkx as nx
from node2vec import Node2Vec
from pathlib import Path
from ruamel.yaml import YAML
from logzero import logger

@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-f', '--filtered-npz', type=click.Path(exists=True), required=True, 
              help='Path to filtered normalized PPI matrix npz (from preprocessing_1.py).')
def main(data_cnf, filtered_npz):
    yaml = YAML(typ='safe')
    data_cnf = yaml.load(Path(data_cnf))
    
    logger.info(f'Loading filtered PPI matrix from {filtered_npz}')
    ppi_mat = ssp.load_npz(filtered_npz).tocsr()
    num_nodes = ppi_mat.shape[0]
    logger.info(f'Matrix loaded: shape={ppi_mat.shape}, nnz={ppi_mat.nnz}')
    
    logger.info('Converting to NetworkX graph')
    nx_G = nx.from_scipy_sparse_array(ppi_mat, edge_attribute='ppi').to_undirected()
    logger.info(f'NetworkX graph created: nodes={nx_G.number_of_nodes()}, edges={nx_G.number_of_edges()}')
    
    logger.info('Generating Node2Vec embeddings')
    node2vec = Node2Vec(nx_G, dimensions=128, walk_length=80, num_walks=10, p=1, q=1, 
                        weight_key='ppi', workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    logger.info('Extracting embeddings')
    embeddings = np.zeros((num_nodes, 128), dtype=np.float32)
    for i in range(num_nodes):
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