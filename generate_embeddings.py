#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Node2Vec embeddings from the PPI graph.
"""

import click
import numpy as np
import dgl
import networkx as nx
from node2vec import Node2Vec
from pathlib import Path
from ruamel.yaml import YAML
from logzero import logger

@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
def main(data_cnf):
    yaml = YAML(typ='safe')
    data_cnf = yaml.load(Path(data_cnf))
    dgl_path = data_cnf['network']['dgl']
    logger.info(f'Loading DGL graph from {dgl_path}')
    dgl_graph = dgl.data.utils.load_graphs(dgl_path)[0][0]
    
    logger.info('Converting to NetworkX graph')
    nx_G = dgl_graph.to_networkx(edge_attrs=['ppi']).to_undirected()  # Node2Vec works on undirected
    
    logger.info('Generating Node2Vec embeddings')
    node2vec = Node2Vec(nx_G, dimensions=128, walk_length=80, num_walks=10, p=1, q=1, 
                        weight_key='ppi', workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    num_nodes = dgl_graph.number_of_nodes()
    embeddings = np.zeros((num_nodes, 128))
    for i in range(num_nodes):
        embeddings[i] = model.wv[str(i)]  # Node2Vec uses str keys internally
    
    emb_path = 'data/ppi_node2vec_embeddings.npy'
    np.save(emb_path, embeddings)
    logger.info(f'Embeddings saved to {emb_path}')

if __name__ == '__main__':
    main()