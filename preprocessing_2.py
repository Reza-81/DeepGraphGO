#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 2: Load processed PPI matrix and build DGL graph.
"""

import click
import numpy as np
import scipy.sparse as ssp
import networkx as nx
import dgl
import dgl.data
from tqdm import tqdm
from logzero import logger
import gc
import time


@click.command()
@click.argument('filtered_mat_input_path', type=click.Path(exists=True))
@click.argument('dgl_graph_output_path', type=click.Path())
@click.argument('top', type=click.INT, default=100, required=False)
def main(filtered_mat_input_path, dgl_graph_output_path, top):
    norm_mat = ssp.load_npz(filtered_mat_input_path)
    logger.info(f'Loaded normalized matrix: {norm_mat.shape}, nnz={norm_mat.nnz}')

    ppi_net_mat_coo = ssp.coo_matrix(norm_mat)
    nx_graph = nx.DiGraph()
    for u, v, d in tqdm(zip(ppi_net_mat_coo.row, ppi_net_mat_coo.col, ppi_net_mat_coo.data),
                        total=ppi_net_mat_coo.nnz, desc='PPI'):
        nx_graph.add_edge(u, v, ppi=d)

    gc.collect()
    time.sleep(2)  # Optional: reduce or remove if not needed
    dgl_graph = dgl.from_networkx(nx_graph, edge_attrs=['ppi'])

    assert dgl_graph.in_degrees().max() <= top
    dgl.data.utils.save_graphs(dgl_graph_output_path, dgl_graph)
    logger.info(f'DGL graph saved to {dgl_graph_output_path}')


if __name__ == '__main__':
    main()
