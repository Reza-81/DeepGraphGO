#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Script 2: Load sparse matrix directly as COO and build DGL graph using low-level API.
"""

import click
import numpy as np
import scipy.sparse as ssp
import dgl
import torch
from logzero import logger


@click.command()
@click.argument('filtered_mat_input_path', type=click.Path(exists=True))
@click.argument('dgl_graph_output_path', type=click.Path())
@click.argument('top', type=click.INT, default=100, required=False)
def main(filtered_mat_input_path, dgl_graph_output_path, top):
    logger.info(f"Loading filtered PPI matrix from: {filtered_mat_input_path}")

    # ✅ Load COO directly to avoid memory duplication
    mat = ssp.load_npz(filtered_mat_input_path).tocoo()

    logger.info(f"Matrix loaded: shape={mat.shape}, nnz={mat.nnz}")

    # ✅ Use DGL low-level API (faster and more memory-efficient than networkx)
    src = torch.tensor(mat.row, dtype=torch.int64)
    dst = torch.tensor(mat.col, dtype=torch.int64)
    edge_attr = torch.tensor(mat.data, dtype=torch.float32)

    g = dgl.graph((src, dst), num_nodes=mat.shape[0])
    g.edata['ppi'] = edge_attr

    # Optional check
    assert g.in_degrees().max().item() <= top, "Max in-degree exceeds top."

    # Save graph
    dgl.data.utils.save_graphs(dgl_graph_output_path, g)
    logger.info(f"DGL graph saved to: {dgl_graph_output_path}")


if __name__ == '__main__':
    main()
