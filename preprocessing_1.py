#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 1: Preprocess and save filtered normalized PPI matrix.
"""

import click
import numpy as np
import scipy.sparse as ssp
from tqdm import trange
from logzero import logger


def get_norm_net_mat(net_mat):
    degree_0 = np.asarray(net_mat.sum(0)).squeeze()
    mat_d_0 = ssp.diags(degree_0 ** -0.5, format='csr')
    degree_1 = np.asarray(net_mat.sum(1)).squeeze()
    mat_d_1 = ssp.diags(degree_1 ** -0.5, format='csr')
    return mat_d_0 @ net_mat @ mat_d_1


@click.command()
@click.argument('ppi_net_mat_path', type=click.Path(exists=True))
@click.argument('filtered_mat_output_path', type=click.Path())
@click.argument('top', type=click.INT, default=100, required=False)
def main(ppi_net_mat_path, filtered_mat_output_path, top):
    ppi_net_mat = (mat_ := ssp.load_npz(ppi_net_mat_path)) + ssp.eye(mat_.shape[0], format='csr')
    logger.info(f'Original: {ppi_net_mat.shape}, nnz={ppi_net_mat.nnz}')

    r, c, v = [], [], []
    for i in trange(ppi_net_mat.shape[0]):
        for v_, c_ in sorted(zip(ppi_net_mat[i].data, ppi_net_mat[i].indices), reverse=True)[:top]:
            r.append(i)
            c.append(c_)
            v.append(v_)

    filtered = ssp.csc_matrix((v, (r, c)), shape=ppi_net_mat.shape).T
    norm_mat = get_norm_net_mat(filtered)
    ssp.save_npz(filtered_mat_output_path, norm_mat)
    logger.info(f'Saved filtered+normalized mat: {filtered_mat_output_path} ({norm_mat.shape}, nnz={norm_mat.nnz})')


if __name__ == '__main__':
    main()
