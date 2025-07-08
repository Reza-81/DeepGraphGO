# --- Part 1: Compute and save nx_graph ---
import numpy as np
import scipy.sparse as ssp
import networkx as nx
from tqdm import trange, tqdm
import gc
import pickle
import time

# Load and preprocess matrix
ppi_net_mat_path = 'data/ppi_mat.npz'  # <-- Change this
top = 100

mat_ = ssp.load_npz(ppi_net_mat_path)
ppi_net_mat = mat_ + ssp.eye(mat_.shape[0], format='csr')

r, c, v = [], [], []
for i in trange(ppi_net_mat.shape[0]):
    for v_, c_ in sorted(zip(ppi_net_mat[i].data, ppi_net_mat[i].indices), reverse=True)[:top]:
        r.append(i)
        c.append(c_)
        v.append(v_)

def get_norm_net_mat(net_mat):
    degree_0 = np.asarray(net_mat.sum(0)).squeeze()
    mat_d_0 = ssp.diags(degree_0 ** -0.5, format='csr')
    degree_1 = np.asarray(net_mat.sum(1)).squeeze()
    mat_d_1 = ssp.diags(degree_1 ** -0.5, format='csr')
    return mat_d_0 @ net_mat @ mat_d_1

ppi_net_mat = get_norm_net_mat(ssp.csc_matrix((v, (r, c)), shape=ppi_net_mat.shape).T)
ppi_net_mat_coo = ssp.coo_matrix(ppi_net_mat)

# Create NetworkX graph
nx_graph = nx.DiGraph()
for u, v, d in tqdm(zip(ppi_net_mat_coo.row, ppi_net_mat_coo.col, ppi_net_mat_coo.data),
                    total=ppi_net_mat_coo.nnz, desc='PPI'):
    nx_graph.add_edge(u, v, ppi=d)

# Clean up memory
del r, c, v, mat_, ppi_net_mat, ppi_net_mat_coo
gc.collect()

# Save nx_graph to disk
with open('nx_graph.pkl', 'wb') as f:
    pickle.dump(nx_graph, f)
print("Saved nx_graph to disk.")