# --- Part 2: Load nx_graph and create dgl_graph ---
import pickle
import dgl
import dgl.data

# Load the graph
with open('nx_graph.pkl', 'rb') as f:
    nx_graph = pickle.load(f)

# Convert to DGL graph
dgl_graph = dgl.from_networkx(nx_graph, edge_attrs=['ppi'])

# Optional: Save to file
dgl_graph_path = 'ppi_dgl_top_100_test_memory'
dgl.data.utils.save_graphs(dgl_graph_path, dgl_graph)

# Validate
top = 100  # Make sure this matches Cell 1
assert dgl_graph.in_degrees().max() <= top

print("DGL graph created and saved.")