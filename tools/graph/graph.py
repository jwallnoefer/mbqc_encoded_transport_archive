"""Various helper functions to do with graphs and graph states.

"""

import numpy as np
import noisy_graph_states as nsf
import networkx as nx


def construct_2d_cluster_graph(
    shape: tuple, graph_type="networkx", closed=False, offset=0
):
    num_rows = shape[0]
    num_cols = shape[1]
    num_vertices = num_rows * num_cols
    adj = np.zeros((num_vertices, num_vertices), dtype=int)
    # construct horizontal lines
    for r_idx in range(num_rows):
        for c_idx in range(num_cols - 1):
            base_idx = num_cols * r_idx + c_idx
            adj[base_idx, base_idx + 1] = 1
        if closed:
            adj[num_cols * r_idx, num_cols * r_idx + num_cols - 1] = 1
    # construct vertical lines
    for c_idx in range(num_cols):
        for r_idx in range(num_rows - 1):
            base_idx = num_cols * r_idx + c_idx
            adj[base_idx, base_idx + num_cols] = 1
        if closed:
            adj[c_idx, c_idx + (num_rows - 1) * num_cols] = 1
    adj = adj + adj.T

    if graph_type == "graphepp":
        return nsf.libs.graph.graph_from_adj_matrix(adj)
    elif graph_type == "networkx":
        return nx.from_numpy_array(
            adj, nodelist=list(range(offset, offset + num_rows * num_cols))
        )
    else:
        raise ValueError(f"{graph_type=} is not supported.")
