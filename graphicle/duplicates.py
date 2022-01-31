import numpy as np
import networkx as nx
import copy


def duplicate_mask(graph):
    graph_masked = copy.deepcopy(graph)
    in_deg = np.unique(
        graph_masked.adj.edges["in"], return_index=True, return_counts=True
    )
    out_deg = np.unique(
        graph_masked.adj.edges["out"], return_index=True, return_counts=True
    )

    mask = np.ones(graph_masked.adj.edges.shape, bool)
    # indices of in_deg == 1
    in_deg_indices = np.where(in_deg[2] == 1)[0]
    for idx in in_deg_indices:
        _out_idx = np.where(out_deg[0] == in_deg[0][idx])[0]
        # if out_deg is also ==1
        if len(_out_idx) > 0 and out_deg[2][_out_idx] == 1:
            mask[out_deg[1][idx]] = False
            duplo = graph_masked.adj.edges[out_deg[1][idx]]

            edge_idx = np.where(graph_masked.adj.edges["in"] == duplo[1])[0]
            graph_masked.adj.edges["in"][edge_idx] = duplo[0]

    return graph_masked[mask]
