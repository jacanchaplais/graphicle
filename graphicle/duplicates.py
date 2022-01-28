import numpy as np
import networkx as nx
import copy


def duplicate_mask2(graph):
    graph_masked = copy.deepcopy(graph)
    in_deg = np.unique(
        graph_masked.adj.edges["in"], return_index=True, return_counts=True
    )
    out_deg = np.unique(
        graph_masked.adj.edges["out"], return_index=True, return_counts=True
    )

    mask = np.ones(graph_masked.adj.edges.shape, bool)
    duplicates = []
    for idx in range(len(in_deg[0])):
        if in_deg[2][idx] == 1:
            _out_idx = np.where(out_deg[0] == in_deg[0][idx])[0]
            if len(_out_idx) > 0 and out_deg[2][_out_idx] == 1:
                duplicates.append(graph_masked.adj.edges[out_deg[1][idx]])
                mask[out_deg[1][idx]] = False
    duplicates = np.array(duplicates, dtype=[("in", "<i4"), ("out", "<i4")])
    # update the edges
    for duplo in duplicates:
        # find and update the edge in the graph_masked
        edge_idx = np.where(graph_masked.adj.edges["in"] == duplo[1])[0]
        graph_masked.adj.edges["in"][edge_idx] = duplo[0]
        # check if needed to update the duplicates list
        duplo_idx = np.where(duplicates["in"] == duplo[1])[0]
        if len(duplo_idx) > 0:
            duplicates["in"][duplo_idx] = duplo[0]
    return graph_masked[mask]
