import numpy as np
import networkx as nx


def duplicate_mask(graph):
    # take the graphicle object and create the network
    edge_dict = graph.adj.to_dicts()
    nx_graph = nx.DiGraph(edge_dict["edges"])
    # edge_dict = {edge: i for i, edge in enumerate(nx_graph.edges())}

    # find the duplicates
    duplicates = np.array(
        [
            list(nx_graph.in_edges(node))[0]
            for node in nx_graph.nodes()
            if nx_graph.in_degree(node) == nx_graph.out_degree(node) == 1
        ],
        dtype=[("in", "<i4"), ("out", "<i4")],
    )

    # create a mask
    # edge_dict = {edge: i for i, edge in enumerate(nx_graph.edges())}
    mask = np.ones(graph.adj.edges.shape, bool)
    indices = np.array(
        [np.where(graph.adj.edges == item)[0] for item in duplicates]
    )
    mask[indices] = False

    # update the edges
    for duplo in duplicates:
        # find and update the edge in the graph
        edge_idx = np.where(graph.adj.edges["in"] == duplo[1])[0]
        graph.adj.edges["in"][edge_idx] = duplo[0]
        # check if needed to update the duplicates list
        duplo_idx = np.where(duplicates["in"] == duplo[1])[0]
        if len(duplo_idx) > 0:
            duplicates["in"][duplo_idx] = duplo[0]
    return graph[mask]
