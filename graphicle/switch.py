import numpy as np


def switch_edgenode(G):
    import networkx as _nx

    edge_dict = {edge: i for i, edge in enumerate(G.edges())}

    L = _nx.line_graph(G)
    L = _nx.relabel.relabel_nodes(L, edge_dict)

    return L


"""
def remove_duplicates(graph:Graphicle) -> Graphicle:
    nx_graph = graph.edges.to_networkx()
    # I find the duplicate as the nodes with in_degree=out_degree=1
    duplicates = [
        node
        for node in nx_graph.nodes()
        if nx_graph.in_degree(node) == nx_graph.out_degree(node) == 1
    ]

    for duplo in duplicates:
        vertex_in = list(G.in_edges(duplo))[0][0]
        vertex_out = list(G.out_edges(duplo))[0][1]

        # remove the node and the edges
        G.remove_node(duplo)

        # add the new edge
        G.add_edge(vertex_in, vertex_out)

    return G
"""


def duplicate_mask(graph):
    nx_graph = graph.edges.to_networkx()
    edge_dict = {edge: i for i, edge in enumerate(nx_graph.edges())}

    duplicates = [
        tuple(nx_graph.in_edges(node))[0]
        for node in nx_graph.nodes()
        if nx_graph.in_degree(node) == nx_graph.out_degree(node) == 1
    ]

    mask = np.ones(graph.edges.edges.shape, bool)

    indices = np.array([edge_dict[item] for item in duplicates])
    mask[indices] = False

    return mask
