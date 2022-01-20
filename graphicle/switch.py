class switch:
    def switch_edgenode(G):
        import networkx as _nx

        edge_dict = {edge: i for i, edge in enumerate(G.edges())}

        L = _nx.line_graph(G)
        L = _nx.relabel.relabel_nodes(L, edge_dict)

        return L
