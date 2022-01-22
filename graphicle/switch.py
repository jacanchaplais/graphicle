class ntool:
    def switch_edgenode(G):
        import networkx as _nx

        edge_dict = {edge: i for i, edge in enumerate(G.edges())}

        L = _nx.line_graph(G)
        L = _nx.relabel.relabel_nodes(L, edge_dict)

        return L

    def remove_duplicates(G):
        # I find the duplicate as the nodes with in_degree=out_degree=1
        duplicates = [
            node
            for node in G.nodes()
            if G.in_degree(node) == G.out_degree(node) == 1
        ]

        for duplo in duplicates:
            vertex_in = list(G.in_edges(duplo))[0][0]
            vertex_out = list(G.out_edges(duplo))[0][1]

            # remove the node and the edges
            G.remove_node(duplo)

            # add the new edge
            G.add_edge(vertex_in, vertex_out)

        return G
