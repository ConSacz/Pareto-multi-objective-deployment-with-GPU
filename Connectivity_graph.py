import networkx as nx

def Connectivity_graph(G, bat_ex):
    G_sub = G.copy()
    #G_sub.remove_nodes_from(bat_ex)
    number_nodes = G.number_of_nodes() - len(bat_ex)
    reachable_nodes = list(nx.dfs_preorder_nodes(G_sub, source=0))

    if len(reachable_nodes) == number_nodes:
        return 1  # connected
    else:
        return 0  # not connected
