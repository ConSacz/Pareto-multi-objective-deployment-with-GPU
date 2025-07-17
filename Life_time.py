import numpy as np
import networkx as nx
# %%
def Life_Time(G):
    """
    Calculate the normalized network lifetime as a fitness function.
    Parameters:
        G (networkx.Graph): weighted graph with edge weights as distances.
    Returns:
        lifetime_normalized (float): inverse-normalized lifetime (lower is better).
    """
    N = G.number_of_nodes()
    
    # Parameters
    b = 0.1   # nJ/bit/m^a
    a = 2     # path loss exponent
    EM = 0    # nJ/bit maintain/process
    ET = 20   # nJ/bit transmit
    ER = 2    # nJ/bit receive
    maxBat = 1000
    # %%
    Bat = np.zeros(N)

    for j in range(1, N):  # from node 1 to N-1 (Python uses 0-indexing)
        try:
            path = nx.shortest_path(G, source=0, target=j, weight='weight')
        except nx.NetworkXNoPath:
            continue  # skip if no path exists
        for i in range(len(path)):
            if i == 0:
                continue  # do nothing for the source node
            elif i == len(path) - 1:
                dt = G[path[i]][path[i - 1]]['weight']
                Bat[path[i]] += ((N+1) * EM + ET + b * dt ** a)
            else:
                dt = G[path[i]][path[i - 1]]['weight']
                dr = G[path[i]][path[i + 1]]['weight']
                Bat[path[i]] += (ER + ET + b * dt ** a + b * dr ** a)


    # %%
    if np.max(Bat) == 0:
        lifetime = np.inf
    else:
        lifetime = maxBat / np.max(Bat)

    # Normalize (same logic as MATLAB version)
    lifetime_normalized = round(1 / lifetime, 5) if lifetime != 0 else 0

    return lifetime_normalized

# %% OLD FUNCS
    # Bat = np.zeros(N)
    # processed = []
    # for j in range(1, N):  # from node 1 to N-1 (Python uses 0-indexing)
    #     try:
    #         path = nx.shortest_path(G, source=j, target=0, weight='weight')
    #     except nx.NetworkXNoPath:
    #         continue  # skip if no path exists
            
    #     for i in range(len(path)):
    #         if i == len(path) - 1:                          # target node (sink node)
    #             continue  
    #         elif i == 0:                                    # source node (node j)
    #             dt = G[path[i]][path[i + 1]]['weight']
    #             Bat[path[i]] += (N+1)*(EM)                  # maintain and process energy
    #             Bat[path[i]] += (ET + b * dt ** a)          # transmit energy
    #         else:                                           # alternative nodes
    #             dt = G[path[i]][path[i + 1]]['weight']
    #             dr = G[path[i]][path[i - 1]]['weight']
    #             if i in processed:
    #                 # add energy
    #                 Bat[path[i]] += (ER + b * dr ** a)          # receive energy
    #                 Bat[path[i]] += (ET + b * dt ** a)          # transmit energy
    #             else:
    #                 # process total energy
    #                 Bat[path[i]] += (N+1)*(EM)                  # maintain energy
    #                 Bat[path[i]] += i*(ER + b * dr ** a)        # receive energy
    #                 Bat[path[i]] += (i+1)*(ET + b * dt ** a)    # transmit energy
    #         processed.append(i)