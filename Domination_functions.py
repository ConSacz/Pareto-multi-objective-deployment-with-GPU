import numpy as np

# %% WEIGHTED SELECTION
def weighted_selection(f1,f2,w,RP):
    x_sta = RP[:,0].flatten()
    x_nad = RP[:,1].flatten()
    f1 = np.array(f1.flatten())
    f2 = np.array(f2.flatten())
    f1_nmlized = (f1 - x_sta) / (x_nad - x_sta)
    f2_nmlized = (f2 - x_sta) / (x_nad - x_sta)
    ff1 = np.sum(f1_nmlized*w)
    ff2 = np.sum(f2_nmlized*w)
    if ff1 < ff2:
        return 1
    else:
        return 0
    
# %% CHECK DOMINATION
def check_domination(f1, f2):
    """
    Check Pareto domination relationship between two solutions f1 and f2.

    Returns:
        1  if f1 check_domination f2
       -1  if f2 check_domination f1
        0  if non-dominated
        2  if f1 == f2
    """
    f1 = np.array(f1)
    f2 = np.array(f2)

    if np.all(f1 == f2):
        return 2
    elif np.all(f1 <= f2) and np.any(f1 < f2):
        return 1
    elif np.all(f2 <= f1) and np.any(f2 < f1):
        return -1
    else:
        return 0
    
# %% GET PARETO FRONT
def get_pareto_front(non_dom_pop):
    N = len(non_dom_pop)
    is_dominated = np.zeros(N, dtype=bool)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if check_domination(non_dom_pop[j]['Cost'], non_dom_pop[i]['Cost']) == 1:
                is_dominated[i] = True
                break

    # Lấy tất cả các cá thể không bị chi phối
    pareto_front_all = [ind for i, ind in enumerate(non_dom_pop) if not is_dominated[i]]

    # --- Loại bỏ cá thể trùng Cost ---
    costs = np.array([ind['Cost'] for ind in pareto_front_all])
    _, unique_indices = np.unique(costs, axis=0, return_index=True)

    # Duy trì thứ tự xuất hiện (giống 'stable' trong MATLAB)
    unique_indices = sorted(unique_indices)
    pareto_front = [pareto_front_all[i] for i in unique_indices]

    return pareto_front

# %% NON DOMINATED SORTING
def NS_Sort(pop):
    nPop = len(pop)

    for i in range(nPop):
        pop[i]['DominationSet'] = []
        pop[i]['DominatedCount'] = 0

    F = [[]]

    for i in range(nPop):
        for j in range(i + 1, nPop):
            p = pop[i]
            q = pop[j]

            if check_domination(p['Cost'], q['Cost']) == 1:
                p['DominationSet'].append(j)
                q['DominatedCount'] += 1
            elif check_domination(q['Cost'], p['Cost']) == 1:
                q['DominationSet'].append(i)
                p['DominatedCount'] += 1

            pop[i] = p
            pop[j] = q

        if pop[i]['DominatedCount'] == 0:
            F[0].append(i)
            pop[i]['Rank'] = 1

    k = 0
    while True:
        Q = []
        for i in F[k]:
            p = pop[i]
            for j in p['DominationSet']:
                q = pop[j]
                q['DominatedCount'] -= 1
                if q['DominatedCount'] == 0:
                    Q.append(j)
                    q['Rank'] = k + 2
                pop[j] = q
        if not Q:
            break
        F.append(Q)
        k += 1

    return pop, F


def CD_calc(pop, F):
# %% CROWDING DISTANCE CALCULATING
    nF = len(F)

    for k in range(nF):
        front = F[k]
        n = len(front)
        if n == 0:
            continue

        # Lấy tất cả Cost của các cá thể trong front
        Costs = np.array([pop[i]['Cost'].flatten() for i in front]).T  # shape: (nObj, n)

        nObj = Costs.shape[0]
        d = np.zeros((n, nObj))

        for j in range(nObj):
            cj = Costs[j]  # all costs of j Obj
            so = np.argsort(cj)
            d[so[0], j] = np.inf
            d[so[-1], j] = np.inf

            denom = abs(cj[so[-1]] - cj[so[0]])
            if denom == 0:
                denom = 1e-12  # tránh chia cho 0

            for i in range(1, n - 1):
                d[so[i], j] = abs(cj[so[i + 1]] - cj[so[i - 1]]) / denom

        # Gán tổng khoảng cách cho từng cá thể trong front
        for i in range(n):
            pop[front[i]]['CrowdingDistance'] = np.sum(d[i])
# %%
    return pop

def sort_pop(pop):
    # %% Sort based on CrowdingDistance (giảm dần)
    pop.sort(key=lambda p: p['CrowdingDistance'], reverse=True)

    # Sort based on Rank (tăng dần)
    pop.sort(key=lambda p: p['Rank'])

    # Update Fronts
    ranks = [p['Rank'] for p in pop]
    max_rank = max(ranks)
    F = []

    for r in range(1, max_rank + 1):
        front = [i for i, rank in enumerate(ranks) if rank == r]
        F.append(front)

    return pop, F
