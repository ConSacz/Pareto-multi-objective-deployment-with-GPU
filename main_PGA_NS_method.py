#-------- Implementation of basic GPU-parallel Genetic Algorithm for with python CUDA using Numba  ---------#

import numpy as np
from numba import cuda
import time
import math
import matplotlib.pyplot as plt
from CostFunction_MOO import CostFunction_MOO
from Graph import Graph
from Connectivity_graph import Connectivity_graph
from GA_functions import Crossover, Mutate
from Domination_functions import get_pareto_front, NS_Sort, CD_calc, sort_pop
#-------- Verify CUDA access  ---------#
print(cuda.gpus)


#%%-------- Parallel kernel function using CUDA  ---------#
@cuda.jit(device=True)
def Cov_Func(pos, rs, Obstacle_Area, Covered_Area):
    size_x, size_y = Obstacle_Area.shape
    N=len(pos)/2
    count = 0
    for j in range (N):
        x0 = pos[j*2-1]
        y0 = pos[j*2]
        rsJ = rs[j]
        for xi in range(1,size_x):
            for yi in range(1,size_y):
                dx = xi - x0
                dy = yi - y0
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= rsJ:
                    if Obstacle_Area[xi, yi] == 1:
                        if Covered_Area[xi, yi] == 1:
                            continue
                        else:
                            Covered_Area[xi, yi] = 1
                            count = count + 1
                    elif Obstacle_Area[xi, yi] == 0:
                        Covered_Area[xi, yi] = -2
    return count/(size_x*size_y)                    
# @cuda.jit(device=True)
# def Liti_Func(pos,rc):
#     N = pos.shape[0]
#     adj_matrix = np.zeros((N, N))
#     for i in range(N):
#         for j in range(N):
#             if i != j:
#                 dist = np.linalg.norm(pos[i, :2] - pos[j, :2])
#                 if dist <= rc:
#                     adj_matrix[i, j] = dist
@cuda.jit
def eval_genomes_kernel(Positions, Costs, stat, OA, CA):
    nPop = len(Positions)
    rs=stat[0,:]
  # Thread id in a 1D block
    tx = cuda.threadIdx.x
  # Block id in a 1D grid
    ty = cuda.blockIdx.x
  # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
  # Compute flattened index inside the array
    i = tx + ty * bw
    if i < nPop:  # Check array boundaries
  # in this example the fitness of an individual is computed by an arbitary set of algebraic operations on the chromosome
        # Costs[i,:] = CostFunction_MOO(Positions[i,:].reshape(60, 2), stat, np.ones((size, size), dtype=int), np.zeros((size, size), dtype=int))
        #Costs[i,0] = Cov_Func(Positions[i,:],rs,OA,CA)
        size_x, size_y = OA.shape
        N=len(Positions[i,:])/2
        count = 0
        for j in range (N):
            x0 = Positions[i,:][j*2-1]
            y0 = Positions[i,:][j*2]
            rsJ = rs[j]
            for xi in range(1,size_x):
                for yi in range(1,size_y):
                    dx = xi - x0
                    dy = yi - y0
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= rsJ:
                        if OA[xi, yi] == 1:
                            if CA[xi, yi] == 1:
                                continue
                            else:
                                CA[xi, yi] = 1
                                count = count + 1
                        elif OA[xi, yi] == 0:
                            CA[xi, yi] = -2
#%%-------- Initialize Population  ---------#
np.random.seed(0)
nPop = 500
N = 60
MaxIt = 2
size = 100

# Network parameters
rc = 10
rs = np.ones(N, dtype=int) * 10
stat = np.zeros((2, N))  # tạo mảng 2xN
stat[0, :] = rs          # dòng 1 là rs
stat[1, 0] = rc          # phần tử đầu dòng 2 = rc
stat_GPU = cuda.to_device(stat)
sink = np.array([size//2, size//2])
Covered_Area = np.zeros((size, size), dtype=int)
CA_GPU = cuda.to_device(Covered_Area)
Obstacle_Area = np.ones((size, size), dtype=int)
OA_GPU = cuda.to_device(Obstacle_Area)

# GA parameters
pCrossover = 0.7                          # Crossover Percentage
nCrossover = 2 * round(pCrossover * nPop / 2)  # Number of Parents (=> even number of Offsprings)
pMutation = 0.4                           # Mutation Percentage
nMutation = round(pMutation * nPop)       # Number of Mutants
mu = 0.02                                 # Mutation Rate
sigma = 0.1 * (100)           # Mutation Step Size

# Init first pop
pop = []
for _ in range(nPop):
    alpop = np.random.uniform(15, 85 , (N, 2))
    alpop[0] = sink
    if Connectivity_graph(Graph(alpop, rc),[]):
        cost = CostFunction_MOO(alpop, stat, Obstacle_Area, Covered_Area.copy())
    else: 
        cost = np.array([[1], [1]])
    pop.append({'Position': alpop, 'Cost': cost})


#%%-------- Prepare kernel ---------#
# Set block & thread size
threads_per_block = 256
#blocks_per_grid = (nPop*N + (threads_per_block - 1))//threads_per_block
blocks_per_grid = nPop

#-------- Measure time to perform some generations of the Genetic Algorithm with CUDA  ---------#
print("CUDA:")
start = time.time()
#%% Genetic Algorithm on GPU
for it in range(MaxIt):
    print("Gen " + str(it) + "/" + str(MaxIt))
    
    # ----- Crossover -----
    popc = []
    for k in range(nCrossover // 2):
        i1 = np.random.randint(0, nPop)
        i2 = np.random.randint(0, nPop)
        p1 = pop[i1]
        p2 = pop[i2]

        y1, y2 = Crossover(p1['Position'], p2['Position'])
        y1[0, :] = sink  # giữ node sink cố định
        y2[0, :] = sink
        c1 = {
            'Position': y1,
            'Cost': np.array([[1], [1]])
        }           
        c2 = {
            'Position': y2,
            'Cost': np.array([[1], [1]])
        }
        popc.extend([c1, c2])

    # ----- Mutation -----
    popm = []
    for k in range(nMutation):
        i = np.random.randint(0, nPop)
        p = pop[i]

        mutated_pos = Mutate(p['Position'], mu, sigma)
        mutated_pos = np.clip(mutated_pos, 0, 100)
        m = {
            'Position': mutated_pos,
            'Cost': np.array([[1], [1]])
        }

        popm.append(m)

    # ----- Merge -----
    popa = popc + popm
    
    Positions = np.array([ind['Position'].flatten() for ind in popa])
    Costs = np.array([ind['Cost'].flatten() for ind in popa])
    Positions_GPU = cuda.to_device(Positions)
    Costs_GPU = cuda.to_device(Costs)
    
    eval_genomes_kernel[blocks_per_grid, threads_per_block](Positions_GPU, Costs_GPU, stat_GPU, OA_GPU, CA_GPU)
    
    Costs = Costs_GPU.copy_to_host()
    for k in range (len(popa)):
        if Connectivity_graph(Graph(popa[k]['Position'], rc),[]):
            popa[k]['Cost'] = Costs[k,:]
    
    pop = pop + popa
    pop, F = NS_Sort(pop)
    pop = CD_calc(pop, F)
    pop, F = sort_pop(pop)
    pop = pop[:nPop]
    pop, F = NS_Sort(pop)
    pop = CD_calc(pop, F)
    pop, F = sort_pop(pop)
# %% ------------------------- PLOT --------------------------
# Tạo mảng data từ Cost của Extra_archive
    data = np.array([ind['Cost'].flatten() for ind in get_pareto_front(pop)])  # mỗi ind là dict có key 'Cost'
    data_set = np.array([ind['Cost'].flatten() for ind in pop])

    # Tạo figure
    fig = plt.figure(1)
    plt.clf()
    
    # Vẽ Pareto front
    #plt.plot(data_set[:, 0], data_set[:, 1], 'o', color='g')
    plt.plot(data[:, 0], data[:, 1], 'o', color='r', label = 'NSABC')
    #plt.plot(data2[:, 0], data2[:, 1], 'o', color='g', label = 'NSGA500it')
    #plt.plot(data3[:, 0], data3[:, 1], 'o', color='b', label = 'NSGA200it')
    plt.legend()
    plt.xlabel('Non-coverage')
    plt.ylabel('Energy')
    None
    # Cập nhật đồ thị theo từng iteration
    plt.pause(0.01)
    
end = time.time()
print("time elapsed: " + str((end-start)))

#-------------------------------------------------------#