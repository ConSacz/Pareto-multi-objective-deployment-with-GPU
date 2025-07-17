import numpy as np
from Cov_Func_v2 import Cov_Func_v2
from Life_time import Life_Time
from Graph import Graph

def CostFunction_MOO(pop, stat, Obstacle_Area, Covered_Area):
    rs = stat[0,:]
    rc = stat[1,0]
    G = Graph(pop,rc)
    
    Coverage, _ = Cov_Func_v2(pop, rs, Obstacle_Area, Covered_Area)
    LifeTime = Life_Time(G)
    
    Cost = np.array([[Coverage], [LifeTime]])
    
    return Cost
