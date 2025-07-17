import numpy as np
# %% ------------------------- FUNCTIONS -------------------------- 
def Crossover(x1, x2):
    alpha = np.random.rand(*x1.shape)
    y1 = alpha * x1 + (1 - alpha) * x2
    y2 = alpha * x2 + (1 - alpha) * x1
    return y1, y2

def Mutate(x, mu, sigma):
    x = x.copy()
    nVar = x.size
    nMu = int(np.ceil(mu * nVar))
    
    j = np.random.choice(nVar, nMu, replace=False)

    if np.ndim(sigma) > 0 and len(sigma) > 1:
        sigma = sigma[j]
    
    if np.isscalar(sigma):
        noise = sigma * np.random.randn(len(j))
    else:
        noise = sigma * np.random.randn(len(j))
    
    x.flat[j] += noise  # dùng flat để đánh index 1 chiều

    return x