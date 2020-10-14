import numpy as np

def tpr_parity_opt(W,y,h_c, norm = 2, rep = 50, solver = 'SCS'):
    acceptance_n = int(np.sum(h_c))
    n = W.shape[0]
    pos_0 = sub_vec(y,(1- prt_group))
    pos_1 = sub_vec(y, prt_group)
   
    x = cp.Variable(n)
    error = cp.sum_squares(y-x)/cp.sum(y)
    PB = cp.norm(pos_0.T@x - pos_1.T@x, norm)
    alpha = cp.Parameter(nonneg=True)

    obj = cp.Minimize( error + alpha*PB ) 
    constraints = [0 <= x, W@x  <=x  ]
    prob = cp.Problem(obj, constraints)

    alpha_vals = np.logspace(-4, 0, rep)
    results = list()
    tpr_p = list()
    for val in alpha_vals:
        alpha.value = val
        prob.solve(solver= solver)#,adaptive_rho_interval = 1)
        h = x.value
        binarize(h,acceptance_n)
        results.append(h)
        tpg ,_,_,_ = group_fairness(y, h, prt_group, return_gap = True)
        tpr_p.append(tpg)
    i = np.argmin(tpr_p)
    return results[i]

def binarize(h, idx):
    indices = np.argsort(h)[::-1]
    h[indices[:idx]] = 1
    h[indices[idx:]] = 0
    
def sub_vec(a, b):
    r = a*b
    return r/np.sum(r)
