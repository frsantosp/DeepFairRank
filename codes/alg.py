import numpy as np
from codes.metrics import average_perception, accuracy
import cvxpy as cp
from tqdm import tqdm

from codes.utils import power_mean, binarize, generate_bins_matrix



def FPRank(W,s,h_c,prt_group, norm = -10, rep = 40, solver = 'SCS', ):  
    acceptance_n = int(np.sum(h_c))
    s_b = binarize(s, acceptance_n)
    n = W.shape[0]
    m = W.shape[1]

    B0,B1 = generate_bins_matrix(s,prt_group)
    B_W = np.diag(np.arange(0,1.1, 0.1))
   
    x = cp.Variable(n)
    z = cp.Variable(m)
    
    error = cp.sum_squares(s-x)
    #PB = cp.norm(pos_0.T@x - pos_1.T@x, norm)
    GF = cp.quad_form(B0@x-B1@x, B_W)
    
    alpha = cp.Parameter(nonneg=True)
    beta = cp.Parameter(nonneg=True)
    norm_z = cp.norm(z, 2)

    obj = cp.Minimize( error + alpha*GF + beta*norm_z ) 
    constraints = [0 <= x, x<= 1, 0<= z, W@x  <=x + z ]
    prob = cp.Problem(obj, constraints)

    alpha_vals = np.logspace(0, 1, 10)
    beta_vals = np.logspace(-2, 2, rep)
    
    tavs = list()
    results = list()
    for val in tqdm(alpha_vals):
        alpha.value = val
        for val_b in beta_vals:   
            beta.value = val_b
            try:
                prob.solve(solver= solver)
                h = x.value
                h_b = binarize(h, acceptance_n)
                fv = average_perception(W, h_b)
                acc_hat = accuracy(h_b,s_b)
                results.append(h)
                tav = power_mean([acc_hat, fv], norm)
                tavs.append(tav)
            except:
                pass     
    i = np.argmax(tavs)
    return results[i]


def MaxPerception(W,y,h_c,prt_group, solver = 'SCS'):
    acceptance_n = int(np.sum(h_c))
    n = W.shape[0]
   
    x = cp.Variable(n)
    error = cp.sum_squares(y-x)/cp.sum(y)

    obj = cp.Minimize( error) 
    constraints = [0 <= x, W@x  <=x  ]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver= solver)#,adaptive_rho_interval = 1)
    h = x.value
    h = binarize(h,acceptance_n)
    return h

