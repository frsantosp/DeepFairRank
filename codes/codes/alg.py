import numpy as np
from codes.metrics import average_perception, weighted_statistical_disparity
from codes.utils import  generate_bins_matrix, power_mean
import cvxpy as cp
from tqdm import tqdm
import sklearn.metrics as metric
from cvxopt import matrix, solvers
from scipy import stats

###########################################
# FPRank Algorithm
###########################################

def FPRank(W, s, y, prt, norm = -2, arep = 10, brep = 20, solver = 'SCS', numBins = 10, bin_width = 0.1, arange = [-2,2], brange = [-1,3]):  
 
    k = int(np.sum(y))
    n, m = W.shape

    B0,B1 = generate_bins_matrix(s,prt,numBins)
    B_W = np.diag(np.arange(0,1.1, bin_width))
   
    x = cp.Variable(n)
    z = cp.Variable(m)
    
    error = cp.sum_squares(s-x)
    GF = cp.quad_form(B0@x-B1@x, B_W)
    
    alpha = cp.Parameter(nonneg=True)
    beta = cp.Parameter(nonneg=True)
    norm_z = cp.norm(z, 2)

    obj = cp.Minimize(error + alpha*GF + beta*norm_z)
    constraints = [0 <= x, x<= 1, 0<= z, W@x <= x + z ]
    prob = cp.Problem(obj, constraints)

    params = {'alpha': list(), 'beta': list()}
    results = {'corr': list(), 'precision': list(), 'fairperception': list(), 'wsd': list(), 'wrd': list(), 'h': list()}
    
    for val in tqdm(np.logspace(arange[0], arange[1], arep)):
        alpha.value = val
        for val_b in np.logspace(brange[0], brange[1], brep):   
            beta.value = val_b
            try:
                params['alpha'].append(val)
                params['beta'].append(val_b)

                prob.solve(solver= solver)
                h = x.value
                
                fv = average_perception(W, h)
                corr = stats.spearmanr(h, s).statistic
                wsd = weighted_statistical_disparity(h, B0, B1, bin_width)
                wrd = weighted_statistical_disparity(h, B0, B1, bin_width, toRank=True)                
                prec = metric.average_precision_score(y, h)
                
                results['h'].append(h)
                results['corr'].append(corr)
                results['precision'].append(prec)
                results['fairperception'].append(fv)
                results['wsd'].append(wsd)
                results['wrd'].append(wrd)
            except:
                pass     
            print('  alpha={:.4f}, beta={:.4f}: prec={:.4f}, corr={:.4f}, fp={:.4f}, wsd={:.6f}, wrd={:.6f}'.format(val, val_b, prec, corr, fv, wsd, wrd))
            
    return results, params

###########################################
# Hyperparameter tuning for FPRank Algorithm
###########################################

def best_FPRank(results, params, criterion = 'pmean', toRank=True, debug=False):
    if toRank:
        parity = 1 - np.array(results['wrd'])
    else:
        parity = 1 - np.array(results['wsd'])

    precs = np.array(results['precision'])
    fp = np.array(results['fairperception'])
    corr = np.array(results['corr'])
              
    utils = list()
    for i in range(precs.shape[0]):
        if criterion == 'pmean':
            util = power_mean([corr[i], fp[i], parity[i]], 3)
        elif criterion == 'amean':
            util = (corr[i] + fp[i] + parity[i])/3
        else:
            util = np.min([corr[i], fp[i], parity[i]])
            
        utils.append(util)
        if debug:
            print('Param = {:.4f},{:.4f} util={:.4f} | prec={:.4f}, corr={:.4f}, parity={:.4f}, fp={:.4f}'.format(params['alpha'][i], params['beta'][i], util, precs[i], corr[i], parity[i], fp[i]))
        
    bestparam = np.argmax(utils)
    print('\nBest parameters: alpha = {:.4f}, beta = {:.4f} (util = {:.4f})'.format(params['alpha'][bestparam], params['beta'][bestparam], utils[bestparam]))
    return results['h'][bestparam], bestparam, utils 

###########################################
# Baseline: FSPR Algorithm
###########################################

def FSPR_model(adj, s, prt, gamma=0.15):
    N = s.shape[0]
    phi = prt.sum()/N
    
    norm_adj = np.diag(1/(np.sum(adj, axis = 1)+np.finfo(np.float32).eps)).dot(adj)
    Q = gamma*np.linalg.inv(np.eye(N)-(1-gamma)*norm_adj)
    qr = np.dot(Q, prt)
    A = matrix(np.vstack((qr, np.ones(N))))
    b = matrix(np.array([phi, 1]).reshape(-1,1))
    
    QP = matrix(2*Q*Q.T)
    p = -2*matrix(np.dot(Q,s))
    G = matrix(np.vstack((-1*np.eye(N), np.eye(N))))
    h = matrix(np.hstack((np.zeros(N), np.ones(N))))
    
    sol=solvers.qp(QP, p, G, h, A, b)
    return np.array(sol['x']) 