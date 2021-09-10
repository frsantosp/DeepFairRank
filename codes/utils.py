import numpy as np

def generate_bins_matrix(s,prt):
    B_0, B_1 = np.zeros((11,s.shape[0])), np.zeros((11,s.shape[0]))   
    # create the bins Matrix
    for j, val in enumerate(s):
        i = int(10*np.round(val,1))
        if prt[j]== 0:
            B_0[i,j] = 1
        else:
            B_1[i,j] = 1
    # Normalized the bins Matrix
    for i in range(11):
        if np.sum(B_0[i,:]) >0:
            B_0[i,:]= B_0[i,:]/np.sum(B_0[i,:])
        if np.sum(B_1[i,:]) >0:
            B_1[i,:]= B_1[i,:]/np.sum(B_1[i,:])
    return B_0, B_1

def binarize(h, idx):
    h_b = h.copy()
    indices = np.argsort(h_b)[::-1]
    h_b[indices[:idx]] = 1
    h_b[indices[idx:]] = 0
    return h_b


def power_mean(x, p):
    x = np.array([v**p for v in x])/len(x)
    return np.power(np.sum(x), 1/p)