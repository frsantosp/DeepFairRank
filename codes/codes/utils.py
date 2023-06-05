import numpy as np
import pandas as pd

def generate_bins_matrix(s,prt,numBins):
    B_0, B_1 = np.zeros((numBins+1,s.shape[0])), np.zeros((numBins+1,s.shape[0]))   
    if s.min() < 0 or s.max() > 1:
        print('Warning: range of s is not between 0 and 1')
        
    # create the bins Matrix
    for j, val in enumerate(s):
        i = int(numBins*np.round(val,1))
        if prt[j]== 0:
            B_0[i,j] = 1
        else:
            B_1[i,j] = 1
    # Normalized the bins Matrix
    for i in range(numBins+1):
        if np.sum(B_0[i,:]) >0:
            B_0[i,:]= B_0[i,:]/np.sum(B_0[i,:])
        if np.sum(B_1[i,:]) >0:
            B_1[i,:]= B_1[i,:]/np.sum(B_1[i,:])
    return B_0, B_1

def generate_bins_matrix_compass(s,prt):
    B_0, B_1 = np.zeros((6,s.shape[0])), np.zeros((6,s.shape[0]))
    # create the bins Matrix
    for j, val in enumerate(s):
        i = int(np.floor(val*5))
        #print(i,j,val)
        if prt[j]== 0:
            B_0[i,j] = 1
        else:
            B_1[i,j] = 1
    # Normalized the bins Matrix
    for i in range(6):
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

def getRank(h):
    df = pd.DataFrame({'id': [i for i in range(len(h))], 'val': h})
    df = df.sort_values(by='val').reset_index().drop(columns=['index'])
    df = df.sort_values(by='id').reset_index()
    return df['index'].values/df.shape[0]