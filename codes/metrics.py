import numpy as np
import sklearn.metrics as metric
from codes.utils import  binarize, generate_bins_matrix


def accuracy(h,y):
    return (np.sum(h*y) + np.sum((1-h)*(1-y)))/h.shape[0]


def average_perception(W, h, eps = 1e-12 ):
    TV = np.sum(W.dot(h)<=h+eps)/W.shape[0]
    return  TV
    
def weighted_statistical_disparity(h, B_0, B_1):
    B_W = np.diag(np.arange(0,1.1, 0.1))
    x = B_0.dot(h) - B_1.dot(h)
    fp = x.T.dot(B_W.dot(x))/5.5
    return fp    
    
def evaluate(h,y,s,prt_group, W, B_0 = None, B_1 = None, binarized = True):   
    if B_0 is None:
        B_0, B_1 = generate_bins_matrix(s,prt_group)
    if binarized:
        h_b = binarize(h, int(np.sum(y)))
    else:
        h_b = h.copy()
    fv = average_perception(W, h_b)
    ap = metric.average_precision_score(y, h)
    fgrp =  weighted_statistical_disparity( h, B_0, B_1)
    print("   average precision: {:.4f} |  weighted statistical disparity: : {:.4f} | Visibility: {:.4f}".format(ap ,fgrp , fv))






