import numpy as np
from tabulate import tabulate

def accuracy(h,y):
    return (np.sum(h*y) + np.sum((1-h)*(1-y)))/h.shape[0]


def fairness_visibility(W ,y, h,prt_group, eps = 1e-12  ,output= False):
    TV = np.sum(W.dot(h)<=h+eps)/W.shape[0]
    idx = (prt_group == 0)
    V0 = np.sum(W[idx].dot(h)<= h[idx]+eps )/np.sum(idx)
    idx = (prt_group == 1)
    V1 = np.sum(W[idx].dot(h) <= h[idx]+eps )/np.sum(idx)
    if not(output):
        print("(total visibility:) {:.4f} | (visibility 0:) {:.4f} | (visibility 1:) {:.4f}".format(TV,V0,V1))
    else:
        return V0, V1, TV
    
def group_fairness(y, h, prt_group, return_gap = False):
    pos_0 = y*(1- prt_group)
    pos_1 = y*prt_group
    neg_0 = (1-y)*(1- prt_group)
    neg_1 = (1-y)*prt_group
    if not(return_gap):
        print('General performance')
        TP, FP, FN, TN = numbers_numpy(y,1-y,h)
        TPR, FPR, FNR, TNR = rates_numpy(y,1-y,h)
        print("   TP {:.4f} | FP {:.4f} | FN {:.4f} | TN {:.4f}".format(TP, FP, FN, TN))
        print("   TPR {:.4f} | FPR {:.4f} | FNR {:.4f} | TNR {:.4f}".format(TPR, FPR, FNR, TNR))
        print('Protected value:', 0)
        TP, FP, FN, TN = numbers_numpy(pos_0, neg_0, h)
        TPR, FPR, FNR, TNR = rates_numpy(pos_0, neg_0, h)
        print("   TP {:.4f} | FP {:.4f} | FN {:.4f} | TN {:.4f}".format(TP, FP, FN, TN))
        print("   TPR {:.4f} | FPR {:.4f} | FNR {:.4f} | TNR {:.4f}".format(TPR, FPR, FNR, TNR))
        print('Protected value:', 1)
        TP, FP, FN, TN = numbers_numpy(pos_1, neg_1, h)
        TPR, FPR, FNR, TNR = rates_numpy(pos_1, neg_1, h)
        print("   TP {:.4f} | FP {:.4f} | FN {:.4f} | TN {:.4f}".format(TP, FP, FN, TN))
        print("   TPR {:.4f} | FPR {:.4f} | FNR {:.4f} | TNR {:.4f}".format(TPR, FPR, FNR, TNR))
    else:
        TPR_0, FPR_0, FNR_0, TNR_0 = rates_numpy(pos_0, neg_0, h)
        TPR_1, FPR_1, FNR_1, TNR_1 = rates_numpy(pos_1, neg_1, h)
        return(np.abs(TPR_0-TPR_1 ), np.abs(FPR_0-FPR_1 ), np.abs(FNR_0-FNR_1 ), np.abs(TNR_0-TNR_1 ))

def numbers_numpy(yp,yn ,h):
    TP = np.sum(yp*h)
    FP = np.sum((yn)*h)                                                      
    FN = np.sum(yp*(1-h))
    TN = np.sum((yn)*(1-h))
    return TP, FP, FN, TN

def rates_numpy(yp,yn ,h):
    TPR = np.sum(yp*h)/np.sum(yp)
    FPR = np.sum((yn)*h)/np.sum(yn)                                                        
    FNR = np.sum(yp*(1-h))/np.sum(yp) 
    TNR = np.sum((yn)*(1-h))/np.sum(yn) 
    return TPR, FPR, FNR, TNR



