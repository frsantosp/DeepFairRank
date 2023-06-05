import numpy as np
import sklearn.metrics as metric
import matplotlib.pyplot as plt
from codes.utils import  binarize, generate_bins_matrix, power_mean, getRank
from scipy import stats

def average_perception(W, h, eps = 1e-12 ):
    TV = np.sum(W.dot(h)<=h+eps)/W.shape[0]
    return  TV
    
def weighted_statistical_disparity(h, B_0, B_1, bin_width, toRank=False, debug=False):
    if toRank:
        h = getRank(h.copy())
    B_W = np.diag(np.arange(0,1.1, bin_width))
    x = B_0.dot(h) - B_1.dot(h)
    if debug:
        print('\t WSD B0:', B_0.dot(h))
        print('\t WSD B1:', B_1.dot(h))    
    return x.T.dot(B_W.dot(x))/B_W.shape[0]

def evaluate(h,y,s,prt_group, W, B_0 = None, B_1 = None, binarized = True, numBins = 10, bin_width = 0.1):   
    if B_0 is None:
        B_0, B_1 = generate_bins_matrix(s,prt_group,numBins)
        
    if binarized:
        h_b = binarize(h, int(np.sum(y)))
    else:
        h_b = h.copy()
    
    #######################################
    # Display distribution of h values
    #######################################
    
    plt.figure(figsize=(6,3))
    plt.hist(h)
    plt.xlabel('h')

    #######################################
    # Compute evaluation metrics
    #######################################

    fv = average_perception(W, h)
    ap = metric.average_precision_score(y, h)
    corr = stats.spearmanr(h, s).statistic
    #print('WSD (original):')
    wsd =  weighted_statistical_disparity( h, B_0, B_1, bin_width)
    #print('WSD (re-rank):')
    wrd =  weighted_statistical_disparity( h, B_0, B_1, bin_width, toRank=True)

    #######################################
    # Display result of evaluation metrics
    #######################################

    print("\n  prec: {:.6f} |  Rank corr: {:.6f}  | fp: {:.6f} | wsd: {:.6f} | wrd: {:.6f}".format(ap, corr, fv, wsd, wrd))
    result = {'precision': ap, 'corr': corr, 'fairperception': fv, 'wsd': wsd, 'wrd': wrd}
    return result    