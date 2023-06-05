import networkx as nx
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import fairsearchcore as fsc
from fairsearchcore.models import FairScoreDoc
import aif360

def W_gen(adj, s):
    K = squareform(pdist(s))
    K = np.exp(-K**2)
    A = np.eye(adj.shape[0])+adj
    B = K*A
    W = np.diag(1/np.sum(B, axis = 1)).dot(B)
    return W

def load_data(file_name, class_label, score_label, prt_group, score_norm = 1):
    """
    Input:
        - file_name: path + filename 
        - class_label: column name for class (h_c)
        - score_label: column_name for score (s)
        - prt_group: column name for protected group

    Output: 
        - adj - adjacency matrix 
        - data - dataframe object
        - W - similariy matrix
        - s - score
        - h - class label
        - prt_attr - protected_group
    """
    data =  pd.read_csv(file_name + '.csv')

    A = pd.read_csv(file_name + '.edges')
    edges = zip(A['n1'].tolist(), A['n2'].tolist())
    g = nx.empty_graph(data.shape[0])
    g.add_edges_from(edges)
    edges = zip(A['n2'].tolist(), A['n1'].tolist())
    g.add_edges_from(edges)
    adj = nx.to_numpy_array(g)
    prt_attr = data[prt_group].values
    s = data[score_label].values/score_norm
    h = data[class_label].values
    prt_f = data[prt_group].values
    W = W_gen(adj, s.reshape(-1,1))
    
    return adj, data, W, s.ravel(), h, prt_attr

def load_data_fairtopk(file_name, class_label, score_label, prt_group, factor = 1):
    data =  pd.read_csv(file_name + '.csv')
    k = int(np.sum(data[class_label])) 
    p = 1-np.sum(data[prt_group])/data.shape[0]
    rankings = list()
    for idx in data[score_label].argsort()[::-1]:
        if data.loc[idx,prt_group] == 1:
            rankings.append( FairScoreDoc(idx, int(np.round(data.loc[idx,score_label]*factor)), False))
        if data.loc[idx,prt_group] == 0:
            rankings.append( FairScoreDoc(idx, int(np.round(data.loc[idx,score_label]*factor)), True))
    return rankings, k, p, data    

def load_data_aif(file_name, class_label, score_label, prt_group):
    """
    Load data in a format compatible with aif360 package.
    """
    data =  pd.read_csv(file_name)
    data['s'] = data[score_label]/10
    data['s_hat'] = binarize(data['s'], int(np.sum(data[class_label])))
    dataset = aif360.datasets.BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=data,
        label_names=['s_hat'],
        protected_attribute_names=[prt_group])
    dataset.scores = data['s'].to_numpy().reshape((data.shape[0],1))
    return dataset, data