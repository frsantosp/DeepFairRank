import networkx as nx
import pandas as pd
import numpy as np



def W_gen(adj, s):
    K = np.zeros(adj.shape)
    for i, s1 in enumerate(s):
        for j, s2 in enumerate(s):
            K[i,j] = np.exp(-(s1-s2)**2)
    A = np.eye(adj.shape[0])+adj
    W = np.diag(1/np.sum(K*A, axis = 0)).dot(K*A)
    return W

def load_data( year = 2020 , path = './data/'):    
    data =  pd.read_csv(path + 'ICLR'+str(year)+'.data')
    g = nx.empty_graph(data.shape[0])
    file = open(path + 'ICLR'+str(year)+'.edges', 'r')
    edges = list()
    for edge  in [line.strip().split(' ') for line in file.readlines()]:
        g.add_edge(int(edge[0]),int(edge[1]))
    adj = nx.to_numpy_array(g)
    s = np.array(data['s'])
    h = np.array(data['h_c'])
    prt_f = np.array(data['famous'])
    prt_t = np.array(data['top'])
    W = W_gen(adj, s)
    return adj, s/10, h, prt_f, prt_t, W