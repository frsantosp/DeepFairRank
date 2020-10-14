import networkx as nx
import pandas as pd
import numpy as np

def W_gen(adj, y):
    k1 = adj.dot(y)
    k0 = adj.dot(1-y)
    W = np.zeros(adj.shape)
    n = adj.shape[0]
    for i in range(n):
        if y[i] == 1 and k1[i] > 0:
            for j in range(n):
                if adj[i,j] == 1:
                    W[i,j] = y[j]/k1[i]
        elif y[i] == 0 and k0[i]>0:
            for j in range(n):
                if adj[i,j] == 1:
                    W[i,j] = (1-y[j])/k0[i]  
        else:
            for j in range(n):
                W[i,j] = 1.0/n
    return W
    
def load_data( year = 2020, path = './data/'):    
    data =  pd.read_csv(path + 'ICLR'+str(year)+'.data')
    g = nx.empty_graph(data.shape[0])
    file = open(path + 'ICLR'+str(year)+'.edges', 'r')
    edges = list()
    for edge  in [line.strip().split(' ') for line in file.readlines()]:
        g.add_edge(int(edge[0]),int(edge[1]))
    adj = nx.to_numpy_array(g)
    y = np.array(data['y'])
    h = np.array(data['h_c'])
    prt_f = np.array(data['famous'])
    prt_t = np.array(data['top'])
    W = W_gen(adj, y)
    return adj, y, h, prt_f, prt_t, W

