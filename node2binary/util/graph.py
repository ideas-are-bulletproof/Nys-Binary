from collections import defaultdict
import logging
import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import sys
import time
seed = 123
np.random.seed(seed)

class Graph(object):
    ''' Note: adj_list shows each edge twice. So edge_num is really two times of edge number for undirected graph.'''

    def __init__(self, node_num, edge_num):
        self.node_num = node_num  # n
        self.edge_num = edge_num  # m
        self.adj_list = np.zeros(edge_num, dtype=np.int32) - 1  # a big array for all the neighbors.
        self.adj_idx = np.zeros(node_num + 1,
                                dtype=np.int32)  # idx of the beginning neighbors in the adj_list. Pad one additional element at the end with value equal to the edge_num, i.e., self.adj_idx[-1] = edge_num
        self.adj_wgt = np.zeros(edge_num,
                                dtype=np.float32)  # same dimension as adj_list, wgt on the edge. CAN be float numbers.
        self.node_wgt = np.zeros(node_num, dtype=np.float32)
        self.cmap = np.zeros(node_num, dtype=np.int32) - 1  # mapped to coarser graph

        # weighted degree: the sum of the adjacency weight of each vertex, including self-loop.
        self.degree = np.zeros(node_num, dtype=np.float32)
        self.A = None
        self.C = None  # Matching Matrix

        self.coarser = None
        self.finer = None

    def resize_adj(self, edge_num):
        '''Resize the adjacency list/wgts based on the number of edges.'''
        self.adj_list = np.resize(self.adj_list, edge_num)
        self.adj_wgt = np.resize(self.adj_wgt, edge_num)

    def get_neighs(self, idx):
        '''obtain the list of neigbors given a node.'''
        istart = self.adj_idx[idx]
        iend = self.adj_idx[idx + 1]
        return self.adj_list[istart:iend]

    def get_neigh_edge_wgts(self, idx):
        '''obtain the weights of neighbors given a node.'''
        istart = self.adj_idx[idx]
        iend = self.adj_idx[idx + 1]
        return self.adj_wgt[istart:iend]

class Mapping:
    '''Used for mapping index of nodes since the data structure used for graph requires continuous index.'''
    def __init__(self, old2new, new2old):
        self.old2new = old2new
        self.new2old = new2old
    
def _read_graph_from_edgelist(ctrl, datafile):
    '''Assume each edge shows up ONLY once: small-id<space>large-id, or small-id<space>large-id<space>weight. 
    Indices are not required to be continuous.'''
#    logging.info("Reading graph from edgelist...")
    in_file = open(datafile)
    neigh_dict = defaultdict(list)
    max_idx = -1
    edge_num = 0
    for line in in_file:
        eles = line.strip().split()
        n0, n1 = [int(ele) for ele in eles[:2]]
        # print(n0, n1)
        if n0 > n1: #first id in a row should be the smaller one...
           continue
        if len(eles) == 3: # weighted graph
            wgt = float(eles[2])
            neigh_dict[n0].append((n1, wgt))
            if n0 != n1:
                neigh_dict[n1].append((n0, wgt))
        else:
            neigh_dict[n0].append(n1)
            if n0 != n1:
                neigh_dict[n1].append(n0)
        if n0 != n1:
            edge_num += 2
        else:
            edge_num += 1
        max_idx = max(max_idx, n1)
    in_file.close()
    weighted = (len(eles) == 3)
    #continuous_idx = (max_idx+1 == len(neigh_dict)) # starting from zero
    mapping = None
    #if not continuous_idx:
    old2new = dict()
    new2old = dict()
    cnt = 0
    sorted_keys = sorted(neigh_dict.keys())
    for key in sorted_keys:
        old2new[key] = cnt
        new2old[cnt] = key
        cnt += 1
    new_neigh_dict = defaultdict(list)
    for key in sorted_keys:
        for tpl in neigh_dict[key]:
            node_u = old2new[key]
            if weighted:
                new_neigh_dict[node_u].append((old2new[tpl[0]], tpl[1]))
            else:
                new_neigh_dict[node_u].append(old2new[tpl])
    del sorted_keys
    neigh_dict = new_neigh_dict # remapped
    mapping = Mapping(old2new, new2old)

    node_num = len(neigh_dict)
    graph = Graph(node_num, edge_num)
    edge_cnt = 0
    graph.adj_idx[0] = 0
    for idx in range(node_num):
        graph.node_wgt[idx] = 1 # default weight to nodes
        for neigh in neigh_dict[idx]:
            if weighted:
                graph.adj_list[edge_cnt] = neigh[0]
                graph.adj_wgt[edge_cnt] = neigh[1]
            else:
                graph.adj_list[edge_cnt] = neigh
                graph.adj_wgt[edge_cnt] = 1.0
            edge_cnt += 1
        graph.adj_idx[idx+1] = edge_cnt

    if ctrl.debug_mode:
        assert nx.is_connected(graph2nx(graph)), "Only single connected component is allowed for embedding."
    
    graph.A = graph_to_adj(graph, self_loop=False)
    return graph, mapping    

def graph2nx(graph): # mostly for debugging purpose. weights ignored.
    G=nx.Graph()
    for idx in range(graph.node_num):
        for neigh_idx in range(graph.adj_idx[idx], graph.adj_idx[idx+1]):
            neigh = graph.adj_list[neigh_idx]
            if neigh>idx:
                G.add_edge(idx, neigh)
    return G

def graph_to_adj(graph, self_loop=False):
    '''self_loop: manually add self loop or not'''
    node_num = graph.node_num
    i_arr = []
    j_arr = []
    data_arr = []
    for i in range(0, node_num):
        for neigh_idx in range(graph.adj_idx[i], graph.adj_idx[i+1]):
            i_arr.append(i)
            j_arr.append(graph.adj_list[neigh_idx])
            data_arr.append(graph.adj_wgt[neigh_idx])
    adj = sp.csr_matrix((data_arr, (i_arr, j_arr)), shape=(node_num, node_num), dtype=np.float32)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    return adj

def cmap2C(cmap): # fine_graph to coarse_graph, matrix format of cmap: C: n x m, n>m.
    node_num = len(cmap)
    i_arr = []
    j_arr = []
    data_arr = []
    for i in range(node_num):
        i_arr.append(i)
        j_arr.append(cmap[i])
        data_arr.append(1)
    return sp.csr_matrix((data_arr, (i_arr, j_arr)))        

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(screen_handler)
    return logger
