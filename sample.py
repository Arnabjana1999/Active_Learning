import os
import random
from random import sample
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt

import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import dgl.nn as dglnn

from utils import *
from sample import *
from models.model import *

def find_nbr_nonnbr(G):
    """
    A routine that processes a networkx graph and emits list of neighbours and non-neighbours for each node.
    Input: NetworkX graph
    Returns: dictionary of neighbour and non-neighbors
    Do not use on large graphs since non-neighbour dictionary is O(n^3) storage, n: number of vertices. 
    """
    
    vertex_set  = set(G.nodes())
    vertex_list = list(vertex_set)
    
    nbr_dict, nonnbr_dict = {}, {}

    for node in range(len(vertex_list)):
        nbr_set = set([nbr for nbr in G[node]])
        nonnbr_set = list(vertex_set - nbr_set)

        nbr_dict[node] = nbr_set
        nonnbr_dict[node] = nonnbr_set

    return nbr_dict, nonnbr_dict

class Graph:
    def __init__(self, u, v, private_nodes, filename=''):
        """
        Initialize a NetworkX graph from a file with edge list.
        Raises Exception if provided file is not an edge list
        """
        # G = nx.read_edgelist(filename)
        # self.GG = G
        # self.G = nx.convert_node_labels_to_integers(G)
#         self.G = nx.DiGraph(self.G)
        self.private_nodes = private_nodes
       	G = dgl.graph((u, v))
        self.G_x = G.to_networkx().to_undirected()
        self.vertex_set = set(self.G_x.nodes())
        self.vertex_list = list(self.vertex_set)
        
    def split_train_test_private(self, test_fraction, prv_fraction=0.7):
        """
        Prepares the graph for training by creating a train, test graph with non-overlapping edges 
        Input test_fraction: Fraction of neighbours per node that make the test split.
        Returns: None
        Adds to the self test_edges_list, train_edges_list both of which are list of triples (in, out, edge-type)
        A new graph with edges from test omitted is attached to self called G_train. 
        """
        assert test_fraction<=1 and test_fraction>=0

        self.test_fraction = test_fraction
        
        nbr_dict, nonnbr_dict = find_nbr_nonnbr(self.G_x)
        self.nbr_dict, self.nonnbr_dict = nbr_dict, nonnbr_dict

        # print(nbr_dict[0])
        ## per_node_private_set consists of both private edges and non_edges-> non_empty only for private_nodes
        per_node_train_set, per_node_test_set, per_node_private_set = {}, {}, {}  
        oracle = {}         
        test_edges_list, train_edges_list = [], []        
        for node in range(len(self.vertex_list)):            
            per_node_test_set[node], per_node_train_set[node], per_node_private_set[node] = {}, {}, {}
            
            x_nbr = int(test_fraction*len(nbr_dict[node]))
            x_nonnbr = int(test_fraction*len(nonnbr_dict[node]))
            x_prv_nbr = int(prv_fraction*len(nbr_dict[node]))
            x_prv_nonnbr = int(0.25 * len(nonnbr_dict[node]))
            
#             print(x_nbr)
            
            per_node_test_set[node]['nbr'] = sample(nbr_dict[node], x_nbr)
            if node in self.private_nodes:
              # print('In')
              # print(nbr_dict[node])
              # print(per_node_test_set[node]['nbr'])
              per_node_private_set[node]['nbr'] =  sample(list(set(nbr_dict[node]) - set(per_node_test_set[node]['nbr'])), x_prv_nbr)
              for nbr in per_node_private_set[node]['nbr']:
                oracle[(node,nbr)] = 1
            else:
              per_node_private_set[node]['nbr'] = []
            per_node_train_set[node]['nbr'] =  list((set(nbr_dict[node])\
                                                       - set(per_node_test_set[node]['nbr'])) - set(per_node_private_set[node]['nbr']))
            ## debug statement
            # if node in private_nodes:
            #   print(node, len(per_node_train_set[node]['nbr']), len(per_node_private_set[node]['nbr']), len(per_node_test_set[node]['nbr']))
    
            per_node_test_set[node]['nonnbr'] = sample(nonnbr_dict[node], x_nonnbr)
            if node in self.private_nodes:
              per_node_private_set[node]['nonnbr'] =  sample(list(set(nonnbr_dict[node]) - set(per_node_test_set[node]['nonnbr'])), x_prv_nonnbr)
              for nonnbr in per_node_private_set[node]['nonnbr']:
                oracle[(node,nonnbr)] = 0
            else:
              per_node_private_set[node]['nonnbr'] = []
            per_node_train_set[node]['nonnbr'] =  list((set(nonnbr_dict[node])\
                                                  - set(per_node_test_set[node]['nonnbr'])) - set(per_node_private_set[node]['nonnbr']))
            
    
            test_edges_per_node = [(node, x) for x in per_node_test_set[node]['nbr']]
            test_non_edges_per_node  = [(node, x) for x in per_node_test_set[node]['nonnbr']]
            train_edges_per_node = [(node, x) for x in per_node_train_set[node]['nbr']]
            train_non_edges_per_node  = [(node, x) for x in per_node_train_set[node]['nonnbr']]
            
            test_edges_list.extend([(a, b, 1) for a, b in test_edges_per_node])
            test_edges_list.extend([(a, b, 0) for a, b in test_non_edges_per_node])

            train_edges_list.extend([(a, b, 1) for a, b in train_edges_per_node])
            train_edges_list.extend([(a, b, 0) for a, b in train_non_edges_per_node])

        # print('Out')
            
        self.test_edges_list = test_edges_list         
        self.train_edges_list = train_edges_list

        self.oracle = oracle
        
        self.test_edges_per_node = per_node_test_set
        self.test_non_edges_per_node = test_non_edges_per_node
        self.train_edges_per_node = per_node_train_set
        # self.test_non_edges_per_node = test_non_edges_per_node

        self.private_edges_per_node = per_node_private_set
        
        # G_train =  copy.deepcopy(self.G)
        # G_train.remove_edges([(a, b) for (a, b, label) in test_edges_list if label==1])
        train_edges_list_1 = [(a,b) for (a,b,label) in train_edges_list if label==1]

        edges_src = [a for (a,b) in train_edges_list_1]
        edges_dst = [b for (a,b) in train_edges_list_1]
        # print(edges_src)
        G_train = dgl.graph((edges_src, edges_dst))

        self.G_train = G_train