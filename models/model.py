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


'''
MODEL DEFINED HERE

GraphSAGE is for computing embeddings for each node.

DotProduct takes those embeddings and computes edge_embeddings for all edges
in the graph (takes as argument in forward(...)). Note this may not be the training graph.
Now it is clear, why we created separate negative_graph and test_graph.

Model takes 2 graphs, computes SAGE embeddings using former graph and returns scores
for edges in both former and later graph.

For reference:-
https://docs.dgl.ai/guide/training-link.html

In this implementation, input-features are 10D, output-features (embeddings) are 50D.
score(u,v) = np.dot(embedding(u), embedding(v)), which is a real number
'''

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

class DotProductPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()
    def forward(self, g, neg_g, x):
        h = self.sage(g, x)
        return self.pred(g, h), self.pred(neg_g, h)