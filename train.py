
"""
TUNABLE PARAMETERS/HYPERPARAMETERS:

1. no_of_clusters = 4
2. size_of_cluster = n * 0.1
3. prv_fraction = 0.7 (fraction of nbrs/nonnbrs which are private)
4. negative_sampling = 0.2 * all non_nbrs
5. embedding_size = 50
6. learning_rate = 0.01
7. Budget = 200
8. No. of queries in single bunch = 40
9. No. of training epochs per oracle-query = 10
10. Cross-validation after every 5 training epochs

4 graphs have been created from a single adjacency list :-

1. Original graph (training)
2. Negative graph (training)
3. Private graph (uncertain edges for oracle-query)
4. Test graph (inference)

Further details have been mentioned as comments, along with its respective code
"""

#imports
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

src_dst = pd.read_csv("cora.cites", sep='\t', header=None, names=["target", "source"]).to_numpy()

src=src_dst[:,0]
dst=src_dst[:,1]

u = np.sort(list(set(np.concatenate([src, dst]))))
cur_ind = 0
node_index = {}

for elem in u:
  node_index[elem] = cur_ind
  cur_ind += 1

src = np.array([node_index[x] for x in src])
dst = np.array([node_index[x] for x in dst])

# Edges are directional in DGL; Make them bi-directional.
u = np.concatenate([src, dst])
v = np.concatenate([dst, src])
# Construct a DGLGraph
G = dgl.graph((u, v))
G_x = G.to_networkx().to_undirected()

print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())
print(G.nodes())

# no. of nodes in the graph
n=len(G.nodes())

# clusters is the no. of uncertain regions
no_of_clusters = 4
# fixing the no. of uncertain nodes in the region
size_of_cluster = int(n * 0.1) + 1

# choosing a representative node from each cluster
cluster_rep = np.random.choice(n,size=no_of_clusters,replace=False)

# sampling a set of nodes around each cluster through random-walk
sam = dgl.sampling.random_walk(G, cluster_rep, length=size_of_cluster)[0]

# aggregating all private_nodes into a set
# private_nodes = set(torch.reshape(sam, (-1,)).tolist())
private_nodes = np.array([0, 28, 15, 71, 84, 27, 72, 47, 87, 13, 52, 198, 333, 809, 279, 3, 48, 68, 543, 131])

'''
CREATING ORIGINAL GRAPH (FOR TRAINING)
'''


g = Graph(u, v, private_nodes)
g.split_train_test_private(test_fraction = 0.2)
graph = g.G_train

nx_G = graph.to_networkx().to_undirected()

priv_cnt = 0
for (key, val) in g.oracle.items():
  if val==1:
    # print(key, val)
    priv_cnt += 1

print('Private nodes = ' + str(len(private_nodes)))
# print('No. of training edges = '+str(len(g.train_edges_list)))
# print('No. of test edges = ' + str(len(g.test_edges_list)))
print('No. of private edges = ' + str(priv_cnt))

## initializing 10D features for each node in the graph
graph.ndata['feat'] = torch.tensor(pd.read_csv("cora.content", sep='\t', header=None).drop([0, 1434], axis=1).to_numpy())

no_of_nodes = len(g.vertex_list)

neg_src, neg_dst = [], []

'''
CREATING NEGATIVE GRAPH (FOR TRAINING)

negative_graph is a graph whose edges are a subset of non_edges of the original graph
(subset of complement graph).
This is required to reduce the complexity of pairwise hinge-loss (ranking-loss) as
no. of non-edges >> no. of edges
'''

for node in g.vertex_list:
  all_neg = g.train_edges_per_node[node]['nonnbr']
  sample_size = int(0.2 * len(all_neg))
  sample_neg = np.random.choice(all_neg, size=sample_size, replace=False)
  neg_src += [node for _ in sample_neg]
  neg_dst += [node for node in sample_neg]

negative_graph = dgl.graph((neg_src, neg_dst), num_nodes=no_of_nodes)

'''
CREATING TEST GRAPH (FOR INFERENCE)

test_graph is a graph whose edges are the test_edges and test_non_edges
of the original graph. 

There is a label (1/0) associated with each edge in test_graph, indicating whether
the edge is an edge/non-edge of the original graph.

Such a non-intuitive graph creation is necessary because during inference, we need to find
score for all node-pairs in the test_set (for ranking) and creating a graph with all
such node-pairs (to be ranked) as edges, makes it an easy and efficient routine in DGL.
'''

N = len(graph.nodes())
test_src, test_dst, edge_label = [], [], []

for node in range(N):
  all_nbr = g.test_edges_per_node[node]['nbr']
  test_src += [node for _ in all_nbr]
  test_dst += [node for node in all_nbr]
  edge_label += [1 for _ in all_nbr]

  all_nonnbr = g.test_edges_per_node[node]['nonnbr']
  test_src += [node for _ in all_nonnbr]
  test_dst += [node for node in all_nonnbr]
  edge_label += [0 for _ in all_nonnbr]

test_graph = dgl.graph((test_src, test_dst), num_nodes=no_of_nodes)
test_graph.edata['label'] = torch.tensor(edge_label)

nx_Gtest = test_graph.to_networkx().to_undirected()

'''
CREATING PRIVATE GRAPH (FOR FINDING UNCERTAIN EDGES BEFORE ORACLE-QUERY)

private_graph is a graph whose edges are the private node-pairs
of the original graph. 
'''

private_src, private_dst = [], []

for node in private_nodes:
  all_nbr = g.private_edges_per_node[node]['nbr']
  private_src += [node for _ in all_nbr]
  private_dst += [node for node in all_nbr]

  all_nonnbr = g.private_edges_per_node[node]['nonnbr']
  private_src += [node for _ in all_nonnbr]
  private_dst += [node for node in all_nonnbr]

private_graph = dgl.graph((private_src, private_dst), num_nodes=no_of_nodes)

nx_Gpriv = private_graph.to_networkx().to_undirected()

'''
LOSS, OPTIMIZER AND TRAINING HERE

loss function to compute mean pair-wise hinge loss across all (edge, non-edge) pairs
with a common incident node.

This is common form of ranking loss we have done in class (probably surrogate of AUC).

Reference:-
https://docs.dgl.ai/guide/training-link.html
'''

def compute_loss(pos_score, neg_score, priv_score, pos, neg, prv, N):
    # Margin loss
    tot_loss = 0

    # sampled_nodes = np.random.choice(N, size=800, replace=False)

    for i in range(N):
      pos_indices, neg_indices, priv_indices = pos[i], neg[i], prv[i]
      # sampled_pos_size = min(20, len(pos_indices))
      sampled_neg_size = min(400, len(pos_indices))
      # sampled_pos_indices = np.random.choice(pos_indices, size=sampled_pos_size, replace=False)
      sampled_neg_indices = np.random.choice(neg_indices, size=sampled_neg_size, replace=False)

      for j in pos_indices:
        for k in sampled_neg_indices:
          tot_loss += (1 + neg_score[k] - pos_score[j]).clamp(min=0)

      if i in private_nodes:
        priv_indices = np.random.choice(priv_indices, size=200, replace=False)

        for j in priv_indices:
          for k in pos_indices:
            if (pos_score[k] < priv_score[j]):
              tot_loss += (1 + priv_score[j] - pos_score[k]).clamp(min=0)/2

        for j in priv_indices:
          for k in sampled_neg_indices:
            if (neg_score[k] > priv_score[j]):
              tot_loss += (1 + neg_score[k] - priv_score[j]).clamp(min=0)/2
        
    loss_mean = tot_loss.mean()
    return loss_mean

## Node features must be float
node_features = graph.ndata['feat'] * 1.0
n_features = node_features.shape[1]
model = Model(n_features, 50, 50)
# model = torch.load('our_model_10')
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

validation_interval = 5 

## Budget and no. of edges queried till now
B = 200
edges_queried = 0

## Uncomment these lines to print status
# print('No. of edges = '+str(len(graph.edges()[0])))
# print('No. of non-edges = '+str(len(negative_graph.edges()[0])))
# print('No. of uncertain-pairs = '+str(len(private_graph.edges()[0])))

global_epoch_counter = 0
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


while edges_queried <= B:

  (graph_src, graph_dst) = graph.edges()
  print('Graph edges extracted')
  print(len(graph_src))
  (neg_graph_src, neg_graph_dst) = negative_graph.edges()
  print('Negative graph edges extracted')
  print(len(neg_graph_src))
  (priv_graph_src, priv_graph_dst) = private_graph.edges()
  print('Negative graph edges extracted')
  print(len(priv_graph_src))

  N = len(graph.nodes())

  pos, neg, prv = {}, {}, {}

  for i in range(N):
    pos[i], neg[i], prv[i] = [], [], [] 

  for i, node in enumerate(graph_src):
    pos[node.item()].append(i)
  for i, node in enumerate(neg_graph_src):
    neg[node.item()].append(i)
  for i, node in enumerate(priv_graph_src):
    prv[node.item()].append(i)

  print('Indices for each node extracted')
  # pos_score, test_score = model(graph, test_graph, node_features)
  # sampled_map, uncertain_map = validation(test_graph, test_score)
  # print('Sampled MAP = ' + str(sampled_map))
  # print('Uncertain MAP = ' + str(uncertain_map)+'\n')
  
  ## Training for 10 epochs, cross-validation after every 5 epochs after each query.
  for epoch in range(10):
      scheduler.step()
      print('Learning rate = ' + str(optimizer.param_groups[0]['lr']))
      global_epoch_counter += 1
      print('Epoch: '+str(epoch+1))
      # negative_graph = construct_negative_graph(graph, k)
      
      pos_score, neg_score = model(graph, negative_graph, node_features)
      pub_score, priv_score = model(graph, private_graph, node_features)
       
      loss = compute_loss(pos_score, neg_score, priv_score, pos, neg, prv, N)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      print(loss.item())

      if (epoch+1) % validation_interval == 0:
        torch.save(model, 'our_model_' + str(global_epoch_counter))
        print('Epoch: ' + str(global_epoch_counter))
        pos_score, test_score = model(graph, test_graph, node_features)
        sampled_map, uncertain_map = validation(test_graph, test_score, private_nodes) 
        print('Selected MAP = ' + str(sampled_map))
        print('Uncertain MAP = ' + str(uncertain_map)+'\n')

  # uncertain_pairs = select_random_pairs(model, graph, negative_graph, private_graph, node_features)
  uncertain_pairs = find_uncertain_edges(model, graph, negative_graph, private_graph, node_features, private_nodes)
  pos_src, pos_dst, neg_src, neg_dst, total_pairs = [], [], [], [], []

  '''
  Based on node-pairs chosen:-
  1. query the oracle
  2. remove the node-pairs from the private_graph
  3. if the oracle returns 1, add the node-pair to graph
  4  if the oracle returns 0, add the node-pair to negative_graph 
  '''

  for (u,v,loss) in uncertain_pairs:
    label = g.oracle[(u,v)]
    total_pairs.append((u,v))
    if label == 1:
      pos_src.append(u)
      pos_dst.append(v)
    else:
      neg_src.append(u)
      neg_dst.append(v)
    edges_queried += 1

  print('Oracle queries made')
    
  graph = dgl.add_edges(graph, pos_src, pos_dst)
  negative_graph = dgl.add_edges(negative_graph, neg_src, neg_dst)

  private_graph_nx = private_graph.to_networkx()
  private_graph_nx.remove_edges_from(total_pairs)
  private_graph = dgl.from_networkx(private_graph_nx)

  # Uncomment these lines to check if graphs modified properly after query
  # print('query status = ' +str(edges_queried)+'/'+str(B))
  # print('No. of edges = '+str(len(graph.edges()[0])))
  # print('No. of non-edges = '+str(len(negative_graph.edges()[0])))
  # print('No. of uncertain-pairs = '+str(len(private_graph.edges()[0])))
