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
VALIDATION (INFERENCE) ROUTINE HERE

Takes graph and score for all test node-pairs as arguments
and computes MAP for all private nodes.

Used for cross-validation during training of LP model 
'''

def validation(graph, score, private_nodes):
  (graph_src, graph_dst) = graph.edges()
  edge_labels = graph.edata['label']

  N = len(graph.nodes())

  ranklists, labels = {}, {}
  for node in range(N):
    ranklists[node] = []

  for i in range(len(graph_src)):
    ranklists[graph_src[i].item()].append((score[i].item(), edge_labels[i].item()))
    ranklists[graph_dst[i].item()].append((score[i].item(), edge_labels[i].item()))

  total_score, val_ct = 0, 0

  ## large degree nodes (degree>=10) hard-coded here
  large_degree_nodes = [0, 28, 121, 71, 15, 2, 84, 328, 27, 72, 47, 122, 379, 13, 87, 173, 222, 52, 198, 333, 809, 236, 355, 279, 464, 3, 48, 64, 68, 91, 182, 543, 85, 131, 240, 509, 822, 1217, 200, 535, 642, 643]
  
  # sampled_nodes = np.random.choice(N, size=100, replace=False)
  for node in large_degree_nodes:
    ranklists[node] = np.array(sorted(ranklists[node], key = lambda x: x[0], reverse=True))
    ground_truth_labels = ranklists[node][:,1]
    one_cnt = np.sum(ground_truth_labels)
    zero_cnt = np.sum(1-ground_truth_labels)
    if one_cnt>0 and zero_cnt>0:
      avp = np.sum((np.cumsum(ground_truth_labels)*ground_truth_labels)/np.arange(1,len(ground_truth_labels)+1))/one_cnt
      total_score+=avp
      val_ct+=1
  sampled_map = total_score/val_ct

  total_score, val_ct = 0, 0

  for node in private_nodes:
    ranklists[node] = np.array(sorted(ranklists[node], key = lambda x: x[0], reverse=True))
    ground_truth_labels = ranklists[node][:,1]
    one_cnt = np.sum(ground_truth_labels)
    zero_cnt = np.sum(1-ground_truth_labels)
    if one_cnt>0 and zero_cnt>0:
      avp = np.sum((np.cumsum(ground_truth_labels)*ground_truth_labels)/np.arange(1,len(ground_truth_labels)+1))/one_cnt
      total_score+=avp
      val_ct+=1
  uncertain_map = total_score/val_ct
  return sampled_map, uncertain_map

'''
NODE-PAIR TO QUERY FOR ACTIVE LEARNING

Objective: Find node-pair from pool-set which maximizes expected value of a separate loss.
Note: This loss is not directly a surrogate for MAP.
'''

def select_random_pairs(model, graph, negative_graph, private_graph, node_features):
  (priv_graph_src, priv_graph_dst) = private_graph.edges()
  priv_pair_count = len(priv_graph_src)

  indices = np.random.choice(priv_pair_count, size=40, replace=False)
  querying_pairs = [(priv_graph_src[index].item(), priv_graph_dst[index].item(), None) for index in indices]

  return querying_pairs

def find_uncertain_edges(model, graph, negative_graph, private_graph, node_features, private_nodes):
  pub_score, neg_score = model(graph, negative_graph, node_features)
  pub_score, priv_score = model(graph, private_graph, node_features)
  (pos_graph_src, pos_graph_dst) = graph.edges()
  (neg_graph_src, neg_graph_dst) = negative_graph.edges()
  (priv_graph_src, priv_graph_dst) = private_graph.edges()

  ranklists, unc_ranklists = {}, {}
  for node in private_nodes:
    ranklists[node], unc_ranklists[node] = [], []

  ## add the scores for labelled pairs to the ranklist of respective node

  for i in range(len(pos_graph_src)):
    if pos_graph_src[i].item() in private_nodes:
      ranklists[pos_graph_src[i].item()].append((pub_score[i].item(), 1))

  for i in range(len(neg_graph_src)):
    if neg_graph_src[i].item() in private_nodes:
      ranklists[neg_graph_src[i].item()].append((neg_score[i].item(), 0))

   ## add the scores of uncertain pairs to the unc_ranklist of respective node 

  for i in range(len(priv_graph_src)):
    if priv_graph_src[i].item() in private_nodes:
      unc_ranklists[priv_graph_src[i].item()].append((priv_graph_dst[i].item(), priv_score[i].item()))

  for node in private_nodes:
    ranklists[node] = sorted(ranklists[node], key=lambda x: x[0], reverse=True)
    unc_ranklists[node] = sorted(unc_ranklists[node], key=lambda x: x[1], reverse=True)

  sorted_queryable = []
  k = int(40/len(private_nodes)) + 1
  '''
  For each private node, calculate the loss for nodes with 3 highest and 3 lowest scores. Because
  of the nature of the Loss function, this heuristic is expected to works and reduces the
  complexity from O(|V|^3) -> O(|V|^2)
  '''

  for node in private_nodes:
    # sorted_pool = sorted(unc_ranklists[node],key= lambda x: x[1],reverse=True)
    choices = unc_ranklists[node] #set([sorted_pool[0], sorted_pool[1], sorted_pool[2], sorted_pool[-1], sorted_pool[-2], sorted_pool[-3]])
    # print(choices)
    # exit()
    pool_pair_losses = []

    for (j, (cur_index, cur_score)) in enumerate(choices):
      if cur_index == node:
        continue
      loss = 0
      tmp = 0
      for (i, (score, index)) in enumerate(unc_ranklists[node]):
        min_pos = min(i, j) + 1
        loss += np.abs(score - cur_score)/4
        # print(loss)
        tmp = loss
      # exit() 
      for (i, (score, label)) in enumerate(ranklists[node]):
        min_pos = min(i, j) + 1
        if score > cur_score and label == 0:
          loss += (score - cur_score)/2
        elif score < cur_score and label == 1:
          loss += (cur_score - score)/2
      # print(node, "intra-inter",tmp, loss-tmp)
      pool_pair_losses.append((node, cur_index, loss))

    per_node_queryable = sorted(pool_pair_losses, key= lambda x: x[2], reverse=True)[:k]
    sorted_queryable += per_node_queryable
  # print(sorted_queryable[:40])
  return sorted_queryable[:40]