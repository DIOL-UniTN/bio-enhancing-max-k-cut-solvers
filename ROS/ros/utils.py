import numpy as np
from .models import GCN_dev
import torch.nn as nn
import torch
from itertools import chain
import networkx as nx
import time
import torch.nn.init as init

import torch.optim as optim


def get_matrix(args, nx_G):
    n_nodes = args.n
    W_mat = torch.zeros(n_nodes, n_nodes).type(args.TORCH_DTYPE).to(args.TORCH_DEVICE)
    for (u, v, val) in nx_G.edges(data = True):
        W_mat[u][v] = val['weight'] / 2
        W_mat[v][u] = val['weight'] / 2
    return W_mat


def get_gnn(args, torch_device, torch_dtype):
    net = GCN_dev(args.dim_embedding, args.hidden_dim, args.k)
    net = net.type(torch_dtype).to(torch_device)
    params = net.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    lr_lambda = lambda epoch: 1.0 if epoch < args.pretraining_epochs * args.pretraining_graphnum // 3 else ( 0.1 if epoch < 2 *  args.pretraining_epochs * args.pretraining_graphnum // 3 else 0.01)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return net, optimizer, scheduler


def get_gnn_tuning(args, graph):
    net = GCN_dev(args.dim_embedding, args.hidden_dim, args.k)
    net = net.type(args.TORCH_DTYPE).to(args.TORCH_DEVICE)
    inputs = nn.Embedding(args.n, args.dim_embedding,device=args.TORCH_DEVICE)
    init.xavier_uniform_(inputs.weight)
    params = chain(net.parameters(), inputs.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    edges_index = [[], []]
    edges_weight = []
    for (u, v, w) in graph.edges(data=True):
        val = w['weight']
        edges_index[0].append(u)
        edges_index[1].append(v)
        edges_weight.append(val)
        edges_index[0].append(v)
        edges_index[1].append(u)
        edges_weight.append(val)
    edges_index = torch.tensor(edges_index).to(args.TORCH_DEVICE)
    edges_weight = torch.tensor(edges_weight).type(args.TORCH_DTYPE).to(args.TORCH_DEVICE)
    return net, optimizer, edges_index, edges_weight, inputs


def run_gnn_tuning(args, W, edges, edges_weight, net, optimizer, number_epochs, tol, patience, inputs):
    prev_loss = np.inf
    best_loss = np.inf
    count = 0
    t_gnn_start = time.time()
    for epoch in range(number_epochs):
        probs = net(inputs.weight, edges, edges_weight)
        loss = torch.trace(probs @ W @ probs.T)
        loss_item = loss.detach().item()
        if loss_item < best_loss:
            best_loss = loss_item
            best_solution_relaxed = probs
        #if epoch % 1000 == 0:
            #print("Epoch " + str(epoch) + ": " + str(W.sum().item() - loss_item) + " " + str(count))
        if (abs(loss_item - prev_loss) <= tol) | ((loss_item - prev_loss) > 0):
            count += 1
        else:
            count = 0
        if count >= patience:
            #print(f"stopping early on epoch {epoch} (patience: {patience})")
            break
        prev_loss = min(prev_loss, loss_item)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    t_gnn = time.time() - t_gnn_start
    #print(f'GNN training (Epoch={epoch}) took {round(t_gnn, 3)}')
    return best_solution_relaxed


def sample_one_hot(prob_matrix):
    prob_matrix = prob_matrix.T
    sampled_indices = torch.multinomial(prob_matrix, 1)
    one_hot_matrix = torch.zeros_like(prob_matrix)
    one_hot_matrix.scatter_(1, sampled_indices, 1)
    return one_hot_matrix.T


def pretraining(args, net, optimizer, scheduler, number_epochs, number_graphs):
    t_gnn_start = time.time()
    for epoch in range(number_epochs):
        loss_itemsum = 0
        for gn in range(number_graphs):
            G = nx.random_regular_graph(d=args.d, n=args.n, seed = 100000 + gn)
            edge_index, edge_weights, adjacency_matrix = get_index_weight_adj(args, G)
            inputs = torch.rand((args.n, args.dim_embedding)).to(args.TORCH_DEVICE)
            init.xavier_uniform_(inputs)
            outputs = net(inputs, edge_index, edge_weights)
            loss = torch.trace(outputs@adjacency_matrix@outputs.T)
            loss_itemsum = loss_itemsum + loss.item()
            if gn % 10 == 0:
                print("gn " + str(gn) + ": " + str(loss.item()) + " lr: " + str(optimizer.param_groups[0]['lr']))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        print("Epoch " + str(epoch) + ": " + str(loss_itemsum / number_graphs) + " lr: " + str(optimizer.param_groups[0]['lr']))
    t_gnn = time.time() - t_gnn_start
    print(f'GNN training (Epoch={args.n}) took {round(t_gnn, 3)}')
    torch.save(net.state_dict(), "gcn_model_ood_k" + str(args.k) + ".pth")
    return net


def get_index_weight_adj(args, G):
    edge_index = [[], []]
    edge_weights = []
    n = args.n
    adjacency_matrix = torch.zeros((n, n)).type(args.TORCH_DTYPE).to(args.TORCH_DEVICE)
    for u, v in G.edges():
        edge_index[0].append(u)
        edge_index[1].append(v)
        edge_index[0].append(v)
        edge_index[1].append(u)
        edge_weights.append(1.)
        edge_weights.append(1.)
        adjacency_matrix[u][v] = 0.5
        adjacency_matrix[v][u] = 0.5
    edge_index = torch.tensor(edge_index).to(args.TORCH_DEVICE)
    edge_weights = torch.tensor(edge_weights).to(args.TORCH_DEVICE).type(args.TORCH_DTYPE)
    return edge_index, edge_weights, adjacency_matrix

