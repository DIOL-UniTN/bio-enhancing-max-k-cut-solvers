
from .models import GCN_dev
import torch.nn as nn
import torch
from itertools import chain
import time
import numpy
import torch.nn.init as init

def get_matrix(args, nx_G):
    n_nodes = args.n
    W_mat = torch.zeros(n_nodes, n_nodes).type(args.TORCH_DTYPE).to(args.TORCH_DEVICE)
    for (u, v, val) in nx_G.edges(data = True):
        W_mat[u][v] = val['weight'] / 2
        W_mat[v][u] = val['weight'] / 2
    return W_mat


def get_gnn(args, graph):
    net = GCN_dev(args.dim_embedding, args.hidden_dim, args.k, args.TORCH_DEVICE)
    net = net.type(args.TORCH_DTYPE).to(args.TORCH_DEVICE)
    inputs = nn.Embedding(args.n, args.dim_embedding).to(args.TORCH_DEVICE)
    init.xavier_uniform_(inputs.weight)
    params = chain(inputs.parameters(), net.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    edges = [[], []]
    edges_weight = []
    for (u, v, w) in graph.edges(data=True):
        val = w['weight']
        edges[0].append(u)
        edges[1].append(v)
        edges_weight.append(val)
        edges[0].append(v)
        edges[1].append(u)
        edges_weight.append(val)
    edges = torch.tensor(edges).to(args.TORCH_DEVICE)
    edges_weight = torch.tensor(edges_weight).type(args.TORCH_DTYPE).to(args.TORCH_DEVICE)
    return net, optimizer, edges, edges_weight, inputs


def run_gnn_training(args, W, graph, edges, edges_weight, net, optimizer, inputs):
    prev_loss = numpy.inf
    best_loss = numpy.inf
    count = 0
    t_gnn_start = time.time()
    for epoch in range(args.epochs):
        probs = net(inputs.weight, edges, edges_weight)
        loss = torch.trace(probs @ W @ probs.T)
        loss_item = loss.detach().item()
        if loss_item < best_loss:
            best_loss = loss_item
            best_solution_relaxed = probs
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, Loss: {W.sum().item() - loss_item}, Patience: {count}/{args.patience}')
        if (abs(loss_item - prev_loss)<=args.tol) | ((loss_item - prev_loss) > 0):
            count += 1
        else:
            count = 0
        if count >= args.patience:
            print(f"stopping early on epoch {epoch} (patience: {args.patience})")
            break
        prev_loss = min(loss_item, prev_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    t_gnn = time.time() - t_gnn_start
    print(f'GNN training (n={graph.number_of_nodes()}) took {round(t_gnn, 3)}')
    print(f'GNN best continuous loss: {best_loss}')
    return best_solution_relaxed


def sample_one_hot(prob_matrix):
    prob_matrix = prob_matrix.T
    sampled_indices = torch.multinomial(prob_matrix, 1)
    
    one_hot_matrix = torch.zeros_like(prob_matrix)
    
    one_hot_matrix.scatter_(1, sampled_indices, 1)
    
    return one_hot_matrix.T