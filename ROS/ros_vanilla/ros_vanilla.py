import time
from .utils import get_gnn, run_gnn_training, get_matrix, sample_one_hot
import torch


def ros_vanilla(args, graph):
    W = get_matrix(args, graph)    
    gnn_start = time.time()
    net, optimizer, edges, edges_weight, inputs = get_gnn(args, graph)
    best_solution_relaxed = run_gnn_training(args, W, graph, edges, edges_weight, net, optimizer, inputs)
    Expectation = torch.trace(best_solution_relaxed @ W @ best_solution_relaxed.T)
    print("Expectation = " + str((W.sum() - Expectation).item()))
    best_val = torch.inf
    for _ in range(args.max_iter):
        Xt = sample_one_hot(best_solution_relaxed)
        val = torch.trace(Xt @ W @ Xt.T)
        if val < best_val:
            best_val = val
            best_solution = Xt
    best_solution = best_solution.argmax(axis=0)
    gnn_time = time.time() - gnn_start
    if args.save:
        with open("./res/ros_vanilla_time.txt", "a") as ff:
            ff.write(str(round(gnn_time, 3)) + ", ")
    return best_solution 
    
