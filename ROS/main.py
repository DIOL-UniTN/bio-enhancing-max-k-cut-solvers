import torch
import numpy as np
import networkx as nx
import logging
import sys
from utils import generate_graph, postprocess, set_random_seed
from add_parser import add_parse
from ros_vanilla.ros_vanilla import ros_vanilla 
from ros.ros import ros, ros_aco, ros_local_search, ros_aco_only, ros_ga, ga_only, ros_ga_improved, ros_mga,ros_abc

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    args = add_parse()
    set_random_seed(args.seed)
    args.TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.TORCH_DTYPE = torch.float32
    graph = generate_graph(args)
    
    if args.alg == "ros_vanilla":
        result = ros_vanilla(args, graph)
    elif args.alg == "ros":
        #  ROS standard (GNN + Random Sampling)
        result, result_relaxed = ros(args, graph)
    elif args.alg == "ros_aco":
        #  ROS ACO (GNN + Ant Colony Sampling)
        result, result_relaxed, phase_tracking = ros_aco(args, graph)
        # Clean up to avoid segfaults
        del result_relaxed, phase_tracking
    elif args.alg == "ros_local_search":
        result, result_relaxed = ros_local_search(args, graph)
    elif args.alg == "ros_aco_only":
        result, result_relaxed = ros_aco_only(args, graph)
    elif args.alg == "ros_ga":
        # ROS + Genetic Algorithm (GNN + GA with local search)
        result, result_relaxed = ros_ga(args, graph)
        del result_relaxed
    elif args.alg == "ros_ga_improved":
        # ROS + GA Improved (smart mutation + partition crossover)
        result, result_relaxed = ros_ga_improved(args, graph)
        del result_relaxed
    elif args.alg == "ros_abc":
        # ROS + Artificial Bee Colony
        result, result_relaxed = ros_abc(args, graph)
        del result_relaxed
    elif args.alg == "ga_only":
        # Pure Genetic Algorithm (without GNN, only GA + local search)
        result = ga_only(args, graph)
    elif args.alg == "ros_mga":
        #  ROS MGA (GNN + Memetic Genetic Algorithm with Speciation)
        result, result_relaxed = ros_mga(args, graph)
    elif args.alg == "from_file":
        result = torch.load(args.sol_dir)
    else:
        print(args.alg)
        raise NotImplementedError("Not Implemented Algorithm")
    
    if type(result) == float:
        if args.save:
            with open("./res/" + args.alg + "_value.txt", "a") as f:
                f.write("numpy.inf, ")
        print("FINAL RESULT: " + str(np.inf))
    else:    
        maxcut = postprocess(result, graph)
        if args.save:
            with open("./res/" + args.alg + "_value.txt", "a") as f:
                f.write(str(maxcut) + ", ")
        print("FINAL RESULT: " + str(maxcut))