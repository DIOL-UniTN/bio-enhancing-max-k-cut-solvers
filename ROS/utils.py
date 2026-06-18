import networkx as nx
import random
import numpy as np
import torch
import copy


def relabel_nodes_sorted_ordered(G):
    sorted_nodes = sorted(G.nodes())
    mapping = {old: new for new, old in enumerate(sorted_nodes)}
    G_temp = nx.relabel_nodes(G, mapping)
    
    # NetworkX >= 2.6 uses Graph() instead of OrderedGraph() (graphs are ordered by default now)
    G_new = nx.Graph()
    G_new.add_nodes_from(sorted(G_temp.nodes()))
    G_new.add_edges_from(G_temp.edges())
    return G_new

def generate_graph(args):
    if args.graph_type == 'reg':
        nx_graph = nx.random_regular_graph(d=args.d, n=args.n, seed=args.seed)
        nx_graph = relabel_nodes_sorted_ordered(nx_graph)
    elif args.graph_type in {'prob', 'erdos'}:
        nx_graph = nx.erdos_renyi_graph(args.n, args.p, seed=args.seed)
        nx_graph = relabel_nodes_sorted_ordered(nx_graph)
    elif args.graph_type == "gset":
        return read_gset(args)
    elif args.graph_type == "COLOR":
        nx_graph = read_COLOR(args)
    elif args.graph_type == "citeceer" or args.graph_type == "cora":
        nx_graph = read_citeceer_cora(args)
    elif args.graph_type == "bitcoin":
        # nx_graph = read_bitcoin(args)
        return read_bitcoin(args)
    elif args.graph_type == "read_from_file":
        return read_from_file(args) 

    else:
        raise NotImplementedError(f'Graph type {args.graph_type} not handled')
    
    # Assign uniform edge weights if applicable
    
    if args.weight_mode == 0:
        for u, v in nx_graph.edges():
            nx_graph[u][v]['weight'] = 1.0
    elif args.weight_mode == 1:
        for u, v in nx_graph.edges():
            nx_graph[u][v]['weight'] = np.random.choice([-1, 1])
    elif args.weight_mode == 2:
        for u, v in nx_graph.edges():
            nx_graph[u][v]['weight'] = np.random.rand() * (args.rrange - args.lrange) + args.lrange

    return nx_graph

def read_from_file(args):
    """Reads a graph from file."""
    path = args.graph_path
    Gset = True if args.Gset == "True" else False

    edges = []
    max_node = -1
    with open(path, 'r') as fh:
        for i,line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                # riga non valida, ignorala
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                # se gli id non sono interi, ignora la riga
                continue
            w=1.0
            if len(parts) >= 3:
                w = float(parts[2])
            if i == 0 and Gset:
                n = int(parts[0])
                continue
            edges.append((u, v, w))
            if u > max_node:
                max_node = u
            if v > max_node:
                max_node = v

    n = max_node + 1 if max_node >= 0 else 0

    args.n = n
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for u, v, w in edges:
        graph.add_edge(u, v, weight=w)
    return graph


def read_bitcoin(args):
    """Reads a bitcoin graph from file."""
    nx_graph = nx.Graph()
    with open(f"./instance/{args.graph_type}/graph.txt") as f:
        cnt = 0
        for line in f:
            i, j, w= map(int, line.split()[:3])
            if args.weight_mode == 2:
                if nx_graph.has_edge(i, j):
                    nx_graph[i][j]['weight'] += w
                else:
                    nx_graph.add_edge(i, j, weight=w)
            else:
                raise "Weight_Mode not support in Bitcoin Dataset"
    mapping = {old: new for new, old in enumerate(sorted(nx_graph.nodes()), start=0)}
    nx_graph = nx.relabel_nodes(nx_graph, mapping)
    args.n = len(nx_graph.nodes())   
    return nx_graph


def read_citeceer_cora(args):
    """Reads a citeceer or cora graph from file."""
    nx_graph = nx.Graph()
    with open(f"./instance/{args.graph_type}/graph.txt") as f:
        num_nodes = int(next(f).split()[0])
        args.n = num_nodes
        nx_graph.add_nodes_from(range(num_nodes))
        for line in f:
            i, j= map(int, line.split()[:2])
            if args.weight_mode == 0:
                nx_graph.add_edge(i, j, weight=1)
            elif args.weight_mode == 1:
                nx_graph.add_edge(i, j, weight=np.random.choice([-1, 1]))
            elif args.weight_mode == 2:
                nx_graph.add_edge(i, j, weight=np.random.rand() * (args.rrange - args.lrange) + args.lrange)
    return nx_graph


def read_COLOR(args):
    """Reads a COLOR graph from file."""
    nx_graph = nx.Graph()
    with open(f"./instance/COLOR/{args.gset}.col") as f:
        for line in f:
            tokens = line.split()
            if tokens[0] == "p":
                args.n = int(tokens[2])
                nx_graph.add_nodes_from(range(args.n))
            elif tokens[0] == "e":
                nx_graph.add_edge(int(tokens[1]) - 1, int(tokens[2]) - 1)
    return nx_graph

def read_gset(args):
    """Reads a GSET graph from file."""
    nx_graph = nx.Graph()
    with open(f"../Gset/G{args.gset}") as f:
        num_nodes = int(next(f).split()[0])
        args.n = num_nodes
        nx_graph.add_nodes_from(range(num_nodes))
        for line in f:
            i, j, w = map(int, line.split()[:3])
            if args.weight_mode == 1:
                nx_graph.add_edge(i - 1, j - 1, weight=float(w))
            elif args.weight_mode == 2:
                nx_graph.add_edge(i - 1, j - 1, weight=float(w) *((args.rrange - args.lrange) * np.random.rand() + args.lrange))
                # nx_graph[i-1][j-1]['weight']=float(w) *((args.rrange - args.lrange) * np.random.rand() + args.lrange)
            elif args.weight_mode == 0:
                raise "Mode 0 can not be used in GSET as the sign is given by the Gset it self."
    return nx_graph


def postprocess(result, graph):
    """
    helper function to postprocess MIS results

    Input:
        best_bitstring: bitstring as torch tensor
    Output:
        size_mis: Size of MIS (int)
        ind_set: MIS (list of integers)
        number_violations: number of violations of ind.set condition
    """
    if isinstance(result, torch.Tensor):
        result = result.detach().cpu().numpy()
        
    maxcut = 0
    for (u, v, val) in graph.edges(data=True):
        wt = val['weight']
        if result[u] != result[v]:
            maxcut = maxcut + wt    
    return maxcut


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)  # 设置 Python 内置随机库的种子
    np.random.seed(seed)  # 设置 NumPy 随机库的种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机库的种子
    torch.cuda.manual_seed(seed)  # 为当前 CUDA 设备设置种子
    torch.cuda.manual_seed_all(seed)  # 为所有 CUDA 设备设置种子


def get_matrix(args, nx_G):
    n_nodes = args.n
    W_mat = np.zeros((n_nodes, n_nodes))
    for (u, v, val) in nx_G.edges(data = True):
        W_mat[u][v] = val['weight'] / 2
        W_mat[v][u] = val['weight'] / 2
    return W_mat

def simulate_ros():
    from ROS.ros.ros import ros
    from ROS.add_parser import add_parse 

    args = add_parse()

    set_random_seed(args.seed)
    args.TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.TORCH_DTYPE = torch.float32

    graph = generate_graph(args)

    if args.alg == "ros":
        result, result_relaxed = ros(args, graph)
    else:
        raise "Not Implemented Algorithm"
    
    return result, result_relaxed