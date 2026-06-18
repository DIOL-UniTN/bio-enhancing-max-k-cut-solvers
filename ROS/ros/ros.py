import time
import torch
import numpy as np
import json
import os
import gc
from datetime import datetime
from .utils import get_gnn_tuning, get_matrix, run_gnn_tuning


# UTILITY FUNCTIONS 
def discrete_crossover_uniform(parent1, parent2):
    #Uniform crossover
    mask = torch.rand(len(parent1)) > 0.5
    child = parent1.clone()
    child[~mask] = parent2[~mask]
    return child

def discrete_crossover_cluster_preserving(parent1, parent2, W, k):
    # Cluster-preserving crossover: inherits entire clusters from one parent
    
    n = len(parent1)
    child = parent1.clone()
    
    # For each cluster in parent2, decide whether to copy it entirely
    for cluster_id in range(k):
        if torch.rand(1).item() < 0.5:
            # Copy all nodes of this cluster from parent2
            mask = (parent2 == cluster_id)
            child[mask] = parent2[mask]
    
    return child


def path_relinking(sol1, sol2, W, k, total_weight, max_steps=20):
    #Path Relinking: constructs a path from sol1 to sol2, exploring intermediate solutions that may be better
    
    current = sol1.clone()
    n = len(current)
    
    # Find differences between sol1 and sol2
    diff_nodes = (current != sol2).nonzero(as_tuple=True)[0]
    
    if len(diff_nodes) == 0:
        return current, total_weight - torch.trace(
            torch.nn.functional.one_hot(current, num_classes=k).T.float() @ W @ 
            torch.nn.functional.one_hot(current, num_classes=k).T.float().T
        ).item()
    
    best_solution = current.clone()
    X = torch.nn.functional.one_hot(current, num_classes=k).T.float()
    best_cut = total_weight - torch.trace(X @ W @ X.T)
    
    # Move towards sol2 one node at a time, choosing the best move
    steps = min(max_steps, len(diff_nodes))
    remaining_diffs = diff_nodes.tolist()
    
    for step in range(steps):
        if not remaining_diffs:
            break
        
        best_move_node = None
        best_move_cut = -float('inf')
        
        # Try each possible move towards sol2
        for node in remaining_diffs[:min(10, len(remaining_diffs))]:  # Limit for speed
            # Try changing this node towards sol2
            test_sol = current.clone()
            test_sol[node] = sol2[node]
            
            X_test = torch.nn.functional.one_hot(test_sol, num_classes=k).T.float()
            cut = total_weight - torch.trace(X_test @ W @ X_test.T)
            
            if cut > best_move_cut:
                best_move_cut = cut
                best_move_node = node
        
        if best_move_node is not None:
            current[best_move_node] = sol2[best_move_node]
            remaining_diffs.remove(best_move_node)
            
            if best_move_cut > best_cut:
                best_cut = best_move_cut
                best_solution = current.clone()
    
    return best_solution, best_cut.item()


# LOCAL SEARCH HELPERS
def local_search_2opt(assignment, W, k, total_weight, max_iters=50):
    #Local search 2-opt: tries to change the assignment of individual nodes to improve the cut
    
    device = W.device
    n = len(assignment)
    best_assignment = assignment.clone()
    
    # Calculate initial cut
    X = torch.nn.functional.one_hot(best_assignment, num_classes=k).T.float()
    best_cut = total_weight - torch.trace(X @ W @ X.T)
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # Try changing each node
        for i in range(n):
            current_cluster = best_assignment[i].item()
            
            # Try all other clusters
            for new_cluster in range(k):
                if new_cluster == current_cluster:
                    continue
                
                # Create new assignment
                test_assignment = best_assignment.clone()
                test_assignment[i] = new_cluster
                
                # Calculate new cut
                X_test = torch.nn.functional.one_hot(test_assignment, num_classes=k).T.float()
                new_cut = total_weight - torch.trace(X_test @ W @ X_test.T)
                
                # If improves, accept
                if new_cut > best_cut:
                    best_cut = new_cut
                    best_assignment = test_assignment
                    improved = True
                    break  # Move to the next node
            
            if improved:
                break  # Restart from the beginning
    
    return best_assignment, best_cut

def local_search_swap(assignment, W, k, total_weight, max_iters=30):
    #Local search with pair swaps: tries to swap the assignments of two nodes
    
    device = W.device
    n = len(assignment)
    best_assignment = assignment.clone()
    
    X = torch.nn.functional.one_hot(best_assignment, num_classes=k).T.float()
    best_cut = total_weight - torch.trace(X @ W @ X.T)
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # Try swapping pairs of nodes
        for i in range(n):
            for j in range(i + 1, min(i + 50, n)):  # Limit search for efficiency
                if best_assignment[i] == best_assignment[j]:
                    continue
                
                # Swap assignments
                test_assignment = best_assignment.clone()
                test_assignment[i], test_assignment[j] = test_assignment[j].item(), test_assignment[i].item()
                
                # Calculate new cut
                X_test = torch.nn.functional.one_hot(test_assignment, num_classes=k).T.float()
                new_cut = total_weight - torch.trace(X_test @ W @ X_test.T)
                
                if new_cut > best_cut:
                    best_cut = new_cut
                    best_assignment = test_assignment
                    improved = True
                    break
            
            if improved:
                break
    
    return best_assignment, best_cut

def calculate_pheromone_entropy(tau):
    # Calculate the entropy of pheromones to measure diversification
   
    # Normalize pheromones
    tau_norm = tau / (tau.sum(dim=1, keepdim=True) + 1e-12)
    # Calculate entropy
    entropy = -(tau_norm * torch.log(tau_norm + 1e-12)).sum(dim=1).mean()
    return entropy

# 1. ROS STANDARD 
def ros(args, graph):
    W = get_matrix(args, graph)

    net, optimizer, edges, edges_weight, inputs = get_gnn_tuning(args, graph)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.load_state_dict(torch.load(
        f"gcn_model_ood_k{args.k}.pth",
        map_location=device,
        weights_only=True
    ))

    print("[ROS] Running GNN...")
    X_relaxed = run_gnn_tuning(
        args, W, edges, edges_weight,
        net, optimizer,
        args.epochs, args.tol, args.patience, inputs
    )

    best_val = torch.inf
    best_X = None

    print("[ROS] Random sampling...")
    sampling_start = best_val.item() if best_val != torch.inf else 0
    
    for it in range(args.max_iter):
        Xt = torch.nn.functional.one_hot(
            torch.multinomial(X_relaxed.T, 1).squeeze(),
            num_classes=args.k
        ).T.float()

        val = torch.trace(Xt @ W @ Xt.T)
        if val < best_val:
            best_val = val
            best_X = Xt

        if it % max(1, args.max_iter // 5) == 0:
            print(f"[ROS] iter {it}/{args.max_iter} | best = {best_val:.4f}")
    
    sampling_end = best_val.item()
    print(f"[ROS] Sampling improvement: {sampling_start - sampling_end:.4f}")
    print(f"[ROS] X_relaxed = {X_relaxed}")
    return best_X.argmax(dim=0), X_relaxed

def ros_aco(args, graph,
            n_ants=20,
            aco_iters=30,
            alpha=1.0,
            beta=2.0,
            rho=0.3,
            Q=50.0,
            elite_ratio=0.2,
            local_search_freq=5,
            restart_threshold=40,
            perturbation_threshold=20):

    # Global to prevent garbage collection issues
    global _ros_aco_cache
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Track timing for each phase
    global_start_time = time.time()
    
    #  GNN 
    ros_start_time = time.time()
    W = get_matrix(args, graph)

    net, optimizer, edges, edges_weight, inputs = get_gnn_tuning(args, graph)
    net.load_state_dict(torch.load(
        f"gcn_model_ood_k{args.k}.pth",
        map_location=device,
        weights_only=True
    ))

    print("[ROS-ACO] Running GNN...")
    X_relaxed = run_gnn_tuning(
        args, W, edges, edges_weight,
        net, optimizer,
        args.epochs, args.tol, args.patience, inputs
    )  # (k, n)
    
    ros_end_time = time.time()
    ros_elapsed = ros_end_time - ros_start_time
    print(f"[TIMING] ROS (GNN) phase: {ros_elapsed:.2f}s")

    k, n = X_relaxed.shape
    
    # Calculate total_weight first
    total_weight = W.sum()
    
    # Sample 5 initial solutions from GNN and keep the best
    print("[ROS-ACO] Sampling 5 initial solutions from GNN...")
    best_initial_cut = -torch.inf
    best_initial_assignment = None
    
    for sample_idx in range(5):
        # Sample an assignment from the GNN probabilities
        sampled_assignment = torch.multinomial(X_relaxed.T, 1).squeeze()
        X_sampled = torch.nn.functional.one_hot(sampled_assignment, num_classes=k).T.float()
        sampled_cut = total_weight - torch.trace(X_sampled @ W @ X_sampled.T)
        
        if sampled_cut > best_initial_cut:
            best_initial_cut = sampled_cut
            best_initial_assignment = sampled_assignment
    
    initial_assignment = best_initial_assignment
    print(f"[ROS-ACO] Best GNN sampled cut (out of 5): {best_initial_cut.item():.4f}")


    #  Initialize pheromones 
    tau = torch.ones(n, k, device=device) * 1.0
    tau = tau + X_relaxed.T * 0.1  # Reduced from 0.5 to 0.1 for less GNN bias
    tau = torch.clamp(tau, min=0.01, max=100.0)

    # Initialize with the best GNN solution
    best_cut = best_initial_cut
    best_assign = best_initial_assignment.clone()
    
    n_elite = max(1, int(n_ants * elite_ratio))
    temperature = 5.0
    no_improvement_count = 0
    restart_count = 0
    
    # Adaptive parameters
    current_alpha = alpha
    current_beta = beta
    current_rho = rho
    
    # Best solution tracking
    best_history = []
    
    phase_tracking = {
        'initial_sampling': {'start': 0, 'end': None, 'iterations': []},
        'pure_aco': {'start': None, 'end': None, 'iterations': []},
        'local_search': {'start': None, 'end': None, 'applications': []},
        'restart_perturbation': {'start': None, 'end': None, 'events': []}
    }

    # Track ACO and Local Search timing
    aco_start_time = time.time()
    local_search_total_time = 0.0
    
    print("[ROS-ACO] Starting ACO refinement...")

    for it in range(aco_iters):

        heuristic = torch.zeros(n, k, device=device)
        
        # For each node and cluster, estimate how much it would contribute to the cut
        for c in range(k):
            # Weight of connections to all nodes
            for other_c in range(k):
                if other_c != c:
                    # Probability that other nodes are in other_c
                    prob_other = tau[:, other_c]
                    # Weighted contribution
                    heuristic[:, c] += (W @ prob_other)
        
        heuristic = torch.clamp(heuristic, min=1e-6)
        heuristic = heuristic / (heuristic.sum(dim=1, keepdim=True) + 1e-12)

        solutions = []
        cuts = []

        for ant in range(n_ants):
            if ant < n_ants // 2 and best_assign is not None:
                # First half: PERTURB the best solution
                c = best_assign.clone()
                # Randomly change between 5% and 20% of the nodes
                n_changes = int(n * (0.05 + torch.rand(1).item() * 0.15))
                nodes_to_change = torch.randperm(n)[:n_changes]
                
                for node in nodes_to_change:
                    # Choose new cluster based on pheromones and heuristic
                    P = ((tau[node] + 1e-6) ** current_alpha) * ((heuristic[node] + 1e-6) ** current_beta)
                    P = P ** (1.0 / temperature)
                    P = P / (P.sum() + 1e-12)
                    c[node] = torch.multinomial(P, 1).item()
            else:
                # Second half: Normal construction with different strategies
                if ant < 2 * n_ants // 3:
                    # Strategy 1: Strong heuristic
                    P = ((tau + 1e-6) ** 0.5) * ((heuristic + 1e-6) ** 3.0)
                elif ant < 5 * n_ants // 6:
                    # Strategy 2: Balanced  
                    P = ((tau + 1e-6) ** current_alpha) * ((heuristic + 1e-6) ** current_beta)
                else:
                    # Strategy 3: More random for exploration
                    P = ((tau + 1e-6) ** 0.3) * ((heuristic + 1e-6) ** 1.0)
                
                P = P ** (1.0 / temperature)
                P = torch.clamp(P, min=1e-12)
                P = P / (P.sum(dim=1, keepdim=True) + 1e-12)
                c = torch.multinomial(P, 1).squeeze()  # (n,)

            if c.dim() == 0:
                c = c.unsqueeze(0)
            
            X = torch.nn.functional.one_hot(c, num_classes=k).T.float()

            val = torch.trace(X @ W @ X.T)
            cut = total_weight - val

            solutions.append(c)
            
            cuts.append(cut)

        cuts = torch.stack(cuts)
        
        if it % 10 == 0:
            print(f"  [DEBUG] Cuts range: min={cuts.min():.2f}, max={cuts.max():.2f}, mean={cuts.mean():.2f}, std={cuts.std():.2f}")
        
        # More aggressive periodic local search
        if it % local_search_freq == 0 and it > 0:
            ls_start = time.time()
            
            if phase_tracking['local_search']['start'] is None:
                phase_tracking['local_search']['start'] = best_cut.item()
                print(f"[TRACKING] Phase 3: Local search starts with cut = {best_cut:.4f}")
            
            top_k_indices = torch.argsort(cuts, descending=True)[:min(5, n_ants)]
            for idx in top_k_indices:
                before_ls = cuts[idx].item()
                # First 2-opt
                improved_assign, improved_cut = local_search_2opt(
                    solutions[idx], W, k, total_weight, max_iters=10
                )
                # Then swap if it makes sense
                if improved_cut > cuts[idx]:
                    improved_assign, improved_cut = local_search_swap(
                        improved_assign, W, k, total_weight, max_iters=10
                    )
                
                if improved_cut > cuts[idx]:
                    after_ls = improved_cut.item()
                    phase_tracking['local_search']['applications'].append({
                        'before': before_ls,
                        'after': after_ls,
                        'improvement': after_ls - before_ls
                    })
                    solutions[idx] = improved_assign
                    cuts[idx] = improved_cut
            
            ls_end = time.time()
            local_search_total_time += (ls_end - ls_start)
        
        # Update best
        iter_best_idx = torch.argmax(cuts)
        if cuts[iter_best_idx] > best_cut:
            if best_cut == -torch.inf:
                phase_tracking['initial_sampling']['end'] = cuts[iter_best_idx].item()
                phase_tracking['pure_aco']['start'] = cuts[iter_best_idx].item()
                print(f"[TRACKING] Phase 2: Pure ACO starts with cut = {cuts[iter_best_idx]:.4f}")
            best_cut = cuts[iter_best_idx]
            best_assign = solutions[iter_best_idx].clone()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        best_history.append(best_cut.item())
        phase_tracking['pure_aco']['iterations'].append({
            'iter': it,
            'best_cut': best_cut.item()
        })
        
        # Perturbation if stuck
        if no_improvement_count == perturbation_threshold:
            if phase_tracking['restart_perturbation']['start'] is None:
                phase_tracking['restart_perturbation']['start'] = best_cut.item()
                print(f"[TRACKING] Phase 4: Restart/Perturbation starts")
            
            print(f"[ROS-ACO] Applying perturbation...")
            phase_tracking['restart_perturbation']['events'].append({
                'type': 'perturbation',
                'iteration': it,
                'cut_before': best_cut.item()
            })
            # Temporarily increase exploration
            current_beta = max(0.1, beta * 0.5)
            current_rho = min(0.3, rho * 2.0)
            # Add noise to pheromones
            tau += torch.rand_like(tau) * 0.5
        elif no_improvement_count > perturbation_threshold:
            # Gradually return to normal values
            current_beta = min(beta, current_beta * 1.05)
            current_rho = max(rho, current_rho * 0.95)
        
        # Calculate entropy for diversification
        entropy = calculate_pheromone_entropy(tau)
        
        # If entropy is low, increase exploration
        if entropy < 0.5 and no_improvement_count > 5:
            current_alpha = max(0.5, alpha * 0.7)
            current_beta = max(1.5, beta * 0.9) 
        else:
            current_alpha = alpha
            current_beta = beta  

        # Pheromone update with adaptive rho
        tau *= (1 - current_rho)
        tau = torch.clamp(tau, min=0.01, max=100.0)

        sorted_indices = torch.argsort(cuts, descending=True)[:n_elite]
        
        for rank, idx in enumerate(sorted_indices):
            c = solutions[idx]
            rank_weight = (n_elite - rank) / n_elite
            # Remove normalization for stronger rewards
            reward = Q * rank_weight
            
            for i in range(n):
                tau[i, c[i]] += reward
        

        temperature = max(1.5, temperature * 0.998)
        
        # Restart
        if no_improvement_count >= restart_threshold:
            if phase_tracking['restart_perturbation']['start'] is None:
                phase_tracking['restart_perturbation']['start'] = best_cut.item()
                print(f"[TRACKING] Phase 4: Restart/Perturbation starts")
            
            print(f"[ROS-ACO] No improvement for {restart_threshold} iters, restarting...")
            phase_tracking['restart_perturbation']['events'].append({
                'type': 'restart',
                'iteration': it,
                'cut_before': best_cut.item()
            })
            restart_count += 1
            no_improvement_count = 0
            
            # Smarter restart based on best solution
            tau = torch.ones(n, k, device=device) * 1.0
            tau = tau + X_relaxed.T * 0.1  # Reduced for more exploration
            # Add LITTLE bias from best solution to avoid convergence
            if best_assign is not None:
                for i in range(n):
                    tau[i, best_assign[i]] += 0.5  
            tau += torch.rand(n, k, device=device) * 1.0  # More noise
            tau = torch.clamp(tau, min=0.01, max=100.0)
            temperature = 5.0  # Increased for more exploration
            # Reset parameters
            current_alpha = alpha
            current_beta = beta
            current_rho = rho

        if it % max(1, aco_iters // 10) == 0:
            print(f"[ROS-ACO] iter {it}/{aco_iters} | best = {best_cut:.4f} | temp = {temperature:.3f} | entropy = {entropy:.3f} | β = {current_beta:.2f} | restarts = {restart_count}")
   
 
    # Calculate final timings
    aco_end_time = time.time()
    aco_elapsed = aco_end_time - aco_start_time
    aco_pure_time = aco_elapsed - local_search_total_time
    global_end_time = time.time()
    total_time = global_end_time - global_start_time
    
    # Print timing report
    print("\n" + "="*60)
    print("TIMING BREAKDOWN")
    print("="*60)
    print(f"ROS (GNN) phase:     {ros_elapsed:8.2f}s  ({ros_elapsed/total_time*100:5.1f}%)")
    print(f"ACO phase:           {aco_pure_time:8.2f}s  ({aco_pure_time/total_time*100:5.1f}%)")
    print(f"Local Search:        {local_search_total_time:8.2f}s  ({local_search_total_time/total_time*100:5.1f}%)")
    print("-" * 60)
    print(f"TOTAL:               {total_time:8.2f}s  (100.0%)")
    print("="*60 + "\n")
    
    # Finalize tracking
    phase_tracking['pure_aco']['end'] = best_cut.item()
    if phase_tracking['local_search']['start'] is not None:
        phase_tracking['local_search']['end'] = best_cut.item()
    if phase_tracking['restart_perturbation']['start'] is not None:
        phase_tracking['restart_perturbation']['end'] = best_cut.item()
    
    # Add timing to phase_tracking
    phase_tracking['timing'] = {
        'ros_time': ros_elapsed,
        'aco_time': aco_pure_time,
        'local_search_time': local_search_total_time,
        'total_time': total_time,
        'ros_percentage': ros_elapsed / total_time * 100,
        'aco_percentage': aco_pure_time / total_time * 100,
        'local_search_percentage': local_search_total_time / total_time * 100
    }
    
    # Save tracking data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracking_file = f"tracking_data_{timestamp}.json"
    
    # Convert tensors to float for JSON
    tracking_data = {
        'timestamp': timestamp,
        'graph_type': getattr(args, 'graph_type', 'unknown'),
        'k': args.k,
        'phases': phase_tracking,
        'final_cut': best_cut.item()
    }
    
    os.makedirs('tracking_logs', exist_ok=True)
    with open(f'tracking_logs/{tracking_file}', 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    print(f"[TRACKING] Data saved in: tracking_logs/{tracking_file}")
    
    return best_assign, X_relaxed, phase_tracking


def ros_aco_only(args, graph,
                 n_ants=20,
                 aco_iters=20,
                 alpha=1.0,
                 beta=2.0,
                 rho=0.1,
                 Q=10.0,
                 elite_ratio=0.3,
                 restart_threshold=60,
                 perturbation_threshold=30):
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Track timing for each phase
    global_start_time = time.time()
    
    #  GNN 
    ros_start_time = time.time()
    W = get_matrix(args, graph)

    net, optimizer, edges, edges_weight, inputs = get_gnn_tuning(args, graph)
    net.load_state_dict(torch.load(
        f"gcn_model_ood_k{args.k}.pth",
        map_location=device,
        weights_only=True
    ))

    print("[ROS-ACO-ONLY] Running GNN...")
    X_relaxed = run_gnn_tuning(
        args, W, edges, edges_weight,
        net, optimizer,
        args.epochs, args.tol, args.patience, inputs
    )  # (k, n)
    
    ros_end_time = time.time()
    ros_elapsed = ros_end_time - ros_start_time
    print(f"[TIMING] ROS (GNN) phase: {ros_elapsed:.2f}s")

    k, n = X_relaxed.shape

    #  Initialize pheromones 
    tau = torch.ones(n, k, device=device) * 1.0
    tau = tau + X_relaxed.T * 1.0  # More weight to GNN probabilities
    tau = torch.clamp(tau, min=0.1, max=5.0)

    best_cut = -torch.inf
    best_assign = None

    total_weight = W.sum() 
    
    n_elite = max(1, int(n_ants * elite_ratio))
    temperature = 1.5  # Lower for more exploitation
    no_improvement_count = 0
    restart_count = 0
    
    # Adaptive parameters
    current_alpha = alpha
    current_beta = beta
    current_rho = rho
    
    # Best solution tracking
    best_history = []
    
    # Track ACO timing (no local search)
    aco_start_time = time.time()
    
    print("[ROS-ACO-ONLY] Starting ACO refinement (NO Local Search)...")

    for it in range(aco_iters):

        solutions = []
        cuts = []

        for ant in range(n_ants):
            heuristic = X_relaxed.T  # (n, k)
            P = (tau ** current_alpha) * (heuristic ** current_beta)  # (n, k)
            P = P ** (1.0 / temperature)
            P = P / (P.sum(dim=1, keepdim=True) + 1e-12)

            c = torch.multinomial(P, 1).squeeze()  # (n,)
            X = torch.nn.functional.one_hot(c, num_classes=k).T.float()

            val = torch.trace(X @ W @ X.T)
            cut = total_weight - val

            solutions.append(c)
            cuts.append(cut)

        cuts = torch.stack(cuts)
                
        # Update best
        iter_best_idx = torch.argmax(cuts)
        if cuts[iter_best_idx] > best_cut:
            best_cut = cuts[iter_best_idx]
            best_assign = solutions[iter_best_idx].clone()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        best_history.append(best_cut.item())
        
        # Perturbation if stuck
        if no_improvement_count == perturbation_threshold:
            print(f"[ROS-ACO-ONLY] Applying perturbation...")
            # Increase exploration temporarily
            current_beta = max(0.1, beta * 0.5)
            current_rho = min(0.3, rho * 2.0)
            # Add noise to pheromones
            tau += torch.rand_like(tau) * 0.5
        elif no_improvement_count > perturbation_threshold:
            # Gradually return to normal values
            current_beta = min(beta, current_beta * 1.05)
            current_rho = max(rho, current_rho * 0.95)
        
        # Calculate entropy for diversification
        entropy = calculate_pheromone_entropy(tau)
        
        # If entropy is low, increase exploration
        if entropy < 0.5 and no_improvement_count > 10:
            current_alpha = max(0.5, alpha * 0.8)
            current_beta = max(0.2, beta * 0.7)
        else:
            current_alpha = alpha
            current_beta = min(beta, current_beta * 1.02)

        # Pheromone update with adaptive rho
        tau *= (1 - current_rho)
        tau = torch.clamp(tau, min=0.01, max=100.0)

        sorted_indices = torch.argsort(cuts, descending=True)[:n_elite]
        
        for rank, idx in enumerate(sorted_indices):
            c = solutions[idx]
            rank_weight = (n_elite - rank) / n_elite
            # Remove normalization for stronger rewards
            reward = Q * rank_weight
            
            for i in range(n):
                tau[i, c[i]] += reward
        
        temperature = max(1.5, temperature * 0.998)
        
        # Restart
        if no_improvement_count >= restart_threshold:
            print(f"[ROS-ACO-ONLY] No improvement for {restart_threshold} iters, restarting...")
            restart_count += 1
            no_improvement_count = 0
            
            # Smarter restart based on best solution
            tau = torch.ones(n, k, device=device) * 0.3
            tau = tau + X_relaxed.T * 0.1  # Reduced for more exploration
            # Add LITTLE bias from best solution to avoid convergence
            if best_assign is not None:
                for i in range(n):
                    tau[i, best_assign[i]] += 0.5  
            tau += torch.rand(n, k, device=device) * 1.0  # More noise
            tau = torch.clamp(tau, min=0.01, max=100.0)
            temperature = 5.0  # Increased for more exploration
            # Reset parameters
            current_alpha = alpha
            current_beta = beta
            current_rho = rho

        if it % max(1, aco_iters // 10) == 0:
            print(f"[ROS-ACO-ONLY] iter {it}/{aco_iters} | best = {best_cut:.4f} | temp = {temperature:.3f} | entropy = {entropy:.3f} | β = {current_beta:.2f} | restarts = {restart_count}")
    
    # Calculate final times
    aco_end_time = time.time()
    aco_elapsed = aco_end_time - aco_start_time
    global_end_time = time.time()
    total_time = global_end_time - global_start_time
    
    # Print timing report
    print("\n" + "="*60)
    print("TIMING BREAKDOWN")
    print("="*60)
    print(f"ROS (GNN) phase:     {ros_elapsed:8.2f}s  ({ros_elapsed/total_time*100:5.1f}%)")
    print(f"ACO phase:           {aco_elapsed:8.2f}s  ({aco_elapsed/total_time*100:5.1f}%)")
    print(f"Local Search:        {'N/A':>8s}  (  0.0%)")
    print("-" * 60)
    print(f"TOTAL:               {total_time:8.2f}s  (100.0%)")
    print("="*60 + "\n")
    
    print(f"[ROS-ACO-ONLY] Final best cut = {best_cut:.4f}")
    
    return best_assign, X_relaxed


def ros_local_search(args, graph,
                     n_initial_samples=50,
                     two_opt_iters=100,
                     swap_iters=50):
    """
    ROS + Local Search (2-opt + swap)
    Baseline non bio-inspired
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #  GNN 
    W = get_matrix(args, graph)

    net, optimizer, edges, edges_weight, inputs = get_gnn_tuning(args, graph)
    net.load_state_dict(torch.load(
        f"gcn_model_ood_k{args.k}.pth",
        map_location=device,
        weights_only=True
    ))

    print("[ROS-LS] Running GNN...")
    X_relaxed = run_gnn_tuning(
        args, W, edges, edges_weight,
        net, optimizer,
        args.epochs, args.tol, args.patience, inputs
    )  # (k, n)

    k, n = X_relaxed.shape
    total_weight = W.sum()

    #  ROS sampling 
    best_cut = -torch.inf
    best_assign = None

    print("[ROS-LS] Sampling initial solutions...")

    for it in range(n_initial_samples):
        c = torch.multinomial(X_relaxed.T, 1).squeeze()  # (n,)
        X = torch.nn.functional.one_hot(c, num_classes=k).T.float()

        val = torch.trace(X @ W @ X.T)
        cut = total_weight - val

        if cut > best_cut:
            best_cut = cut
            best_assign = c.clone()

    print(f"[ROS-LS] Best initial cut = {best_cut:.4f}")

    #  Local Search 
    print("[ROS-LS] Running 2-opt...")
    best_assign, best_cut = local_search_2opt(
        best_assign, W, k, total_weight, max_iters=two_opt_iters
    )
    print(f"[ROS-LS] After 2-opt: cut = {best_cut:.4f}")

    print("[ROS-LS] Running swap...")
    best_assign, best_cut = local_search_swap(
        best_assign, W, k, total_weight, max_iters=swap_iters
    )
    print(f"[ROS-LS] After swap: cut = {best_cut:.4f}")

    #  Optional final polish 
    best_assign, best_cut = local_search_2opt(
        best_assign, W, k, total_weight, max_iters=two_opt_iters // 2
    )
    print(f"[ROS-LS] Final cut = {best_cut:.4f}")

    return best_assign, X_relaxed


def calculate_pheromone_entropy(tau):
    
    # Calculate pheromone entropy to measure diversification
    
    tau_norm = tau / (tau.sum(dim=1, keepdim=True) + 1e-12)
    entropy = -(tau_norm * torch.log(tau_norm + 1e-12)).sum(dim=1).mean()
    return entropy


# GENETIC ALGORITHM IMPLEMENTATION
def tournament_selection(population, fitness, tournament_size):
    # Tournament selection: choose k random individuals and return the best
    
    indices = torch.randperm(len(population))[:tournament_size]
    tournament_fitness = fitness[indices]
    winner_idx = indices[torch.argmax(tournament_fitness)]
    return population[winner_idx].clone()


def uniform_crossover(parent1, parent2, k):
    # Uniform crossover: each gene is chosen from one of the two parents
    mask = torch.rand(len(parent1)) > 0.5
    child1 = torch.where(mask, parent1, parent2)
    child2 = torch.where(mask, parent2, parent1)
    return child1, child2


def mutate(individual, k, n, mutation_strength=0.1):
    # Mutation: randomly change some clusters
    # mutation_strength: percentage of nodes to mutate

    n_mutations = max(1, int(n * mutation_strength))
    nodes_to_mutate = torch.randperm(n)[:n_mutations]
    
    mutated = individual.clone()
    for node in nodes_to_mutate:
        # Choose a cluster different from the current one
        current_cluster = individual[node].item()
        possible_clusters = [c for c in range(k) if c != current_cluster]
        if possible_clusters:
            mutated[node] = torch.tensor(possible_clusters[torch.randint(len(possible_clusters), (1,)).item()], device=individual.device)
    
    return mutated


def smart_mutate(individual, W, k, n, mutation_strength=0.1):
    # Intelligent mutation: tries to move nodes that improve the cut

    device = individual.device
    total_weight = W.sum()
    n_mutations = max(1, int(n * mutation_strength))
    
    mutated = individual.clone()
    
    # Calculate current cut
    X = torch.nn.functional.one_hot(mutated, num_classes=k).T.float()
    current_cut = total_weight - torch.trace(X @ W @ X.T)
    
    # Try to mutate nodes with probability proportional to potential improvement
    nodes = torch.randperm(n)[:min(n_mutations * 3, n)]  # Consider more nodes
    
    mutations_done = 0
    for node in nodes:
        if mutations_done >= n_mutations:
            break
            
        current_cluster = mutated[node].item()
        best_cluster = current_cluster
        best_delta = 0
        
        # Try all alternative clusters
        for new_cluster in range(k):
            if new_cluster == current_cluster:
                continue
            
            # Fast approximate estimate of delta
            # Delta = difference between cut with new cluster vs current cluster
            test = mutated.clone()
            test[node] = new_cluster
            X_test = torch.nn.functional.one_hot(test, num_classes=k).T.float()
            new_cut = total_weight - torch.trace(X_test @ W @ X_test.T)
            delta = new_cut - current_cut
            
            if delta > best_delta:
                best_delta = delta
                best_cluster = new_cluster
        
        # With probability 0.7, make the best move; otherwise random (exploration)
        if torch.rand(1).item() < 0.7 and best_cluster != current_cluster:
            mutated[node] = best_cluster
            current_cut += best_delta
            mutations_done += 1
        elif torch.rand(1).item() < 0.3:  # Random mutation for exploration
            possible = [c for c in range(k) if c != current_cluster]
            if possible:
                mutated[node] = torch.tensor(possible[torch.randint(len(possible), (1,)).item()], device=device)
                X = torch.nn.functional.one_hot(mutated, num_classes=k).T.float()
                current_cut = total_weight - torch.trace(X @ W @ X.T)
                mutations_done += 1
    
    return mutated


def crossover_2opt(parent1, parent2, W, k, max_tries=10, acceptance_prob=0.7):
    """
    2-opt based crossover: starts from parent1 and applies 2-opt using parent2 as a guide to choose nodes to change
    acceptance_prob: probability of accepting improving moves (< 1.0 for more exploration)
    """
    n = len(parent1)
    device = parent1.device
    total_weight = W.sum()
    
    # Start from parent1
    child = parent1.clone()
    X = torch.nn.functional.one_hot(child, num_classes=k).T.float()
    current_cut = total_weight - torch.trace(X @ W @ X.T)
    
    # Find nodes where parents differ
    diff_mask = (parent1 != parent2)
    diff_nodes = torch.where(diff_mask)[0]
    
    if len(diff_nodes) == 0:
        return child
    
    # Try to change some nodes where they differ, looking for improvements
    n_tries = min(max_tries, len(diff_nodes))
    nodes_to_try = diff_nodes[torch.randperm(len(diff_nodes))[:n_tries]]
    
    for node in nodes_to_try:
        # Try the cluster of parent2
        new_cluster = parent2[node].item()
        if new_cluster == child[node].item():
            continue
        
        # Test the change
        test_child = child.clone()
        test_child[node] = new_cluster
        X_test = torch.nn.functional.one_hot(test_child, num_classes=k).T.float()
        new_cut = total_weight - torch.trace(X_test @ W @ X_test.T)
        
        # Accept if improves OR with probability (1-acceptance_prob) even if slightly worse
        if new_cut > current_cut:
            child = test_child
            current_cut = new_cut
        elif torch.rand(1).item() > acceptance_prob:
            # Exploration: accept even neutral/worse moves
            child = test_child
            current_cut = new_cut
    
    return child


def crossover_swap(parent1, parent2, W, k, max_swaps=5, acceptance_prob=0.7):
    """
    Swap-based crossover: starts from parent1 and swaps pairs
    using parent2 to identify candidate pairs
    """
    n = len(parent1)
    device = parent1.device
    total_weight = W.sum()
    
    # Start from parent1
    child = parent1.clone()
    X = torch.nn.functional.one_hot(child, num_classes=k).T.float()
    current_cut = total_weight - torch.trace(X @ W @ X.T)
    
    # Find nodes where parents differ
    diff_mask = (parent1 != parent2)
    diff_nodes = torch.where(diff_mask)[0]
    
    if len(diff_nodes) < 2:
        return child
    
    # Try some swaps between nodes that differ
    swaps_done = 0
    for _ in range(max_swaps * 2):  # Try multiple times
        if swaps_done >= max_swaps:
            break
        
        # Choose two random nodes from the nodes that differ
        if len(diff_nodes) < 2:
            break
        indices = torch.randperm(len(diff_nodes))[:2]
        i, j = diff_nodes[indices[0]].item(), diff_nodes[indices[1]].item()
        
        if child[i] == child[j]:
            continue
        
        # Test the swap
        test_child = child.clone()
        test_child[i], test_child[j] = child[j].item(), child[i].item()
        X_test = torch.nn.functional.one_hot(test_child, num_classes=k).T.float()
        new_cut = total_weight - torch.trace(X_test @ W @ X_test.T)
        
        # Accept if improves
        if new_cut > current_cut:
            child = test_child
            current_cut = new_cut
            swaps_done += 1
    
    return child


def ros_ga(args, graph,
           pop_size=50,
           n_generations=100,
           tournament_size=5,
           crossover_rate=0.8,
           mutation_rate=0.9,
           elite_size=5,
           local_search_freq=10,
           local_search_elite_only=True):
    """
    ROS + Genetic Algorithm
    - Initialization from GNN (not random)
    - Tournament selection
    - Uniform crossover
    - Mutation: random cluster change
    - Local search periodic (2-opt + swap)
    - Elitism
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Track timing
    global_start_time = time.time()
    
    # ---------- GNN ----------
    ros_start_time = time.time()
    W = get_matrix(args, graph)
    
    net, optimizer, edges, edges_weight, inputs = get_gnn_tuning(args, graph)
    net.load_state_dict(torch.load(
        f"gcn_model_ood_k{args.k}.pth",
        map_location=device,
        weights_only=True
    ))
    
    print("[ROS-GA] Running GNN...")
    X_relaxed = run_gnn_tuning(
        args, W, edges, edges_weight,
        net, optimizer,
        args.epochs, args.tol, args.patience, inputs
    )  # (k, n)
    
    ros_end_time = time.time()
    ros_elapsed = ros_end_time - ros_start_time
    print(f"[TIMING] ROS (GNN) phase: {ros_elapsed:.2f}s")
    
    k, n = X_relaxed.shape
    total_weight = W.sum()
    
    # ---------- Inizializza popolazione dalla GNN ----------
    print(f"[ROS-GA] Initializing population (size={pop_size})...")
    print(f"[ROS-GA] - Creating {min(10, pop_size)} seeds from GNN")
    print(f"[ROS-GA] - Creating {pop_size - min(10, pop_size)} individuals via GA operators")
    
    population = []
    fitness = []
    
    # Create only a few seed solutions from GNN (10 or fewer)
    n_seeds = min(10, pop_size)
    seeds = []
    
    for i in range(n_seeds):
        if i < n_seeds // 2:
            # Pure sampling from GNN
            individual = torch.multinomial(X_relaxed.T, 1).squeeze()
        else:
            # Sampling with noise
            probs = X_relaxed.T + torch.rand(n, k, device=device) * 0.3
            probs = probs / probs.sum(dim=1, keepdim=True)
            individual = torch.multinomial(probs, 1).squeeze()
        
        seeds.append(individual)
        X = torch.nn.functional.one_hot(individual, num_classes=k).T.float()
        cut = total_weight - torch.trace(X @ W @ X.T)
        population.append(individual)
        fitness.append(cut)
    
    # Generate the rest of the population using GA operators on the seeds
    while len(population) < pop_size:
        # Take two random seeds
        idx1, idx2 = torch.randperm(n_seeds)[:2]
        parent1, parent2 = seeds[idx1], seeds[idx2]
        
        # Crossover
        if torch.rand(1).item() < 0.8:
            child, _ = uniform_crossover(parent1, parent2, k)
        else:
            child = parent1.clone()
        
        # Mutation
        if torch.rand(1).item() < 0.5:
            child = mutate(child, k, n, mutation_strength=0.15)
        
        # Add to population
        X = torch.nn.functional.one_hot(child, num_classes=k).T.float()
        cut = total_weight - torch.trace(X @ W @ X.T)
        population.append(child)
        fitness.append(cut)
    
    fitness = torch.stack(fitness)
    best_idx = torch.argmax(fitness)
    best_individual = population[best_idx].clone()
    best_fitness = fitness[best_idx].item()
    
    print(f"[ROS-GA] Initial best fitness: {best_fitness:.4f}")
    
    # Track timing
    ga_start_time = time.time()
    
    #  GA Loop 
    print(f"[ROS-GA] Starting GA evolution ({n_generations} generations)...")
    no_improvement_count = 0
    best_history = []
    
    for gen in range(n_generations):
        new_population = []
        new_fitness = []
        
        # Elitism: conserves the best
        elite_indices = torch.argsort(fitness, descending=True)[:elite_size]
        for idx in elite_indices:
            new_population.append(population[idx].clone())
            new_fitness.append(fitness[idx])
        
        # Generate new individuals
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)
            
            # Crossover
            if torch.rand(1).item() < crossover_rate:
                child1, child2 = uniform_crossover(parent1, parent2, k)
            else:
                child1, child2 = parent1.clone(), parent2.clone()
            
            # Mutation
            if torch.rand(1).item() < mutation_rate:
                child1 = mutate(child1, k, n, mutation_strength=0.1)
            if torch.rand(1).item() < mutation_rate:
                child2 = mutate(child2, k, n, mutation_strength=0.1)
            
            # Calculate fitness
            for child in [child1, child2]:
                if len(new_population) >= pop_size:
                    break
                X = torch.nn.functional.one_hot(child, num_classes=k).T.float()
                child_fitness = total_weight - torch.trace(X @ W @ X.T)
                new_population.append(child)
                new_fitness.append(child_fitness)
        
        population = new_population
        fitness = torch.stack(new_fitness)
        
        # Update best
        gen_best_idx = torch.argmax(fitness)
        if fitness[gen_best_idx] > best_fitness:
            best_fitness = fitness[gen_best_idx].item()
            best_individual = population[gen_best_idx].clone()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        best_history.append(best_fitness)
        
        # Diversity injection if stuck (uses GA operators, not GNN)
        if no_improvement_count >= 20:
            print(f"[ROS-GA] Injecting diversity at generation {gen} via mutation...")
            # Replace worst with heavy mutations of the best
            worst_indices = torch.argsort(fitness)[:pop_size // 4]
            best_indices = torch.argsort(fitness, descending=True)[:5]
            
            for idx in worst_indices:
                # Take a good individual and heavily mutate it
                source = population[best_indices[torch.randint(len(best_indices), (1,)).item()]].clone()
                new_ind = mutate(source, k, n, mutation_strength=0.3)  # Stronger mutation
                population[idx] = new_ind
                X = torch.nn.functional.one_hot(new_ind, num_classes=k).T.float()
                fitness[idx] = total_weight - torch.trace(X @ W @ X.T)
            no_improvement_count = 0
        
        if gen % max(1, n_generations // 10) == 0:
            avg_fitness = fitness.mean().item()
            print(f"[ROS-GA] gen {gen}/{n_generations} | best = {best_fitness:.4f} | avg = {avg_fitness:.4f} | no_imp = {no_improvement_count}")
    
    # Timing report
    ga_end_time = time.time()
    ga_elapsed = ga_end_time - ga_start_time
    global_end_time = time.time()
    total_time = global_end_time - global_start_time
    
    print("\n" + "="*60)
    print("TIMING BREAKDOWN")
    print("="*60)
    print(f"ROS (GNN) phase:     {ros_elapsed:8.2f}s  ({ros_elapsed/total_time*100:5.1f}%)")
    print(f"GA phase:            {ga_elapsed:8.2f}s  ({ga_elapsed/total_time*100:5.1f}%)")
    print("-" * 60)
    print(f"TOTAL:               {total_time:8.2f}s  (100.0%)")
    print("="*60 + "\n")
    
    print(f"[ROS-GA] Final best fitness: {best_fitness:.4f}")
    
    return best_individual, X_relaxed


def ros_ga_improved(args, graph,
                    pop_size=30,
                    n_generations=50,
                    tournament_size=5,
                    crossover_rate=0.8,
                    mutation_rate=0.7,
                    elite_size=5,
                    use_smart_operators=True,
                    local_search_freq=5,
                    local_search_elite_only=True):
    """
    ROS + Memetic Algorithm (GA + Local Search)
    - GNN for initialization
    - Smart mutation (mutation that looks at the graph)
    - 2-opt/swap crossover (local search operators)
    - Periodic local search on elites (intensification)
    - Elitism
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Track timing
    global_start_time = time.time()
    
    #  GNN 
    ros_start_time = time.time()
    W = get_matrix(args, graph)
    
    net, optimizer, edges, edges_weight, inputs = get_gnn_tuning(args, graph)
    net.load_state_dict(torch.load(
        f"gcn_model_ood_k{args.k}.pth",
        map_location=device,
        weights_only=True
    ))
    
    print("[ROS-GA-IMPROVED] Running GNN...")
    X_relaxed = run_gnn_tuning(
        args, W, edges, edges_weight,
        net, optimizer,
        args.epochs, args.tol, args.patience, inputs
    )  # (k, n)
    
    ros_end_time = time.time()
    ros_elapsed = ros_end_time - ros_start_time
    print(f"[TIMING] ROS (GNN) phase: {ros_elapsed:.2f}s")
    
    k, n = X_relaxed.shape
    total_weight = W.sum()
    
    #  Initialize population from GNN 
    print(f"[ROS-GA-IMPROVED] Initializing population (size={pop_size})...")
    print(f"[ROS-GA-IMPROVED] Using {'SMART' if use_smart_operators else 'STANDARD'} operators")
    
    population = []
    fitness = []
    
    # Create seeds from GNN
    n_seeds = min(10, pop_size)
    seeds = []
    
    for i in range(n_seeds):
        if i < n_seeds // 2:
            individual = torch.multinomial(X_relaxed.T, 1).squeeze()
        else:
            probs = X_relaxed.T + torch.rand(n, k, device=device) * 0.3
            probs = probs / probs.sum(dim=1, keepdim=True)
            individual = torch.multinomial(probs, 1).squeeze()
        
        seeds.append(individual)
        X = torch.nn.functional.one_hot(individual, num_classes=k).T.float()
        cut = total_weight - torch.trace(X @ W @ X.T)
        population.append(individual)
        fitness.append(cut)
    
    # Generate the rest of the population using GA operators
    while len(population) < pop_size:
        idx1, idx2 = torch.randperm(n_seeds)[:2]
        parent1, parent2 = seeds[idx1], seeds[idx2]
        
        if use_smart_operators:
            # 50% 2-opt crossover, 50% swap crossover
            if torch.rand(1).item() < 0.5:
                child = crossover_2opt(parent1, parent2, W, k)
            else:
                child = crossover_swap(parent1, parent2, W, k)
        else:
            child, _ = uniform_crossover(parent1, parent2, k)
        
        if torch.rand(1).item() < 0.5:
            if use_smart_operators:
                child = smart_mutate(child, W, k, n, mutation_strength=0.1)
            else:
                child = mutate(child, k, n, mutation_strength=0.15)
        
        X = torch.nn.functional.one_hot(child, num_classes=k).T.float()
        cut = total_weight - torch.trace(X @ W @ X.T)
        population.append(child)
        fitness.append(cut)
    
    fitness = torch.stack(fitness)
    best_idx = torch.argmax(fitness)
    best_individual = population[best_idx].clone()
    best_fitness = fitness[best_idx].item()
    
    print(f"[ROS-GA-IMPROVED] Initial best fitness: {best_fitness:.4f}")
    
    # Track timing
    ga_start_time = time.time()
    local_search_total_time = 0.0
    
    #  GA Loop 
    print(f"[ROS-GA-IMPROVED] Starting Memetic evolution ({n_generations} generations)...")
    print(f"[ROS-GA-IMPROVED] Local search every {local_search_freq} generations")
    no_improvement_count = 0
    best_history = []
    
    for gen in range(n_generations):
        new_population = []
        new_fitness = []
        
        # Elitism
        elite_indices = torch.argsort(fitness, descending=True)[:elite_size]
        for idx in elite_indices:
            new_population.append(population[idx].clone())
            new_fitness.append(fitness[idx])
        
        # Generate new individuals
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)
            
            # Crossover
            if torch.rand(1).item() < crossover_rate:
                if use_smart_operators:
                    # Use local search operators as crossover
                    child1 = crossover_2opt(parent1, parent2, W, k)
                    child2 = crossover_swap(parent2, parent1, W, k)
                else:
                    child1, child2 = uniform_crossover(parent1, parent2, k)
            else:
                child1, child2 = parent1.clone(), parent2.clone()
            
            # Mutation
            if torch.rand(1).item() < mutation_rate:
                if use_smart_operators:
                    child1 = smart_mutate(child1, W, k, n, mutation_strength=0.1)
                else:
                    child1 = mutate(child1, k, n, mutation_strength=0.1)
            if torch.rand(1).item() < mutation_rate:
                if use_smart_operators:
                    child2 = smart_mutate(child2, W, k, n, mutation_strength=0.1)
                else:
                    child2 = mutate(child2, k, n, mutation_strength=0.1)
            
            # Calculate fitness
            for child in [child1, child2]:
                if len(new_population) >= pop_size:
                    break
                X = torch.nn.functional.one_hot(child, num_classes=k).T.float()
                child_fitness = total_weight - torch.trace(X @ W @ X.T)
                new_population.append(child)
                new_fitness.append(child_fitness)
        
        population = new_population
        fitness = torch.stack(new_fitness)
        
        # Periodic local search (MEMETIC COMPONENT)
        if gen % local_search_freq == 0 and gen > 0:
            ls_start = time.time()
            
            if local_search_elite_only:
                # Only on elites (top 3)
                indices_to_improve = torch.argsort(fitness, descending=True)[:min(3, elite_size)]
            else:
                # On multiple individuals
                indices_to_improve = torch.argsort(fitness, descending=True)[:5]
            
            improvements = 0
            for idx in indices_to_improve:
                before = fitness[idx].item()
                
                # Apply 2-opt (reduced to 15 iterations for speed)
                improved, improved_fitness = local_search_2opt(
                    population[idx], W, k, total_weight, max_iters=15
                )
                # Then swap (reduced to 8 iterations)
                improved, improved_fitness = local_search_swap(
                    improved, W, k, total_weight, max_iters=8
                )
                
                if improved_fitness > fitness[idx]:
                    population[idx] = improved
                    fitness[idx] = improved_fitness
                    improvements += 1
            
            ls_end = time.time()
            local_search_total_time += (ls_end - ls_start)
            
            if improvements > 0:
                print(f"[ROS-GA-IMPROVED] Local search at gen {gen}: improved {improvements}/{len(indices_to_improve)} individuals")
        
        # Update best
        gen_best_idx = torch.argmax(fitness)
        if fitness[gen_best_idx] > best_fitness:
            best_fitness = fitness[gen_best_idx].item()
            best_individual = population[gen_best_idx].clone()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        best_history.append(best_fitness)
        
        # Diversity injection (MORE FREQUENT)
        if no_improvement_count >= 10:
            print(f"[ROS-GA-IMPROVED] Injecting diversity at generation {gen}...")
            worst_indices = torch.argsort(fitness)[:pop_size // 4]
            best_indices = torch.argsort(fitness, descending=True)[:5]
            
            for idx in worst_indices:
                source = population[best_indices[torch.randint(len(best_indices), (1,)).item()]].clone()
                if use_smart_operators:
                    new_ind = smart_mutate(source, W, k, n, mutation_strength=0.3)
                else:
                    new_ind = mutate(source, k, n, mutation_strength=0.3)
                population[idx] = new_ind
                X = torch.nn.functional.one_hot(new_ind, num_classes=k).T.float()
                fitness[idx] = total_weight - torch.trace(X @ W @ X.T)
            no_improvement_count = 0
        
        if gen % max(1, n_generations // 10) == 0:
            avg_fitness = fitness.mean().item()
            print(f"[ROS-GA-IMPROVED] gen {gen}/{n_generations} | best = {best_fitness:.4f} | avg = {avg_fitness:.4f} | no_imp = {no_improvement_count}")
    
    # Final intensive local search (optional polish)
    print("[ROS-GA-IMPROVED] Final intensive local search on best solution...")
    ls_start = time.time()
    best_individual, best_fitness = local_search_2opt(
        best_individual, W, k, total_weight, max_iters=50
    )
    best_individual, best_fitness = local_search_swap(
        best_individual, W, k, total_weight, max_iters=30
    )
    ls_end = time.time()
    local_search_total_time += (ls_end - ls_start)
    
    # Timing report
    ga_end_time = time.time()
    ga_elapsed = ga_end_time - ga_start_time
    ga_pure_time = ga_elapsed - local_search_total_time
    total_time = ga_end_time - global_start_time
    
    print("\n" + "="*60)
    print("TIMING BREAKDOWN (MEMETIC ALGORITHM)")
    print("="*60)
    print(f"ROS (GNN) phase:     {ros_elapsed:8.2f}s  ({ros_elapsed/total_time*100:5.1f}%)")
    print(f"GA phase:            {ga_pure_time:8.2f}s  ({ga_pure_time/total_time*100:5.1f}%)")
    print(f"Local Search:        {local_search_total_time:8.2f}s  ({local_search_total_time/total_time*100:5.1f}%)")
    print("-" * 60)
    print(f"TOTAL:               {total_time:8.2f}s  (100.0%)")
    print("="*60 + "\n")
    
    print(f"[ROS-GA-IMPROVED] Final best fitness: {best_fitness:.4f}")
    
    return best_individual, X_relaxed


def ga_only(args, graph,
            pop_size=None,
            n_generations=None,
            tournament_size=5,
            crossover_rate=None,
            mutation_rate=None,
            elite_size=5,
            local_search_freq=10,
            local_search_elite_only=True):
    """
    Pure Genetic Algorithm (without GNN)
    - Random initialization
    - Tournament selection
    - Uniform crossover
    - Mutation: random cluster change
    - Periodic local search (2-opt + swap)
    - Elitism
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use parameters from args if not specified
    if pop_size is None:
        pop_size = args.POPSIZE
    if n_generations is None:
        n_generations = args.MAXGENS
    if crossover_rate is None:
        crossover_rate = args.PXOVER
    if mutation_rate is None:
        mutation_rate = args.PMUTATION
    
    # Track timing
    global_start_time = time.time()
    
    #  Setup 
    W = get_matrix(args, graph)
    k = args.k
    n = W.shape[0]
    total_weight = W.sum()
    
    #  Initialize RANDOM population 
    print(f"[GA-ONLY] Initializing RANDOM population (size={pop_size})...")
    population = []
    fitness = []
    
    for i in range(pop_size):
        # Completely random initialization
        individual = torch.randint(0, k, (n,), device=device)
        
        # Calculate fitness
        X = torch.nn.functional.one_hot(individual, num_classes=k).T.float()
        cut = total_weight - torch.trace(X @ W @ X.T)
        
        population.append(individual)
        fitness.append(cut)
    
    fitness = torch.stack(fitness)
    best_idx = torch.argmax(fitness)
    best_individual = population[best_idx].clone()
    best_fitness = fitness[best_idx].item()
    
    print(f"[GA-ONLY] Initial best fitness: {best_fitness:.4f}")
    
    # Track timing
    ga_start_time = time.time()
    local_search_total_time = 0.0
    
    #  GA Loop 
    print(f"[GA-ONLY] Starting GA evolution ({n_generations} generations)...")
    no_improvement_count = 0
    best_history = []
    
    for gen in range(n_generations):
        new_population = []
        new_fitness = []
        
        # Elitism: keep the best
        elite_indices = torch.argsort(fitness, descending=True)[:elite_size]
        for idx in elite_indices:
            new_population.append(population[idx].clone())
            new_fitness.append(fitness[idx])
        
        # Generate new individuals
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)
            
            # Crossover
            if torch.rand(1).item() < crossover_rate:
                child1, child2 = uniform_crossover(parent1, parent2, k)
            else:
                child1, child2 = parent1.clone(), parent2.clone()
            
            # Mutation
            if torch.rand(1).item() < mutation_rate:
                child1 = mutate(child1, k, n, mutation_strength=0.1)
            if torch.rand(1).item() < mutation_rate:
                child2 = mutate(child2, k, n, mutation_strength=0.1)
            
            # Calculate fitness
            for child in [child1, child2]:
                if len(new_population) >= pop_size:
                    break
                X = torch.nn.functional.one_hot(child, num_classes=k).T.float()
                child_fitness = total_weight - torch.trace(X @ W @ X.T)
                new_population.append(child)
                new_fitness.append(child_fitness)
        
        population = new_population
        fitness = torch.stack(new_fitness)
        
        # Local search periodico
        if gen % local_search_freq == 0 and gen > 0:
            ls_start = time.time()
            
            if local_search_elite_only:
                # Only on elites
                indices_to_improve = elite_indices[:min(3, elite_size)]
            else:
                # On top-k of the current population
                indices_to_improve = torch.argsort(fitness, descending=True)[:5]
            
            for idx in indices_to_improve:
                # 2-opt
                improved, improved_fitness = local_search_2opt(
                    population[idx], W, k, total_weight, max_iters=20
                )
                # Swap
                improved, improved_fitness = local_search_swap(
                    improved, W, k, total_weight, max_iters=10
                )
                
                if improved_fitness > fitness[idx]:
                    population[idx] = improved
                    fitness[idx] = improved_fitness
            
            ls_end = time.time()
            local_search_total_time += (ls_end - ls_start)
        
        # Update best
        gen_best_idx = torch.argmax(fitness)
        if fitness[gen_best_idx] > best_fitness:
            best_fitness = fitness[gen_best_idx].item()
            best_individual = population[gen_best_idx].clone()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        best_history.append(best_fitness)
        
        # Diversity injection if stuck
        if no_improvement_count >= 20:
            print(f"[GA-ONLY] Injecting diversity at generation {gen}...")
            # Replace worst with new random individuals
            worst_indices = torch.argsort(fitness)[:pop_size // 4]
            for idx in worst_indices:
                new_ind = torch.randint(0, k, (n,), device=device)
                population[idx] = new_ind
                X = torch.nn.functional.one_hot(new_ind, num_classes=k).T.float()
                fitness[idx] = total_weight - torch.trace(X @ W @ X.T)
            no_improvement_count = 0
        
        if gen % max(1, n_generations // 10) == 0:
            avg_fitness = fitness.mean().item()
            print(f"[GA-ONLY] gen {gen}/{n_generations} | best = {best_fitness:.4f} | avg = {avg_fitness:.4f} | no_imp = {no_improvement_count}")
    
    # Final intensive local search
    print("[GA-ONLY] Final intensive local search...")
    ls_start = time.time()
    best_individual, best_fitness = local_search_2opt(
        best_individual, W, k, total_weight, max_iters=100
    )
    best_individual, best_fitness = local_search_swap(
        best_individual, W, k, total_weight, max_iters=50
    )
    ls_end = time.time()
    local_search_total_time += (ls_end - ls_start)
    
    # Timing report
    ga_end_time = time.time()
    ga_elapsed = ga_end_time - ga_start_time
    ga_pure_time = ga_elapsed - local_search_total_time
    total_time = ga_end_time - global_start_time
    
    print("\n" + "="*60)
    print("TIMING BREAKDOWN")
    print("="*60)
    print(f"GA phase:            {ga_pure_time:8.2f}s  ({ga_pure_time/total_time*100:5.1f}%)")
    print(f"Local Search:        {local_search_total_time:8.2f}s  ({local_search_total_time/total_time*100:5.1f}%)")
    print("-" * 60)
    print(f"TOTAL:               {total_time:8.2f}s  (100.0%)")
    print("="*60 + "\n")
    
    print(f"[GA-ONLY] Final best fitness: {best_fitness:.4f}")
    
    return best_individual

# ROS + MEMETIC GENETIC ALGORITHM + SPECIATION
def ros_mga(args, graph,
            pop_size=20,
            mga_iters=30,
            elite_size=6,
            crossover_rate=0.7,
            local_search_freq=3,
            n_species=3):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #  Phase 1: ROS Standard 
    W = get_matrix(args, graph)
    net, optimizer, edges, edges_weight, inputs = get_gnn_tuning(args, graph)
    
    net.load_state_dict(torch.load(
        f"gcn_model_ood_k{args.k}.pth",
        map_location=device,
        weights_only=True
    ))
    
    X_relaxed = run_gnn_tuning(
        args, W, edges, edges_weight,
        net, optimizer,
        args.epochs, args.tol, args.patience, inputs
    )
    
    k, n = X_relaxed.shape
    total_weight = W.sum()
    
    best_val = torch.inf
    ros_best_assign = None
    
    for it in range(args.max_iter):
        Xt = torch.nn.functional.one_hot(
            torch.multinomial(X_relaxed.T, 1).squeeze(),
            num_classes=args.k
        ).T.float()
        
        val = torch.trace(Xt @ W @ Xt.T)
        if val < best_val:
            best_val = val
            ros_best_assign = Xt.argmax(dim=0)
    
    ros_cut = total_weight - best_val
    print(f"[ROS-MGA] ROS baseline: {ros_cut:.2f}")
    
    #  Phase 2: Species Initialization 
    species_size = pop_size // n_species
    pop_discrete = []
    pop_cuts = []
    pop_species = []
    
    # Perturbation ranges proportional to the number of nodes
    perturbation_ranges = [
    (int(n * 0.01), int(n * 0.03)),  # Species 0: 1%–3% of the nodes
    (int(n * 0.03), int(n * 0.06)),  # Species 1: 3%–6% of the nodes 
    (int(n * 0.05), int(n * 0.10))   # Species 2: 5%–10% of the nodes 
    ]
    
    for species_id in range(n_species):
        p_min, p_max = perturbation_ranges[species_id]
        
        for i in range(species_size):
            solution = ros_best_assign.clone()
            
            n_changes = torch.randint(p_min, p_max + 1, (1,)).item()
            nodes_to_change = torch.randperm(n)[:n_changes]
            for node in nodes_to_change:
                solution[node] = torch.randint(0, k, (1,)).item()
            
            solution, cut_val = local_search_2opt(solution, W, k, total_weight, max_iters=50)
            
            pop_discrete.append(solution)
            pop_cuts.append(cut_val)
            pop_species.append(species_id)
    
    best_global_cut = max(pop_cuts)
    best_global_assign = pop_discrete[pop_cuts.index(best_global_cut)].clone()
    stagnation_count = 0
    
    # Fitness weights (fixed, the adaptive part is marginal)
    alpha, beta, gamma, delta = 0.75, 0.15, 0.08, 0.02
    
    elite_pool = []
    
    #  Phase 3: Evolution 
    for it in range(mga_iters):
        #  FITNESS with niching 
        pop_cuts_tensor = torch.tensor(pop_cuts, device=device)
        
        max_cut = max(pop_cuts)
        min_cut = min(pop_cuts)
        if max_cut > min_cut:
            normalized_cuts = (pop_cuts_tensor - min_cut) / (max_cut - min_cut)
        else:
            normalized_cuts = torch.ones_like(pop_cuts_tensor)
        
        fitness_within_species = torch.zeros(len(pop_cuts), device=device)
        for i in range(len(pop_cuts)):
            my_species = pop_species[i]
            beaten = sum(1 for j in range(len(pop_cuts)) 
                        if pop_species[j] == my_species and i != j and pop_cuts[i] > pop_cuts[j])
            total_in_species = sum(1 for j in range(len(pop_cuts)) if pop_species[j] == my_species and i != j)
            fitness_within_species[i] = beaten / max(total_in_species, 1)
        
        fitness_between_species = torch.zeros(len(pop_cuts), device=device)
        for i in range(len(pop_cuts)):
            my_species = pop_species[i]
            beaten = sum(1 for j in range(len(pop_cuts)) 
                        if pop_species[j] != my_species and pop_cuts[i] > pop_cuts[j])
            total_other_species = sum(1 for j in range(len(pop_cuts)) if pop_species[j] != my_species)
            fitness_between_species[i] = beaten / max(total_other_species, 1)
        
        novelty = torch.zeros(len(pop_cuts), device=device)
        for i in range(len(pop_cuts)):
            my_species = pop_species[i]
            total_diff = sum((pop_discrete[i] != pop_discrete[j]).sum().item()
                           for j in range(len(pop_cuts)) 
                           if pop_species[j] == my_species and i != j)
            count = sum(1 for j in range(len(pop_cuts)) if pop_species[j] == my_species and i != j)
            novelty[i] = (total_diff / max(count, 1)) / n
        
        fitness = (alpha * normalized_cuts + 
                  beta * fitness_within_species + 
                  gamma * fitness_between_species + 
                  delta * novelty)
        
        #  Best global 
        best_idx = pop_cuts.index(max(pop_cuts))
        if pop_cuts[best_idx] > best_global_cut:
            best_global_cut = pop_cuts[best_idx]
            best_global_assign = pop_discrete[best_idx].clone()
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        #  Elite selection 
        elite_discrete = []
        elite_cuts = []
        elite_species = []
        
        elites_per_species = max(1, elite_size // n_species)
        
        for species_id in range(n_species):
            species_indices = [i for i, s in enumerate(pop_species) if s == species_id]
            if not species_indices:
                continue
            
            species_fitness = fitness[species_indices]
            n_elite = min(elites_per_species, len(species_indices))
            elite_indices_in_species = species_fitness.topk(n_elite).indices.tolist()
            
            for elite_idx in elite_indices_in_species:
                global_idx = species_indices[elite_idx]
                elite_discrete.append(pop_discrete[global_idx])
                elite_cuts.append(pop_cuts[global_idx])
                elite_species.append(species_id)
        
        #  Local search + Path relinking 
        if it % local_search_freq == 0 and it > 0:
            for i in range(min(3, len(elite_discrete))):
                improved, improved_cut = local_search_2opt(
                    elite_discrete[i], W, k, total_weight, max_iters=60
                )
                if improved_cut > elite_cuts[i]:
                    elite_discrete[i] = improved
                    elite_cuts[i] = improved_cut
                    
                    if improved_cut > best_global_cut:
                        best_global_cut = improved_cut
                        best_global_assign = improved.clone()
                        stagnation_count = 0
            
            # Path relinking
            best_pr_cut = 0
            best_pr_sol = None
            
            for i in range(min(5, len(elite_discrete))):
                for j in range(i + 1, min(5, len(elite_discrete))):
                    diff = (elite_discrete[i] != elite_discrete[j]).sum().item()
                    if diff > n * 0.03:
                        pr_solution, pr_cut = path_relinking(
                            elite_discrete[i], 
                            elite_discrete[j], 
                            W, k, total_weight, max_steps=40
                        )
                        
                        if pr_cut > best_pr_cut:
                            best_pr_cut = pr_cut
                            best_pr_sol = pr_solution
            
            if best_pr_sol is not None and best_pr_cut > best_global_cut:
                print(f"[ROS-MGA] iter {it}: Path relinking {best_global_cut:.2f} -> {best_pr_cut:.2f}")
                best_global_cut = best_pr_cut
                best_global_assign = best_pr_sol.clone()
                stagnation_count = 0
        
        elite_pool = elite_discrete[:min(10, len(elite_discrete))]
        
        #  Evolution per species 
        new_pop_discrete = []
        new_pop_cuts = []
        new_pop_species = []
        
        for species_id in range(n_species):
            species_elite_discrete = [elite_discrete[i] for i, s in enumerate(elite_species) if s == species_id]
            species_elite_cuts = [elite_cuts[i] for i, s in enumerate(elite_species) if s == species_id]
            
            if not species_elite_discrete:
                continue
            
            new_pop_discrete.extend(species_elite_discrete)
            new_pop_cuts.extend(species_elite_cuts)
            new_pop_species.extend([species_id] * len(species_elite_discrete))
            
            n_offspring = species_size - len(species_elite_discrete)
            for _ in range(n_offspring):
                if torch.rand(1).item() < crossover_rate and len(species_elite_discrete) >= 2:
                    idx1, idx2 = torch.randperm(len(species_elite_discrete))[:2].tolist()
                    parent1_discrete = species_elite_discrete[idx1]
                    parent2_discrete = species_elite_discrete[idx2]
                    
                    if torch.rand(1).item() < 0.5:
                        child_discrete = discrete_crossover_cluster_preserving(parent1_discrete, parent2_discrete, W, k)
                    else:
                        child_discrete = discrete_crossover_uniform(parent1_discrete, parent2_discrete)
                else:
                    idx = torch.randint(0, len(species_elite_discrete), (1,)).item()
                    child_discrete = species_elite_discrete[idx].clone()
                    
                    n_mut = torch.randint(1, 4, (1,)).item()
                    mut_nodes = torch.randperm(n)[:n_mut]
                    for node in mut_nodes:
                        child_discrete[node] = torch.randint(0, k, (1,)).item()
                
                child_discrete, child_cut = local_search_2opt(child_discrete, W, k, total_weight, max_iters=40)
                
                new_pop_discrete.append(child_discrete)
                new_pop_cuts.append(child_cut)
                new_pop_species.append(species_id)
        
        pop_discrete = new_pop_discrete
        pop_cuts = new_pop_cuts
        pop_species = new_pop_species
        
        #  Kick moves 
        if stagnation_count > 8:
            for _ in range(3):
                kicked = best_global_assign.clone()
                n_kicks = torch.randint(int(n * 0.1), int(n * 0.2) + 1, (1,)).item()
                nodes_to_kick = torch.randperm(n)[:n_kicks]
                for node in nodes_to_kick:
                    kicked[node] = torch.randint(0, k, (1,)).item()
                
                kicked, kicked_cut = local_search_2opt(kicked, W, k, total_weight, max_iters=40)
                
                worst_idx = pop_cuts.index(min(pop_cuts))
                pop_discrete[worst_idx] = kicked
                pop_cuts[worst_idx] = kicked_cut
                
                if kicked_cut > best_global_cut:
                    best_global_cut = kicked_cut
                    best_global_assign = kicked.clone()
                    stagnation_count = 0
            
            if stagnation_count > 0:
                stagnation_count = 0
        
        # Log
        if it % max(1, mga_iters // 10) == 0:
            print(f"[ROS-MGA] iter {it}/{mga_iters}: best = {best_global_cut:.2f} | +{best_global_cut - ros_cut:.2f}")
    
    print(f"[ROS-MGA] Final: {best_global_cut:.2f} (+{best_global_cut - ros_cut:.2f} / +{(best_global_cut - ros_cut)/ros_cut*100:.2f}%)")
    
    return best_global_assign, X_relaxed

def ros_abc(args, graph, 
            pop_size=60,      # Number of Food Sources (and Employed Bees)
            max_iters=100, 
            limit=20):        # Number of failed attempts before becoming Scout
    """
    ROS + Artificial Bee Colony (ABC).
    Uses bee logic: 
    1. Employed Bees: Local refinement of assigned solutions.
    2. Onlooker Bees: Choose the best solutions and refine them further.
    3. Scout Bees: If a solution is stuck ('limit' reached), sample a new one from the GNN.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 1. GNN Phase (Common initial setup) ---
    W = get_matrix(args, graph)
    net, optimizer, edges, edges_weight, inputs = get_gnn_tuning(args, graph)
    
    try:
        net.load_state_dict(torch.load(f"gcn_model_ood_k{args.k}.pth", map_location=device, weights_only=True))
    except (FileNotFoundError, RuntimeError):
        pass # Training on the fly if missing

    print("[ROS-ABC] Running GNN...")
    X_relaxed = run_gnn_tuning(args, W, edges, edges_weight, net, optimizer, 
                               args.epochs, args.tol, args.patience, inputs)
    
    k_dim, n = X_relaxed.shape
    total_weight = W.sum()
    
    # --- 2. Population Initialization (Food Sources) ---
    # Sample everything from the GNN at the start
    population = []
    fitness = []
    trial_counters = torch.zeros(pop_size, device=device) # Count failures for each source
    
    print(f"[ROS-ABC] Initializing colony with {pop_size} bees...")
    
    # Initial batch generation
    probs_flat = X_relaxed.T  # (n, k)
    
    for i in range(pop_size):
        # Add some noise to diversify initial sources
        p_noisy = probs_flat + torch.rand_like(probs_flat) * 0.2
        p_noisy = p_noisy / p_noisy.sum(dim=1, keepdim=True)
        sol = torch.multinomial(p_noisy, 1).squeeze().to(device)
        
        X_mat = torch.nn.functional.one_hot(sol, num_classes=k_dim).T.float()
        cut = total_weight - torch.trace(X_mat @ W @ X_mat.T)
        
        population.append(sol)
        fitness.append(cut)

    fitness = torch.stack(fitness) # (pop_size,)
    best_val, best_idx = torch.max(fitness, 0)
    best_solution = population[best_idx].clone()
    print(f"[ROS-ABC] Best initial: {best_val:.4f}")

    # Improved helper: Crossover/Interaction with a partner
    def get_candidate(solution, partner, k_dim):
        """
        Generate a candidate solution using info from the partner (ABC logic).
        Try to inherit traits from the partner or explore nearby.
        """
        candidate = solution.clone()
        n = len(solution)
        
        # Find where solutions differ (conflict loci)
        diff_mask = (solution != partner)
        diff_indices = torch.nonzero(diff_mask).flatten()
        
        if len(diff_indices) > 0:
            # Interaction: copy N traits from the partner (simulate v_ij = x_ij + phi*(x_ij - x_kj))
            # Choose a random number of genes to swap
            n_changes = torch.randint(1, min(len(diff_indices), 20) + 1, (1,)).item()
            perm = torch.randperm(len(diff_indices))[:n_changes]
            indices_to_change = diff_indices[perm]
            candidate[indices_to_change] = partner[indices_to_change]
        else:
            # If they are identical or almost, pure random mutation (disturbance)
            n_changes = torch.randint(1, 5, (1,)).item()
            nodes = torch.randperm(n)[:n_changes]
            for node in nodes:
                current = candidate[node].item()
                new_c = (current + torch.randint(1, k_dim, (1,)).item()) % k_dim
                candidate[node] = new_c
                
        return candidate

    # --- 3. Main Loop ABC ---
    for it in range(max_iters):
        
        # --- PHASE 1: EMPLOYED BEES ---
        # Each bee selects a random partner different from itself
        for i in range(pop_size):
            # Select random partner k != i
            idxs = [x for x in range(pop_size) if x != i]
            k_idx = idxs[torch.randint(0, len(idxs), (1,)).item()]
            
            new_sol = get_candidate(population[i], population[k_idx], k_dim)
            
            # Calculate neighbor fitness
            X_new = torch.nn.functional.one_hot(new_sol, num_classes=k_dim).T.float()
            new_cut = total_weight - torch.trace(X_new @ W @ X_new.T)
            
            # Greedy selection
            if new_cut > fitness[i]:
                population[i] = new_sol
                fitness[i] = new_cut
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1

        # --- PHASE 2: ONLOOKER BEES ---
        # Probability based on fitness
        fit_vals = fitness - fitness.min() + 1e-6
        probs = fit_vals / fit_vals.sum()
        
        selected_indices = torch.multinomial(probs, pop_size, replacement=True)
        
        for idx in selected_indices:
            i = idx.item()
            # Onlooker tries to improve solution i using a random partner
            partner_idx = torch.randint(0, pop_size, (1,)).item()
            if partner_idx == i: partner_idx = (i + 1) % pop_size
            
            new_sol = get_candidate(population[i], population[partner_idx], k_dim)
            
            X_new = torch.nn.functional.one_hot(new_sol, num_classes=k_dim).T.float()
            new_cut = total_weight - torch.trace(X_new @ W @ X_new.T)
            
            if new_cut > fitness[i]:
                population[i] = new_sol
                fitness[i] = new_cut
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1

        # --- PHASE 3: SCOUT BEES ---
        n_scouts = 0
        for i in range(pop_size):
            if trial_counters[i] > limit:
                n_scouts += 1
                # Scout: reset using GNN sampling
                p_scout = X_relaxed.T + torch.rand_like(X_relaxed.T) * 0.5 
                p_scout = p_scout / p_scout.sum(dim=1, keepdim=True)
                new_sol = torch.multinomial(p_scout, 1).squeeze().to(device)
                
                X_new = torch.nn.functional.one_hot(new_sol, num_classes=k_dim).T.float()
                new_cut = total_weight - torch.trace(X_new @ W @ X_new.T)
                
                population[i] = new_sol
                fitness[i] = new_cut
                trial_counters[i] = 0
        
        # Update Best Global
        current_best_val, current_best_idx = torch.max(fitness, 0)
        
        # Mini local search on the best bee every now and then to avoid stagnation
        if it % 20 == 0:
             # Try to quickly refine the current best
             improved_best, improved_val = local_search_2opt(population[current_best_idx], W, k_dim, total_weight, max_iters=20)
             if improved_val > current_best_val:
                 population[current_best_idx] = improved_best
                 fitness[current_best_idx] = improved_val
                 current_best_val = improved_val

        if current_best_val > best_val:
            best_val = current_best_val
            best_solution = population[current_best_idx].clone()
            print(f"[ROS-ABC] Iter {it}: New Best found -> {best_val:.4f}")
        
        if it % 10 == 0:
            print(f"[ROS-ABC] Iter {it}/{max_iters} | Best: {best_val:.4f} | Mean: {fitness.mean():.2f} | Scouts sent: {n_scouts}")

    # Final greedy refinement on the best solution found
    print("[ROS-ABC] Final local search refinement on Global Best...")
    refined_sol, refined_cut = local_search_2opt(best_solution, W, k_dim, total_weight, max_iters=100)
    
    return refined_sol, X_relaxed
