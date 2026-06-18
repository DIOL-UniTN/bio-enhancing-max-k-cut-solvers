import argparse


def add_parse():
    # alg: pmd, pgp, md, pg, al, fw
    parser = argparse.ArgumentParser(description="parser")


    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--alg", type=str, default="ros")
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=int(1e4))
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--soft", type=float, default=1)
    parser.add_argument("--epsilon_PMD", type=float, default=1e-8)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--gset", type=str, default="1")

    parser.add_argument("--pretraining_graphnum", type=int, default=500)
    parser.add_argument("--pretraining_epochs", type=int, default=1)
    parser.add_argument("--dim_embedding", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=100)


  # phase analysis
    parser.add_argument("--param-analysis", action='store_true', help="Run parameter analysis")
    
    # genetic

    parser.add_argument("--POPSIZE", type=int, default=30)
    parser.add_argument("--MAXGENS", type=int, default=200)
    parser.add_argument("--PXOVER", type=float, default=0.77)
    parser.add_argument("--PMUTATION", type=float, default=0.3)
    parser.add_argument("--PSELECT", type=float, default=0.80)

    parser.add_argument("--sol_dir", type=str, default="../ANYCSP/solution.pickle")
    parser.add_argument("--checkpoint", type=str, default='best', help="Name of the checkpoint")
    parser.add_argument("--model_dir", type=str, help="Model directory")
    parser.add_argument("--num_boost", type=int, default=20, help="Number of parallel evaluate runs")
    parser.add_argument("--verbose", action='store_true', default=True, help="Output intermediate optima")
    parser.add_argument("--network_steps", type=int, default=1000, help="Number of network steps during evaluation")
    parser.add_argument("--timeout", type=float, default=180, help="Timeout in seconds")

    # Graph Type
    
    parser.add_argument("--graph_type", type=str, default="reg")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--weight_mode", type=int, default=0) # 0: edge_weights all 1; 1: edge_weights all \pm 1; 2: arbitrary edge_weights uniform distribution [-lrange, rrange]
    parser.add_argument("--lrange", type=float, default=1)
    parser.add_argument("--rrange", type=float, default=1)
    parser.add_argument("--graph_path", type=str, default="")
    parser.add_argument("--Gset", default=False)
    

    # Parameters for (ACO)
    parser.add_argument('--n_ants', type=int, default=50, help='Number of ants for ACO')
    parser.add_argument('--n_iter', type=int, default=100, help='Number of iterations for ACO')

    args = parser.parse_args()
    return args
