# Enhancing ROS with Bio-Inspired approaches
Project for the course Bio-Inspired AI

## Description

This project implements several meta-heuristic algorithms to solve the Max-k-Cut problem using the ROS (Relax-Optimize-and-Sample) approach based on Graph Neural Networks (GNN).

## Installation


### Automatic Installation

The project includes an automatic installation script that handles all dependencies:

```bash
bash install.sh
```

The script will ask you to choose the appropriate version:
1. **CPU only** - For systems without CUDA GPU
2. **CUDA 11.8** - For GPUs with CUDA 11.8
3. **CUDA 12.1** - For GPUs with CUDA 12.1

The script will automatically install the needed dependencies.

### Installation Verification

After installation, verify everything is working correctly:

```bash
python -c 'import torch; import torch_geometric; print("Installation successful!")'
```

## Usage

### Available Algorithms

The project includes the following ROS approaches (situated in the ROS/ros/ros.py file):

1. **ros_aco_only** - ROS + Ant Colony Optimization 
2. **ros_aco** - ROS + Ant Colony Optimization with Local Search
3. **ros_ga** - ROS with Genetic Algorithm
4. **ros_ga_ls** - ROS + Genetic Algorithm with Local Search
5. **ros_mga** - ROS + Multi-species GA with Local Search

### Supported k Values

Each algorithm can be run with different k values (number of partitions):
- **k=2** - 2-way partitioning (classic Max-Cut)
- **k=3** - 3-way partitioning
- **k=10** - 10-way partitioning

### Batch Execution on Gset

To run an algorithm on a subset of Gset benchmarks (G1-G48), use the corresponding batch scripts:

#### Example: ROS+ACO+LS  with k=2
```bash
bash run_all_gset_ros_aco_ls_k2.sh
```

### Script Naming Convention

The scripts follow this naming scheme:

```
run_all_gset_ros_[ALGORITHM]_k[VALUE].sh
```

Where:
- `[ALGORITHM]` = aco_ls, aco_only, ga, ga_ls, mga
- `[VALUE]` = 2, 3, or 10

**Available scripts:**
- `run_all_gset_ros_aco_ls_k2.sh`, `run_all_gset_ros_aco_ls_k3.sh`, `run_all_gset_ros_aco_ls_k10.sh`
- `run_all_gset_ros_aco_only_k2.sh`, `run_all_gset_ros_aco_only_k3.sh`, `run_all_gset_ros_aco_only_k10.sh`
- `run_all_gset_ros_ga_ls_k2.sh`, `run_all_gset_ros_ga_ls_k3.sh`, `run_all_gset_ros_ga_ls_k10.sh`
- `run_all_gset_ros_ga_k2.sh`, `run_all_gset_ros_ga_k3.sh`, `run_all_gset_ros_ga_k10.sh`
- `run_all_gset_ros_mga_k2.sh`, `run_all_gset_ros_mga_k3.sh`, `run_all_gset_ros_mga_k10.sh`

### Results

Each batch script creates a dedicated directory with the results:

```
results_gset_ros_[ALGORITHM]_k[VALUE]/
├── execution_log.txt       # Detailed execution log
├── summary.csv             # Results summary in CSV format
├── G1_output.txt           # Output for graph G1
├── G2_output.txt           # Output for graph G2
└── ...
```

The `summary.csv` file contains:
- Graph name (G1, G2, ..., G48)
- Final result
- Relaxed result (ROS)
- Execution time (seconds)
- Status (SUCCESS/FAILED)

### Single Execution

To run a single test on a specific graph:

```bash
cd ROS
python main.py --alg ros_aco --graph_type gset --gset 1 --weight_mode 1 --k 2
```

Parameters:
- `--alg`: Algorithm to use (ros_aco, ros_ga, ros_mga, etc.)
- `--graph_type`: Graph type (gset for benchmarks)
- `--gset`: Gset graph number (1-48)
- `--weight_mode`: 1 to use original weights
- `--k`: Number of partitions (2, 3, or 10)

## Project Structure

```
Max-k-Cut/
├── README.md              # This file
├── install.sh             # Installation script
├── Gset/                  # Gset benchmark dataset
├── ROS/                   # ROS algorithms implementation
│   ├── main.py           # Main entry point
│   ├── ros/              # ROS algorithm modules
│   └── utils.py          # Utility functions
└── run_all_gset_*.sh     # Batch scripts for complete execution
```

## Notes

- Batch scripts must be run from the project root directory
- Execution times vary based on graph size and chosen algorithm
- For large graphs, execution may take several hours
- GPU usage is recommended when available to accelerate GNN training

## References

For more details on the ROS approach and implemented algorithms, please refer to the reference papers in the `ROS/` directory.
