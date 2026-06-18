# Enhancing Max-k-Cut Solvers through Bio-Inspired Approaches

Code for the BIOMAP 2026 workshop paper, part of the BIO-inspired Methods for Pattern Recognition track at ICPR 2026.

## Authors

- Pietro De Angeli
- Giorgia Gabardi
- Alberto Vendramini
- Erik Nielsen
- Stefano Genetti
- Giovanni Iacca

Pietro De Angeli, Giorgia Gabardi, and Alberto Vendramini contributed equally to this work.

Department of Information Engineering and Computer Science, University of Trento, Italy

Emails:

- pietro.deangeli, giorgia.gabardi, alberto.vendramini@studenti.unitn.it
- erik.nielsen, stefano.genetti, giovanni.iacca@unitn.it

## Overview

This repository studies a hybrid solver for the Max-$k$-Cut problem. The pipeline starts from solutions produced by ROS (Relax-Optimize-and-Sample), a Graph Neural Network-based framework, and refines them with bio-inspired optimization methods.

The paper evaluates three main families of refiners:

1. Genetic Algorithm (GA)
2. Ant Colony Optimization (ACO)
3. Multi-species Genetic Algorithm (MGA)

Local search is also used in the hybrid variants, and the experiments focus on the Gset benchmark for $k \in \{2, 3, 10\}$.

## Repository Layout

- `ROS/`: implementation of ROS and the bio-inspired refiners
- `Gset/`: benchmark instances used in the experiments
- `run_all_gset_*.sh`: batch scripts for running the reported experiments
- `install.sh`: helper script for environment setup

## Installation

The easiest way to set up the project is with the provided installer:

```bash
bash install.sh
```

During installation you will be asked to choose the target environment:

1. CPU only
2. CUDA 11.8
3. CUDA 12.1

If you already manage environments manually, you can also use the dependencies listed in `ROS/requirements.txt` or `ROS/environment.yml`.

After installation, a quick sanity check is:

```bash
python -c "import torch; import torch_geometric; print('Installation successful!')"
```

## Running the Code

All executable entry points are under `ROS/`. The main script accepts the algorithm name through `--alg` and the benchmark configuration through the remaining arguments.

### Single Run

Example: run ROS with ACO refinement on the first Gset instance for Max-2-Cut.

```bash
cd ROS
python main.py --alg ros_aco --graph_type gset --gset 1 --weight_mode 1 --k 2
```

Common arguments:

- `--alg`: algorithm to run, for example `ros`, `ros_aco`, `ros_aco_only`, `ros_ga`, `ros_ga_improved`, `ros_mga`, `ros_local_search`, `ros_vanilla`, `ros_abc`, or `ga_only`
- `--graph_type`: graph family, typically `gset`
- `--gset`: Gset instance index
- `--weight_mode`: use the original benchmark weights with `1`
- `--k`: number of partitions

### Batch Experiments

The top-level `run_all_gset_*.sh` scripts reproduce the benchmark runs reported in the paper. Their naming convention is:

```text
run_all_gset_ros_[method]_k[value].sh
```

where `[method]` is one of `aco_ls`, `aco_only`, `ga`, `ga_ls`, or `mga`, and `[value]` is `2`, `3`, or `10`.

Example:

```bash
bash run_all_gset_ros_ga_ls_k2.sh
```

Each batch script creates a results directory such as `results_gset_ros_ga_ls_k2/` containing:

- `execution_log.txt`: full terminal log
- `summary.csv`: per-instance summary
- `G*_output.txt`: output for each Gset instance

## Paper-Focused Methods

The paper’s experiments compare ROS against the following post-optimizers:

- `ros_ga` and `ros_ga_improved`: GA-based refiners used for the paper’s genetic-algorithm experiments
- `ros_aco` and `ros_aco_only`: ACO-based refiners, with and without local search
- `ros_mga`: multi-species GA with local search and explicit diversity preservation

The codebase also contains related variants used for additional experimentation and ablation studies.

## Notes

- The experiments were run on the Gset subset `G1` to `G48`.
- GPU acceleration is recommended when available, since ROS relies on GNN inference.
- The hybrid solvers are computation-heavy; runtime grows with graph size, `k`, and the amount of local search.

## Citation

If you use this code, please cite the BIOMAP 2026 workshop paper:

> Enhancing Max-$k$-Cut Solvers through Bio-Inspired Approaches

## Acknowledgements

This repository builds on the public ROS implementation and adapts it for bio-inspired refinement experiments on Max-$k$-Cut.
