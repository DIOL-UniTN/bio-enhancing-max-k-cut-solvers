"""
Analisi Completa ACO - Boxplot Gset
====================================

Questo script esegue un'analisi completa confrontando:
- ROS (baseline)
- ROS+LS (ROS + Local Search)
- ROS+ACO+LS (ROS + ACO + Local Search)

Genera un boxplot aggregato su tutti i grafi Gset.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import sys
from pathlib import Path
from utils import generate_graph, postprocess, set_random_seed
from add_parser import add_parse
from ros.ros import ros, ros_aco, ros_local_search
import json


def load_gset_graphs():
    """
    Carica 3 grafi rappresentativi dal dataset Gset (piccolo, medio, grande)
    """
    gset_dir = Path("../Gset")
    if not gset_dir.exists():
        print(f"Directory Gset non trovata: {gset_dir}")
        return []
    
    # Seleziona 3 grafi rappresentativi di dimensioni e densità diverse
    # G1: 800 nodi, 19176 archi → densità alta (~6%)
    # G14: 800 nodi, 4694 archi → densità bassa (~1.5%)
    # G55: 5000 nodi, 12498 archi → grafo grande e sparso (~0.1%)
    selected_graphs = ["G1", "G14", "G55"]
    
    gset_files = []
    for graph_name in selected_graphs:
        graph_path = gset_dir / graph_name
        if graph_path.exists():
            gset_files.append(graph_path)
        else:
            print(f"Grafo {graph_name} non trovato, saltato")
    
    print(f"Selezionati {len(gset_files)} grafi Gset: {selected_graphs}")
    return gset_files


def run_algorithm(args, graph, algorithm_name):
    """
    Esegue un algoritmo specifico e ritorna il valore del cut
    
    Args:
        args: Argomenti
        graph: Grafo NetworkX
        algorithm_name: "ros", "ros_local_search", o "ros_aco"
    
    Returns:
        cut_value: Valore del max-cut trovato
    """
    try:
        if algorithm_name == "ros":
            result, result_relaxed = ros(args, graph)
        elif algorithm_name == "ros_local_search":
            result, result_relaxed = ros_local_search(args, graph)
        elif algorithm_name == "ros_aco":
            result, result_relaxed, phase_tracking = ros_aco(args, graph)
        else:
            raise ValueError(f"Algoritmo sconosciuto: {algorithm_name}")
        
        if isinstance(result, float):
            return np.inf
        
        cut_value = postprocess(result, graph)
        return cut_value
    
    except Exception as e:
        print(f"Errore durante l'esecuzione di {algorithm_name}: {e}")
        return np.inf


def run_comprehensive_analysis(args, gset_files, num_runs=5):
    """
    Esegue l'analisi completa su tutti i grafi Gset
    
    Args:
        args: Argomenti
        gset_files: Lista di file Gset
        num_runs: Numero di esecuzioni per ogni grafo/algoritmo
    
    Returns:
        results_df: DataFrame con tutti i risultati
    """
    results = []
    
    algorithms = {
        "ROS": "ros",
        "ROS+LS": "ros_local_search",
        "ROS+ACO+LS": "ros_aco"
    }
    
    total_experiments = len(gset_files) * len(algorithms) * num_runs
    current = 0
    
    print(f"\n{'='*80}")
    print(f"INIZIO ANALISI COMPLETA")
    print(f"{'='*80}")
    print(f"Grafi: {len(gset_files)}")
    print(f"Algoritmi: {len(algorithms)}")
    print(f"Esecuzioni per configurazione: {num_runs}")
    print(f"Esperimenti totali: {total_experiments}")
    print(f"{'='*80}\n")
    
    for gset_file in gset_files:
        graph_name = gset_file.name
        print(f"\nProcessando grafo: {graph_name}")
        
        # Aggiorna args per il grafo corrente
        args.graph_type = "gset"
        args.graph_name = graph_name
        
        for display_name, alg_name in algorithms.items():
            print(f"  > Algoritmo: {display_name}")
            
            for run in range(num_runs):
                current += 1
                
                # Set seed diverso per ogni run
                seed = args.seed + run
                set_random_seed(seed)
                
                # Genera il grafo
                graph = generate_graph(args)
                
                # Esegui algoritmo
                cut_value = run_algorithm(args, graph, alg_name)
                
                # Salva risultato
                results.append({
                    'graph': graph_name,
                    'algorithm': display_name,
                    'run': run + 1,
                    'cut_value': cut_value
                })
                
                progress = (current / total_experiments) * 100
                print(f"    Run {run+1}/{num_runs}: {cut_value:.2f} [{progress:.1f}% completato]")
    
    # Crea DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print(f"ANALISI COMPLETATA")
    print(f"{'='*80}\n")
    
    return results_df


def create_boxplot(results_df, output_file="boxplot_gset_cuts.png"):
    """
    Crea il boxplot principale aggregato su tutti i grafi Gset
    
    Args:
        results_df: DataFrame con i risultati
        output_file: Nome del file di output
    """
    print(f"\nGenerazione boxplot...")
    
    # Filtra valori infiniti
    valid_results = results_df[results_df['cut_value'] != np.inf].copy()
    
    if len(valid_results) == 0:
        print("Nessun risultato valido da visualizzare!")
        return
    
    # Statistiche di base
    print("\nSTATISTICHE DESCRITTIVE:")
    print("="*80)
    stats = valid_results.groupby('algorithm')['cut_value'].describe()
    print(stats)
    print()
    
    # Crea figura
    plt.figure(figsize=(12, 8))
    
    # Imposta stile seaborn
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    
    # Ordine degli algoritmi
    algorithm_order = ["ROS", "ROS+LS", "ROS+ACO+LS"]
    
    # Crea boxplot
    ax = sns.boxplot(
        data=valid_results,
        x='algorithm',
        y='cut_value',
        order=algorithm_order,
        width=0.6,
        linewidth=2
    )
    
    # Aggiungi punti individuali (opzionale, commentato per evitare sovrapposizione)
    # sns.stripplot(
    #     data=valid_results,
    #     x='algorithm',
    #     y='cut_value',
    #     order=algorithm_order,
    #     color='black',
    #     alpha=0.3,
    #     size=3
    # )
    
    # Personalizza grafico
    plt.xlabel('Algoritmo', fontsize=14, fontweight='bold')
    plt.ylabel('Valore del Cut', fontsize=14, fontweight='bold')
    plt.title('Confronto Algoritmi ROS - Valori del Cut su Gset', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Griglia
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Aggiungi numero di campioni
    for i, alg in enumerate(algorithm_order):
        n_samples = len(valid_results[valid_results['algorithm'] == alg])
        median_val = valid_results[valid_results['algorithm'] == alg]['cut_value'].median()
        plt.text(i, median_val, f'n={n_samples}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Layout
    plt.tight_layout()
    
    # Salva figura
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Boxplot salvato in: {output_file}")
    
    # Mostra figura
    plt.show()
    plt.close()


def save_results(results_df, output_file="results_gset_comprehensive.csv"):
    """
    Salva i risultati in formato CSV e JSON
    """
    # CSV
    results_df.to_csv(output_file, index=False)
    print(f"Risultati salvati in CSV: {output_file}")
    
    # JSON con statistiche
    json_file = output_file.replace('.csv', '.json')
    summary = {
        'total_experiments': len(results_df),
        'algorithms': results_df['algorithm'].unique().tolist(),
        'graphs': results_df['graph'].unique().tolist(),
        'statistics': {}
    }
    
    valid_results = results_df[results_df['cut_value'] != np.inf]
    
    for alg in results_df['algorithm'].unique():
        alg_data = valid_results[valid_results['algorithm'] == alg]['cut_value']
        summary['statistics'][alg] = {
            'mean': float(alg_data.mean()),
            'median': float(alg_data.median()),
            'std': float(alg_data.std()),
            'min': float(alg_data.min()),
            'max': float(alg_data.max()),
            'count': int(len(alg_data))
        }
    
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Statistiche salvate in JSON: {json_file}")


def main():
    """
    Funzione principale
    """
    # Configura logging
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
    
    # Parse argomenti
    args = add_parse()
    
    # Configurazione device
    args.TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.TORCH_DTYPE = torch.float32
    
    # Configurazione specifica per Gset
    args.weight_mode = 1  # Necessario per Gset (weights ±1)
    
    print(f"\nDevice: {args.TORCH_DEVICE}")
    
    # Carica grafi Gset
    gset_files = load_gset_graphs()
    
    if len(gset_files) == 0:
        print("Nessun grafo Gset trovato!")
        return
    
    # Esegui analisi con 3 run per algoritmo
    num_runs = 3
    results_df = run_comprehensive_analysis(args, gset_files, num_runs=num_runs)
    
    # Salva risultati
    save_results(results_df)
    
    # Crea boxplot
    create_boxplot(results_df)
    
    print("\nANALISI COMPLETA TERMINATA!\n")


if __name__ == "__main__":
    main()
