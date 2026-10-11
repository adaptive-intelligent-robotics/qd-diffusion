import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional

import config

# ---- Visual Hyperparameters ----
FIG_WIDTH = 18
FIG_HEIGHT = 18
FIGURE_DPI = 300

# Font sizes
TITLE_FONT_SIZE = 36
TITLE_FONT_WEIGHT = 'bold' 
LABEL_FONT_SIZE = 36
LEGEND_FONT_SIZE = 28
MARKER_SIZE = 550
ANNOTATION_FONT_SIZE = 28

# Fixed color per algorithm key, consistent across all plots
ALGO_COLOR_MAP = {
    "dns_svd":                 "#1f77b4",  # Blue
    "cvt-map-elites":          "#d62728",  # Red
    "ablation":                "#aec7e8",  # Light blue
    "cma-es":                  "#2ca02c",  # Green
    "cma-mae":                 "#9467bd",  # Purple
    "adam":                    "#ff7f0e",  # Orange
    "qdhf":                    "#8c564b",  # Brown
    "cads":                    "#bcbd22",  # Olive
    "particle-guidance":       "#7f7f7f",  # Grey
    "particle-guidance-fixed": "#c7c7c7",  # Light grey
    "diverse-diffusion":       "#98df8a",  # Light green
    "dns_mixed":               "#ffbb78",  # Light orange
    "dns_random":              "#ff9896",  # Light red
    "dns_hidden_gaussian":     "#e377c2",  # Pink
    "dns_gaussian":            "#c49c94",  # Light brown
    "dns_isoline":             "#17becf",  # Cyan
}

def get_experiment_subdir(base_path: str) -> Optional[str]:
    """
    Finds the inner timestamped directory inside the algorithm folder.
    E.g. turns '.../seed/algo' into '.../seed/algo/2026-01-01_12345'
    """
    if not os.path.exists(base_path):
        return None
        
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    if not subdirs:
        return None
    
    subdirs.sort() 
    return os.path.join(base_path, subdirs[-1])


def load_final_metrics(
    root_dir: str,
    algorithm_dicts: Dict,
    metrics_list: List[str],
    seeds: List[int],
) -> Dict[str, Dict[str, float]]:
    """
    Loads the final (last logged) value for each metric for each algorithm,
    averaged across seeds.
    
    Returns:
        Dict mapping algorithm key -> {metric: mean_final_value}
    """
    
    final_values = {}
    
    for algo_key, algo_config in algorithm_dicts.items():
        print(f"\nProcessing Algorithm: {algo_config['label']} ({algo_key})")
        
        algo_path_name = algo_config["path"]
        
        # Store final values for each seed: {metric: [val_seed1, val_seed2, ...]}
        algo_seed_values = {m: [] for m in metrics_list}
        
        valid_seeds_count = 0

        for seed in seeds:
            # Construct the path to the Algorithm/Emitter folder
            initial_path = os.path.join(root_dir, str(seed), algo_path_name)
            
            # Find the timestamped subdirectory
            experiment_path = get_experiment_subdir(initial_path)
            
            if experiment_path is None:
                continue

            # Construct paths to actual files
            metrics_path = os.path.join(experiment_path, "metrics", "mapelites-logs.csv")
            check_file_path = os.path.join(experiment_path, "repertoire", "final_fitnesses.pt")

            # Check for successful run file
            if not os.path.exists(check_file_path):
                print(f"  !! WARNING: Seed {seed} missing 'final_fitnesses.pt' - Excluding.")
                continue
            
            if not os.path.exists(metrics_path):
                print(f"  !! WARNING: Seed {seed} missing metrics file - Excluding.")
                continue

            try:
                # Load Data
                df = pd.read_csv(metrics_path)
                
                # Get the final (last) row
                if len(df) == 0:
                    print(f"  !! WARNING: Empty metrics file for seed {seed} - Excluding.")
                    continue
                    
                final_row = df.iloc[-1]
                
                # Store final metric values
                for metric in metrics_list:
                    if metric == "num_solutions":
                        if 'coverage' in df.columns:
                            algo_seed_values[metric].append(final_row['coverage'] * 64)
                    elif metric == "total_fitnesses":
                        if 'avg_fitness' in df.columns and 'coverage' in df.columns:
                            algo_seed_values[metric].append(
                                final_row['avg_fitness'] * final_row['coverage'] * 64
                            )
                    elif metric in df.columns:
                        algo_seed_values[metric].append(final_row[metric])
                    else:
                        print(f"    !! Metric {metric} missing in CSV for seed {seed}")

                valid_seeds_count += 1

            except Exception as e:
                print(f"  !! Error processing seed {seed}: {e}")

        print(f"  -> Processed {valid_seeds_count} valid seeds.")

        # Compute mean across seeds for this algorithm
        if valid_seeds_count > 0:
            algo_means = {}
            for metric in metrics_list:
                if algo_seed_values[metric]:
                    algo_means[metric] = np.nanmean(algo_seed_values[metric])

            final_values[algo_key] = algo_means

    return final_values


def is_pareto_optimal(costs: np.ndarray, maximize: Tuple[bool, bool] = (True, True)) -> np.ndarray:
    """
    Determine which points are Pareto optimal for maximization problems.
    
    Args:
        costs: Nx2 array where each row is a point [metric1, metric2]
        maximize: Tuple of bools indicating whether to maximize each metric
        
    Returns:
        Boolean array indicating which points are Pareto optimal
    """
    # Convert to maximization problem if needed (negate if minimizing)
    costs_adj = costs.copy()
    if not maximize[0]:
        costs_adj[:, 0] = -costs_adj[:, 0]
    if not maximize[1]:
        costs_adj[:, 1] = -costs_adj[:, 1]
    
    is_optimal = np.ones(costs_adj.shape[0], dtype=bool)
    
    for i in range(len(costs_adj)):
        # Check if any other point dominates costs_adj[i]
        # For maximization: point j dominates i if j >= i in all objectives AND j > i in at least one
        for j in range(len(costs_adj)):
            if i == j:
                continue
            # Check if j dominates i
            if np.all(costs_adj[j] >= costs_adj[i]) and np.any(costs_adj[j] > costs_adj[i]):
                is_optimal[i] = False
                break
    
    return is_optimal


def plot_pareto_front(
    output_path: str,
    algorithm_dicts: Dict,
    final_values: Dict[str, Dict[str, float]],
    metric_x: str,
    metric_y: str,
    metric_x_label: str,
    metric_y_label: str,
    maximize_x: bool = True,
    maximize_y: bool = True,
    title: Optional[str] = None,
) -> None:
    """
    Plots a Pareto front for two metrics across algorithms.
    
    Args:
        output_path: Where to save the plot
        algorithm_dicts: Dictionary of algorithm configurations
        final_values: Dictionary mapping algo_key -> {metric: value}
        metric_x: Name of the metric for x-axis
        metric_y: Name of the metric for y-axis
        metric_x_label: Label for x-axis
        metric_y_label: Label for y-axis
        maximize_x: Whether metric_x should be maximized (True) or minimized (False)
        maximize_y: Whether metric_y should be maximized (True) or minimized (False)
        title: Optional plot title
    """
    
    print("\n-------------------------------------------------------------------------")
    print("                        Generating Pareto Front Plot                     ")
    print("-------------------------------------------------------------------------")
    
    # Assign colors: use the global map for known keys, fall back to extras for unknown ones
    algo_keys = list(algorithm_dicts.keys())
    fallback_colors = ["#000080", "#ffD700", "#00ff00"]
    fallback_idx = 0
    color_map = {}
    for key in algo_keys:
        if key in ALGO_COLOR_MAP:
            color_map[key] = ALGO_COLOR_MAP[key]
        else:
            color_map[key] = fallback_colors[fallback_idx % len(fallback_colors)]
            fallback_idx += 1
    
    # Collect data points
    points = []
    labels = []
    colors = []
    
    for algo_key, algo_config in algorithm_dicts.items():
        if algo_key not in final_values:
            continue
        
        metrics = final_values[algo_key]

        def resolve_metric(name):
            if name in metrics:
                return metrics[name]
            if name == "total_fitnesses":
                return metrics["avg_fitness"] * metrics["num_solutions"]
            raise KeyError(f"Metric '{name}' missing for {algo_key}")

        x_val = resolve_metric(metric_x)
        y_val = resolve_metric(metric_y)
        
        points.append([x_val, y_val])
        labels.append(algo_config["label"])
        colors.append(color_map[algo_key])
    
    if len(points) == 0:
        print("  !! No valid data points to plot")
        return
    
    points = np.array(points)
    
    # Determine Pareto optimal points
    is_optimal = is_pareto_optimal(points, maximize=(maximize_x, maximize_y))
    
    print(f"\nPareto Optimality Analysis:")
    print(f"Points shape: {points.shape}")
    print(f"Number of Pareto optimal points: {np.sum(is_optimal)}")
    for i, (label, opt, point) in enumerate(zip(labels, is_optimal, points)):
        print(f"  {label}: {'OPTIMAL' if opt else 'dominated'} - ({metric_x}={point[0]:.4f}, {metric_y}={point[1]:.4f})")
    
    # Pairwise dominance check
    print(f"\nDominance check (maximize_x={maximize_x}, maximize_y={maximize_y}):")
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            if i >= j:
                continue
            pt_i = points[i]
            pt_j = points[j]
            i_dominates_j = np.all(pt_i >= pt_j) and np.any(pt_i > pt_j)
            j_dominates_i = np.all(pt_j >= pt_i) and np.any(pt_j > pt_i)
            if i_dominates_j:
                print(f"  {label_i} dominates {label_j}")
            elif j_dominates_i:
                print(f"  {label_j} dominates {label_i}")
            else:
                print(f"  {label_i} and {label_j} are non-dominated")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # Plot each algorithm with its own colour so the legend can distinguish them.
    # Square marker = Pareto optimal, circle = dominated.
    for i, (point, label, color) in enumerate(zip(points, labels, colors)):
        marker = 's' if is_optimal[i] else 'o'
        linewidth = 2 if is_optimal[i] else 1.5
        ax.scatter(point[0], point[1],
                   color=color,
                   marker=marker,
                   s=MARKER_SIZE,
                   edgecolors='black',
                   linewidths=linewidth,
                   alpha=0.9,
                   label=label,
                   zorder=3 if is_optimal[i] else 2)

    # Draw Pareto front line
    if np.sum(is_optimal) > 1:
        pareto_points = points[is_optimal]
        sorted_indices = np.argsort(pareto_points[:, 0])
        pareto_sorted = pareto_points[sorted_indices]
        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
                'k--', alpha=0.5, linewidth=1.5, zorder=1)

    # Legend for algorithm names
    ax.legend(fontsize=LEGEND_FONT_SIZE,
              framealpha=0.95,
              edgecolor='#aaaaaa',
              loc='best')

    ax.margins(0.15)
    
    # Styling
    ax.set_xlabel(metric_x_label, fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.set_ylabel(metric_y_label, fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=28)
    
    if title:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=FIGURE_DPI)
    print(f"Plot saved to: {output_path}")
    plt.close()
    
    # Print Pareto optimal algorithms
    print("\nPareto Optimal Algorithms:")
    for i, (label, is_opt) in enumerate(zip(labels, is_optimal)):
        if is_opt:
            print(f"  - {label}: ({metric_x}={points[i, 0]:.4f}, {metric_y}={points[i, 1]:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the fitness/diversity Pareto front across algorithms.")
    config.add_model_argument(parser)
    args = parser.parse_args()

    cfg = config.get_model_config(args.model)
    algorithm_dicts = cfg["algorithms"]
    metrics_list = ["avg_fitness", "avg_pairwise_dist", "coverage", "num_solutions", "total_fitnesses"]

    final_values = load_final_metrics(
        root_dir=cfg["root_dir"],
        algorithm_dicts=algorithm_dicts,
        metrics_list=metrics_list,
        seeds=cfg["seeds"],
    )

    plot_pareto_front(
        output_path=os.path.join(cfg["output_dir"], "pareto_front_fitness_diversity.png"),
        algorithm_dicts=algorithm_dicts,
        final_values=final_values,
        metric_x="avg_pairwise_dist",
        metric_y="total_fitnesses",
        metric_x_label="Diversity (Avg. Pairwise Distance)",
        metric_y_label="Total Fitness",
        maximize_x=True,
        maximize_y=True,
        title="Pareto Front: Total Fitness vs Diversity",
    )
