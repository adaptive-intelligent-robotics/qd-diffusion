import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional
import textwrap

# ---- Visual Hyperparameters ----
FIG_WIDTH = 18
FIG_HEIGHT = 18
FIGURE_DPI = 300

# Font sizes
TITLE_FONT_SIZE = 36
TITLE_FONT_WEIGHT = 'bold' 
LABEL_FONT_SIZE = 36
LEGEND_FONT_SIZE = 28
MARKER_SIZE = 350
ANNOTATION_FONT_SIZE = 28

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
    seeds: range,
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
                    if metric in df.columns:
                        algo_seed_values[metric].append(final_row[metric])
                    elif metric == "num_solutions":
                        if 'coverage' in df.columns:
                            algo_seed_values[metric].append(final_row['coverage'] * 64)
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
                    # algo_means[metric] = np.nanmean(algo_seed_values[metric])
                    algo_means[metric] = np.nanmedian(algo_seed_values[metric])

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
    
    # Setup colors using custom palette
    algo_keys = list(algorithm_dicts.keys())
    custom_colors = [
        "#1f77b4", # Blue
        "#d62728", # Red
        "#2ca02c", # Green
        "#9467bd", # Purple
        "#ff7f0e", # Orange
        "#8c564b", # Brown
        "#e377c2", # Pink
        "#7f7f7f", # Gray
        "#bcbd22", # Olive
        "#17becf", # Cyan
        "#000000", # Black
        "#8c000f", # Dark Red
        "#000080", # Navy
        "#ffD700", # Gold
        "#00ff00", # Lime
    ]
    color_map = {key: custom_colors[i % len(custom_colors)] for i, key in enumerate(algo_keys)}
    
    # Collect data points
    points = []
    labels = []
    colors = []
    
    for algo_key, algo_config in algorithm_dicts.items():
        if algo_key not in final_values:
            continue
        
        metrics = final_values[algo_key]
        
        if metric_x == "total_fitnesses":
            x_val = metrics["avg_fitness"] * 64 * metrics["coverage"]
        else:
            x_val = metrics[metric_x]
            
        if metric_y == "total_fitnesses":
            y_val = metrics["avg_fitness"] * 64 * metrics["coverage"]
        else:
            y_val = metrics[metric_y]
        
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
    
    # Debug: Check pairwise dominance
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
    
    # Single color scheme
    pareto_color = '#2E86AB'  # Blue
    dominated_color = '#A4B8C4'  # Light blue-gray
    
    # Plot all points first
    for i, (point, label, color) in enumerate(zip(points, labels, colors)):
        marker = 's' if is_optimal[i] else 'o'  # Square for Pareto, circle for dominated
        linewidth = 2 if is_optimal[i] else 1.5
        point_color = pareto_color if is_optimal[i] else dominated_color
        
        ax.scatter(point[0], point[1], 
                  color=point_color, 
                  marker=marker,
                  s=MARKER_SIZE,
                  edgecolors='black',
                  linewidths=linewidth,
                  alpha=0.9,
                  zorder=3 if is_optimal[i] else 2)
    
    # Draw Pareto front line before annotations
    if np.sum(is_optimal) > 1:
        pareto_points = points[is_optimal]
        # Sort by x-coordinate for proper line drawing
        sorted_indices = np.argsort(pareto_points[:, 0])
        pareto_sorted = pareto_points[sorted_indices]
        
        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 
               'k--', alpha=0.5, linewidth=1.5, zorder=1)
    
    # Now add annotations with smart positioning
    annotations = []
    label_boxes = []  # Track label bounding boxes to avoid overlaps
    
    # Process points in order: Pareto optimal first, then by x-coordinate
    point_order = np.argsort([not is_optimal[i] for i in range(len(points))])
    
    for idx in point_order:
        point = points[idx]
        label = labels[idx]
        color = colors[idx]
        i = idx
        
        # Wrap long labels (narrower for better fit)
        wrapped_label = '\n'.join(textwrap.wrap(label, width=9, break_long_words=False))
        
        # Smart positioning based on location in plot
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        x_center = (points[:, 0].min() + points[:, 0].max()) / 2
        y_center = (points[:, 1].min() + points[:, 1].max()) / 2
        
        # Identify point groups based on position
        is_top_right = point[1] > y_center + 0.3 * y_range and point[0] > x_center + 0.2 * x_range
        is_bottom = point[1] < points[:, 1].min() + 0.3 * y_range
        
        # Check if point is in middle cluster
        distances_to_others = [np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2) 
                              for j, p in enumerate(points) if j != idx]
        min_distance = min(distances_to_others) if distances_to_others else float('inf')
        is_clustered = min_distance < 0.15 * np.sqrt(x_range**2 + y_range**2)
        
        # Use custom offsets based on point position and label name
        if 'h-layer' in label or 'Intermed' in label:
            # h-layer Gaussian: label NE (northeast) and much closer
            base_offset = 50
            offset_options = [(int(base_offset*0.8), int(base_offset*0.8)), (int(base_offset*1.0), int(base_offset*1.0)), (int(base_offset*0.6), int(base_offset*1.0))]
            ha = 'left'
        elif 'Isoline' in label:
            # Isoline-DD: label SE (southeast)
            base_offset = 70
            offset_options = [(int(base_offset*1.2), -int(base_offset*1.0)), (int(base_offset*1.0), -int(base_offset*1.2)), (int(base_offset*1.4), -int(base_offset*0.8))]
            ha = 'left'
        elif is_top_right:
            # Top right point: label to the SOUTHWEST (further below and to the left)
            base_offset = 100
            offset_options = [(-base_offset, -int(base_offset*1.2)), (-int(base_offset*1.15), -int(base_offset*1.3)), (-int(base_offset*1.3), -int(base_offset*1.1))]
            ha = 'right'
        elif is_bottom:
            # Bottom points: labels near points
            base_offset = 60
            offset_options = [(base_offset, base_offset), (-base_offset, base_offset), (0, base_offset), (base_offset, -base_offset)]
            ha = 'left'
        elif is_clustered:
            # Middle cluster: Force specific positions based on order
            base_offset = 100
            # Sort cluster points by x-coordinate to get consistent ordering
            cluster_points_indices = []
            for j in range(len(points)):
                p_distances = [np.sqrt((points[j][0] - p[0])**2 + (points[j][1] - p[1])**2) 
                              for k, p in enumerate(points) if k != j]
                min_d = min(p_distances) if p_distances else float('inf')
                if min_d < 0.15 * np.sqrt(x_range**2 + y_range**2):
                    cluster_points_indices.append(j)
            
            cluster_points_indices.sort(key=lambda i: points[i][0])
            position_in_cluster = cluster_points_indices.index(idx) if idx in cluster_points_indices else 0
            
            if position_in_cluster == 0:  # Leftmost cluster point - label LEFT
                offset_options = [(-int(base_offset*2.0), 0), (-int(base_offset*2.2), int(base_offset*0.3)), (-int(base_offset*1.8), -int(base_offset*0.3))]
                ha = 'right'
            elif position_in_cluster == 1:  # Middle cluster point - label ABOVE
                offset_options = [(0, int(base_offset*2.0)), (int(base_offset*0.3), int(base_offset*2.2)), (-int(base_offset*0.3), int(base_offset*2.2))]
                ha = 'center'
            else:  # Rightmost cluster point - label RIGHT
                offset_options = [(int(base_offset*2.0), 0), (int(base_offset*2.2), int(base_offset*0.3)), (int(base_offset*1.8), -int(base_offset*0.3))]
                ha = 'left'
        else:
            # Default positioning
            base_offset = 80
            offset_options = [(base_offset, base_offset), (-base_offset, base_offset), (base_offset, -base_offset), (-base_offset, -base_offset)]
            ha = 'center'
        
        # Find best offset that doesn't overlap and spreads labels out
        best_offset = offset_options[0]
        # Estimate label dimensions with extremely large padding to prevent overlaps
        max_line_length = max(len(line) for line in wrapped_label.split('\n'))
        label_width = max_line_length * 0.065 * x_range
        label_height = len(wrapped_label.split('\n')) * 0.20 * y_range
        label_width *= 3.5
        label_height *= 3.5
        
        # Score each offset option
        best_score = -float('inf')
        for offset in offset_options:
            overlaps = False
            label_x = point[0] + offset[0] / 45.0 * (x_range / 4.0)
            label_y = point[1] + offset[1] / 45.0 * (y_range / 4.0)
            
            # Check for overlaps
            for ex_x, ex_y, ex_w, ex_h in label_boxes:
                dx = abs(label_x - ex_x)
                dy = abs(label_y - ex_y)
                if dx < (label_width + ex_w) / 2 * 3.5 and dy < (label_height + ex_h) / 2 * 3.5:
                    overlaps = True
                    break
            
            if not overlaps:
                # Check that label position is far enough from ALL data points
                covers_point = False
                for p in points:
                    # Calculate distance from point to label position in data coordinates
                    pt_to_label_dist = np.sqrt((label_x - p[0])**2 + (label_y - p[1])**2)
                    # Require label to be at least 0.45 of diagonal distance away from any point
                    min_dist = 0.45 * np.sqrt(x_range**2 + y_range**2)
                    if pt_to_label_dist < min_dist:
                        covers_point = True
                        break
                
                if covers_point:
                    continue  # Try next offset
                
                # Calculate diversity score - strongly prefer directions different from ALL labels
                diversity_score = 0
                for j, (ex_x, ex_y, ex_w, ex_h) in enumerate(label_boxes):
                    # Angle difference between this offset and existing label offset
                    angle1 = np.arctan2(offset[1], offset[0])
                    angle2 = np.arctan2(ex_y - point[1], ex_x - point[0])
                    angle_diff = abs(angle1 - angle2)
                    angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Handle wraparound
                    # Strongly reward different directions
                    diversity_score += angle_diff ** 2  # Square to emphasize larger differences
                
                # Prefer offsets with higher diversity score
                if diversity_score > best_score:
                    best_score = diversity_score
                    best_offset = offset
        
        # Store the final label box position
        final_label_x = point[0] + best_offset[0] / 45.0 * (x_range / 4.0)
        final_label_y = point[1] + best_offset[1] / 45.0 * (y_range / 4.0)
        label_boxes.append((final_label_x, final_label_y, label_width, label_height))
        
        if len(label_boxes) == len(point_order):  # All attempted, none worked
            label_boxes.append((label_x, label_y))
        
        xytext = best_offset
        
        # Add text label with visible connecting line
        ann = ax.annotate(wrapped_label, 
                   xy=(point[0], point[1]),
                   xytext=xytext,
                   textcoords='offset points',
                   fontsize=ANNOTATION_FONT_SIZE,
                   fontweight='bold',
                   ha=ha,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.95, linewidth=1.5),
                   arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0', color='black', lw=1.0, alpha=0.7),
                   zorder=4)
        annotations.append(ann)
    
    # Styling
    ax.set_xlabel(metric_x_label, fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.set_ylabel(metric_y_label, fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=28)
    
    if title:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add larger margins to prevent point clipping at edges
    ax.margins(0.08)
    
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
    """
    Example usage - modify these paths and settings to match your setup
    """
    
    # Example configuration
    ROOT_DIR = "/workspace/src/hpc_results/output/sdxl-turbo/"
    ANALYSIS_DIR = "analysis_output"
    SEEDS = range(1, 37)
    
    baseline_dicts = {
        "dns_svd": {
            "path": "dns/multi-hidden-layer-multi-svd-alpha-10",
            "label": "Spectral-Guided QD-Diffusion (Ours)",
            "grid_plot_linestyle": "solid",
        },  
        "cvt-map-elites": {
            "path": "cvt-map-elites/multi-hidden-layer-multi-svd-alpha-10",
            "label": "CVT MAP-Elites",
            "grid_plot_linestyle": "solid",
        },
        "ablation": {   
            "path": "dns/multi-hidden-layer-multi-svd-threshold-0-alpha-10",
            "label": "Fitness Threshold Ablation",
            "grid_plot_linestyle": "solid",
        },
        "cma-es": {
            "path": "dns/multi-sep-cma-es",
            "label": "Sep-CMA-ES",
            "grid_plot_linestyle": "solid",
        },
            "adam": {
            "path": "dns/multi-adam",
            "label": "Adam",
            "grid_plot_linestyle": "solid",
        },
        "qdhf": {
            "path": "qdhf/multi-hidden-layer-multi-svd-alpha-10",
            "label": "QDHF",
            "grid_plot_linestyle": "solid",
        },
    }


    # Define metrics to analyze
    metrics_list = ["avg_fitness", "avg_pairwise_dist", "coverage", "num_solutions"]
    
    # Load final values
    final_values = load_final_metrics(
        root_dir=ROOT_DIR,
        algorithm_dicts=baseline_dicts,
        metrics_list=metrics_list,
        seeds=SEEDS,
    )
    
    # Plot Pareto front: Average Fitness vs Diversity
    output_path = os.path.join(ANALYSIS_DIR, "pareto_front_fitness_diversity.png")
    plot_pareto_front(
        output_path=output_path,
        algorithm_dicts=baseline_dicts,
        final_values=final_values,
        metric_x="avg_pairwise_dist",
        metric_y="total_fitnesses",
        metric_x_label="Diversity (Avg. Pairwise Distance)",
        metric_y_label="Total Fitness",
        maximize_x=True,
        maximize_y=True,
        title="Pareto Front: Total Fitness vs Diversity"
    )


    emitter_dicts = {
        "dns_svd": {
            "path": "dns/multi-hidden-layer-multi-svd-alpha-10",
            "label": "Spectral-Guided QD-Diffusion (Ours)",
            "grid_plot_linestyle": "solid",
        }, 
        "dns_random": {
            "path": "dns/random-seeds", 
            "label": "Random Seeds",
            "grid_plot_linestyle": "solid",
        },
        "dns_hidden_gaussian": {
            "path": "dns/multi-hidden-layer-gaussian",
            "label": "Intermed.-layer Gaussian",
            "grid_plot_linestyle": "solid",
        },
        "dns_mixed": {
            "path": "dns/mixed-all",
            "label": "ImageBreeder",
            "grid_plot_linestyle": "solid",
        },
        "dns_gaussian": {
            "path": "dns/multi-gaussian",
            "label": "Gaussian",
            "grid_plot_linestyle": "solid",
        },
        "dns_isoline": {
            "path": "dns/multi-isoline",
            "label": "Isoline-DD",
            "grid_plot_linestyle": "solid",
        },
    }
    
    # Load final values
    final_values = load_final_metrics(
        root_dir=ROOT_DIR,
        algorithm_dicts=emitter_dicts,
        metrics_list=metrics_list,
        seeds=SEEDS,
    )
    
    # Plot Pareto front: Average Fitness vs Diversity
    output_path = os.path.join(ANALYSIS_DIR, "pareto_front_fitness_diversity_emitters.png")
    plot_pareto_front(
        output_path=output_path,
        algorithm_dicts=emitter_dicts,
        final_values=final_values,
        metric_x="avg_pairwise_dist",
        metric_y="total_fitnesses",
        metric_x_label="Diversity (Avg. Pairwise Distance)",
        metric_y_label="Total Fitness",
        maximize_x=True,
        maximize_y=True,
        title="Pareto Front: Total Fitness vs Diversity"
    )