import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import List, Dict, Any, Optional

# ---- Visual Hyperparameters (Keeping your style) ----
FIG_WIDTH = 24 
FIG_HEIGHT = 6
FIGURE_DPI = 200

TITLE_FONT_SIZE = 20
TITLE_FONT_WEIGHT = 'bold' 
LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 16
LEGEND_FONT_WEIGHT = 'bold'

NROWS = 1
NCOLS = 3 

def customize_axis(ax: Any) -> Any:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1.0)
    return ax

def get_experiment_subdir(base_path: str) -> Optional[str]:
    if not os.path.exists(base_path): return None
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs: return None
    subdirs.sort() 
    return os.path.join(base_path, subdirs[-1])

def load_time_series_data(root_dir: str, algorithm_dicts: Dict, seeds: List[int]):
    metrics_to_load = {
        "qd_score": "Total Fitness",
        "avg_pairwise_dist": "Descriptor Distance",
        "coverage": "Num. Solutions"
    }
    
    raw_data = {m: {algo: [] for algo in algorithm_dicts} for m in metrics_to_load}

    for algo_key, config in algorithm_dicts.items():
        found_any = False
        for seed in seeds:
            path = os.path.join(root_dir, str(seed), config["path"])
            exp_path = get_experiment_subdir(path)
            if not exp_path: continue
            
            csv_path = os.path.join(exp_path, "metrics", "mapelites-logs.csv")
            if not os.path.exists(csv_path): continue
            
            df = pd.read_csv(csv_path)
            if df.empty or "emitter_time" not in df.columns: continue

            time_x = df["emitter_time"].values
            found_any = True
            
            for m_csv in metrics_to_load:
                if m_csv in df.columns:
                    values = df[m_csv].values
                    if m_csv == "coverage":
                        values = values * 64 
                    raw_data[m_csv][algo_key].append((time_x, values))
        
        if not found_any:
            print(f"  [Warning] No valid data found for: {algo_key}")
                
    return raw_data

def plot_emitter_time_grid(output_path: str, algorithm_dicts: Dict, raw_data: Dict):
    metrics_list = ["qd_score", "avg_pairwise_dist", "coverage"]
    metrics_labels = {
        "qd_score": "Total Fitnesses",
        "avg_pairwise_dist": "Average Descriptor Distance",
        "coverage": "Number of Solutions"
    }

    # Setup Colors
    algo_keys = list(algorithm_dicts.keys())
    custom_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f"]
    color_map = {key: custom_colors[i % len(custom_colors)] for i, key in enumerate(algo_keys)}

    fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(FIG_WIDTH, FIG_HEIGHT))
    axes_flat = axes.ravel()

    has_plotted_anything = False

    for idx, metric in enumerate(metrics_list):
        ax = axes_flat[idx]
        
        for algo_key, algo_config in algorithm_dicts.items():
            seeds_data = raw_data[metric][algo_key]
            if not seeds_data: continue
            
            has_plotted_anything = True

            all_times = np.concatenate([d[0] for d in seeds_data])
            common_x = np.linspace(all_times.min(), all_times.max(), 200)
            
            interpolated_values = []
            for tx, ty in seeds_data:
                interpolated_values.append(np.interp(common_x, tx, ty))
            
            interpolated_values = np.array(interpolated_values)
            median_y = np.median(interpolated_values, axis=0)
            lq_y = np.percentile(interpolated_values, 25, axis=0)
            uq_y = np.percentile(interpolated_values, 75, axis=0)

            ax.plot(common_x, median_y, 
                    label=algo_config["label"], 
                    color=color_map[algo_key],
                    linewidth=2.5)
            
            ax.fill_between(common_x, lq_y, uq_y, 
                            color=color_map[algo_key], alpha=0.15, linewidth=0)

        ax.set_title(metrics_labels[metric], fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
        ax.set_xlabel("Emitter Time (seconds)", fontsize=LABEL_FONT_SIZE)
        customize_axis(ax)

    # CRITICAL FIX: Ensure handles exist before calling legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    
    if labels:
        # Avoid division by zero in ncol if only one label exists
        ncol = max(1, len(labels))
        fig.legend(handles, labels, loc='lower center', ncol=ncol, 
                   prop={'weight': LEGEND_FONT_WEIGHT, 'size': LEGEND_FONT_SIZE}, frameon=False)
    else:
        print("  [Error] Cannot create plot: No data was successfully loaded.")
        plt.close()
        return

    plt.subplots_adjust(bottom=0.25, top=0.85, wspace=0.2)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=FIGURE_DPI)
    print(f"Time-based plot saved to: {output_path}")

if __name__ == "__main__":
    # --- UPDATED CONFIG ---
    ROOT_DIR = "/workspace/src/hpc_results/output/sdxl-turbo/"
    SEEDS = [1, 4, 7, 10, 13, 16, 20, 22, 25, 28, 31, 34]
    
    # Use your specific algorithm dict here
    ALGORITHM_DICTS = {
        "dns_svd": {"path": "dns/multi-hidden-layer-multi-svd-alpha-10", "label": "Spectral-Guided QD-Diffusion (Ours)"},
        "adam": {"path": "dns/multi-adam", "label": "Adam"},
        "cma-es": {"path": "dns/multi-sep-cma-es", "label": "Sep-CMA-ES"},
    }

    data = load_time_series_data(ROOT_DIR, ALGORITHM_DICTS, SEEDS)
    plot_emitter_time_grid("plots/wall_clock_performance.png", ALGORITHM_DICTS, data)