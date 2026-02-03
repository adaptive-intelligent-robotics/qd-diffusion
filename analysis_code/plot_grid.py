import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import List, Dict, Any

# ---- Visual Hyperparameters ----
FIG_WIDTH = 30
FIG_HEIGHT = 5
FIGURE_DPI = 200

# Font sizes
TITLE_FONT_SIZE = 20
TITLE_FONT_WEIGHT = 'bold' 

LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 20
LEGEND_FONT_WEIGHT = 'bold'
LEGEND_TWO_LINES = False  # Set to True for two lines, False for one line

# Spacing
LEFTSPACING = 0.01 # the left side of the subplots of the figure
RIGHTSPACING = 0.999 # the right side of the subplots of the figure
BOTTOMSPACING = 0.22  # the bottom of the subplots of the figure
TOPSPACING = 0.9 # the top of the subplots of the figure
WIDTHSPACING = 0.1 # the proportion of width reserved for blank space between subplots
HEIGHTSPACING = 0.3  # the proportion of height reserved for blank space between subplots

NROWS = 1
NCOLS = 5


# FIG_WIDTH = 20
# FIG_HEIGHT = 6
# FIGURE_DPI = 200

# # Font sizes
# TITLE_FONT_SIZE = 20
# TITLE_FONT_WEIGHT = 'bold' 
# LABEL_FONT_SIZE = 12
# LEGEND_FONT_SIZE = 20
# LEGEND_FONT_WEIGHT = 'bold'

# # Legend layout
# LEGEND_TWO_LINES = True  # Set to True for two lines, False for one line

# # Spacing
# LEFTSPACING = 0.00 # the left side of the subplots of the figure
# RIGHTSPACING = 1.0 # the right side of the subplots of the figure
# BOTTOMSPACING = 0.15  # the bottom of the subplots of the figure
# TOPSPACING = 0.9 # the top of the subplots of the figure
# WIDTHSPACING = 0.1 # the proportion of width reserved for blank space between subplots
# HEIGHTSPACING = 0.3  # the proportion of height reserved for blank space between subplots

# NROWS = 2
# NCOLS = 3

def customize_axis(ax: Any) -> Any:
    """Removes spines and adds grid for cleaner look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)

    ax.get_xaxis().tick_bottom()
    ax.tick_params(axis="y", length=0)
    
    # Grid behind data
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1.0)
    return ax

def plot_grid(
    output_path: str,
    algorithm_dicts: Dict,
    metrics_list: List[str],
    metrics_labels: Dict[str, str],
    medians: Dict,
    lqs: Dict,
    uqs: Dict,
) -> None:
    
    print("\n-------------------------------------------------------------------------")
    print("                           Generating Grid Plot                          ")
    print("-------------------------------------------------------------------------")

    num_plots = len(metrics_list)
    nrows, ncols = NROWS, NCOLS
    
    # Setup Colors using Matplotlib 'tab20'
    algo_keys = list(algorithm_dicts.keys())
    
    # 'tab20' has 20 distinct colors. If you have >20 algos, this will cycle.
    # cmap = plt.get_cmap('Set3') 
    
    # # Create a mapping of algorithm key -> color
    # # We use modulo just in case you ever have more than 20 algorithms
    # color_map = {key: cmap(i % 20) for i, key in enumerate(algo_keys)}
    
    algo_keys = list(algorithm_dicts.keys())
    
    # A list of 15 distinct, high-contrast hex colors
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

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(FIG_WIDTH, FIG_HEIGHT * nrows))
    
    if num_plots == 1: axes = [axes]
    axes_flat = np.array(axes).ravel()

    for idx, metric in enumerate(metrics_list):
        ax = axes_flat[idx]
        
        for algo_key, algo_config in algorithm_dicts.items():
            if algo_key not in medians[metric]: continue # Skip if no data
            
            # Retrieve data
            data_series = medians[metric][algo_key]
            
            x = data_series.index.values # The evaluations
            med = data_series.values
            lq = lqs[metric][algo_key].values
            uq = uqs[metric][algo_key].values
            
            # Plot Median
            ax.plot(x, med, 
                   label=algo_config["label"],
                   color=color_map[algo_key],
                   linestyle=algo_config["grid_plot_linestyle"],
                   linewidth=2)
            
            # Plot IQR Shading
            ax.fill_between(x, lq, uq, color=color_map[algo_key], alpha=0.2, linewidth=0)

        # Labels and Styling
        ax.set_title(metrics_labels.get(metric, metric), fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
        ax.set_xlabel("Evaluations", fontsize=LABEL_FONT_SIZE)
        customize_axis(ax)

    # Hide empty subplots
    for i in range(num_plots, len(axes_flat)):
        axes_flat[i].axis('off')

    # Legend
    handles = []
    labels = []
    for algo_key, algo_config in algorithm_dicts.items():
        line = plt.Line2D([0], [0], color=color_map[algo_key], lw=2, linestyle=algo_config["grid_plot_linestyle"])
        handles.append(line)
        labels.append(algo_config["label"])

    # Determine number of columns based on LEGEND_TWO_LINES setting
    legend_ncol = (len(labels) + 1) // 2 if LEGEND_TWO_LINES else len(labels)
    fig.legend(handles, labels, loc='lower center', ncol=legend_ncol, prop={'weight': LEGEND_FONT_WEIGHT, 'size': LEGEND_FONT_SIZE}, frameon=False)

    plt.subplots_adjust(
        left=LEFTSPACING, right=RIGHTSPACING, 
        bottom=BOTTOMSPACING, top=TOPSPACING, 
        wspace=WIDTHSPACING, hspace=HEIGHTSPACING
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=FIGURE_DPI)
    print(f"Plot saved to: {output_path}")
    plt.close()