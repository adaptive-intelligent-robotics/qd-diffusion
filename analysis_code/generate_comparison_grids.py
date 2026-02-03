import os
import glob
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict
import argparse


def generate_comparison_grids(
    root_dir: str,
    algorithm_dicts: Dict[str, Dict],
    seeds: range,
    metric: str = "fitness",
    output_dir: str = "analysis_output/comparisons",
):
    """
    Generate comparison plots showing top_k grids from multiple algorithms side by side.
    
    Args:
        root_dir: Root directory containing seed folders
        algorithm_dicts: Dictionary of dictionaries with 'path' and 'label' keys
        seeds: Range of seeds to process
        metric: Metric to use for the grid (e.g., 'fitness', 'dns', 'mixed')
        output_dir: Directory to save comparison plots
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each seed
    for seed in seeds:
        print(f"Processing seed {seed}...")
        
        # Collect images for this seed
        images_found = []
        labels_found = []
        algo_keys_found = []
        
        for algo_key, algo_info in algorithm_dicts.items():
            algo_path = algo_info["path"]
            label = algo_info["label"]
            
            # Construct path to the grid image
            # Pattern: {root_dir}/{seed}/{algo_path}/*/images/top_16_{metric}_grid.png
            search_pattern = os.path.join(
                root_dir,
                str(seed),
                algo_path,
                "*",
                "images",
                f"top_16_{metric}_grid.png"
            )
            
            matching_files = glob.glob(search_pattern)
            
            if matching_files:
                # Use the first match (should only be one per seed/algo combo)
                image_path = matching_files[0]
                images_found.append(image_path)
                labels_found.append(label)
                algo_keys_found.append(algo_key)
            else:
                print(f"  Warning: No grid found for {algo_key} (seed {seed})")
        
        # Skip if no images found for this seed
        if not images_found:
            print(f"  Skipping seed {seed} - no images found")
            continue
        
        # Create comparison plot
        n_algos = len(images_found)
        fig, axes = plt.subplots(1, n_algos, figsize=(5 * n_algos, 5))
        
        # Handle single subplot case
        if n_algos == 1:
            axes = [axes]
        
        for idx, (image_path, label) in enumerate(zip(images_found, labels_found)):
            # Read and display image
            img = plt.imread(image_path)
            axes[idx].imshow(img)
            axes[idx].set_title(label, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save comparison plot
        output_filename = f"comparison_seed_{seed}_{metric}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {output_path}")
    
    print(f"\nComparison grids saved to: {output_dir}")


def main():
    """
    Example usage when running the script directly.
    Modify the algorithm_dicts below to match your experiments.
    """
    parser = argparse.ArgumentParser(
        description="Generate comparison grids from top_k image grids"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/workspace/src/hpc_results/output/sdxl-turbo/",
        help="Root directory containing seed folders"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mixed",
        choices=["fitness", "dns", "mixed"],
        help="Metric to use for comparison"
    )
    parser.add_argument(
        "--seeds_start",
        type=int,
        default=1,
        help="Starting seed number"
    )
    parser.add_argument(
        "--seeds_end",
        type=int,
        default=37,
        help="Ending seed number (exclusive)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_output/comparisons",
        help="Output directory for comparison plots"
    )
    
    args = parser.parse_args()
    
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
    seeds = range(args.seeds_start, args.seeds_end)
    
    generate_comparison_grids(
        root_dir=args.root_dir,
        algorithm_dicts=baseline_dicts,
        seeds=seeds,
        metric=args.metric,
        output_dir=args.output_dir+"/baselines",
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
        
    seeds = range(args.seeds_start, args.seeds_end)
    
    generate_comparison_grids(
        root_dir=args.root_dir,
        algorithm_dicts=emitter_dicts,
        seeds=seeds,
        metric=args.metric,
        output_dir=args.output_dir+"/emitters",
    )


if __name__ == "__main__":
    main()
