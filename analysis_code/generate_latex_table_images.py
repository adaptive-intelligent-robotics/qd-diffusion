import os
import glob
import shutil
from typing import Dict, List
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


def generate_latex_table_images(
    root_dir: str,
    algorithm_dicts: Dict[str, Dict],
    seeds: List[int],
    metric: str = "fitness",
    output_dir: str = "analysis_output/comparison_tables",
    prompts_dict: Dict[int, str] = None,
):
    """
    Generate comparison table figures with seed image on left and algorithm grids on right.
    
    Args:
        root_dir: Root directory containing seed folders
        algorithm_dicts: Dictionary of dictionaries with 'path' and 'label' keys
        seeds: List of seeds to process
        metric: Metric to use for the grid (e.g., 'fitness', 'dns', 'mixed')
        output_dir: Directory to save comparison table figures
        prompts_dict: Dictionary mapping seed numbers to prompts for captions
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of algorithm keys in order
    algo_keys = list(algorithm_dicts.keys())
    
    # Process each seed
    for seed in seeds:
        print(f"Processing seed {seed}...")
        
        # Get prompt for title
        if prompts_dict and seed in prompts_dict:
            prompt = prompts_dict[seed]
        else:
            prompt = f"Seed {seed}"
        
        # Find initial seed image
        first_algo_key = algo_keys[0]
        first_algo_path = algorithm_dicts[first_algo_key]["path"]
        
        initial_image_pattern = os.path.join(
            root_dir,
            str(seed),
            first_algo_path,
            "*",
            "initial_values",
            "initial_image.png"
        )
        
        initial_image_files = glob.glob(initial_image_pattern)
        
        if not initial_image_files:
            print(f"  Warning: No initial image found for seed {seed}, skipping...")
            continue
        
        initial_image_path = initial_image_files[0]
        
        # Collect algorithm grid images
        algo_images = []
        algo_labels = []
        
        for algo_key in algo_keys:
            algo_info = algorithm_dicts[algo_key]
            algo_path = algo_info["path"]
            label = algo_info["label"]
            
            # Find the grid image
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
                algo_images.append(matching_files[0])
                algo_labels.append(label)
            else:
                print(f"  Warning: No grid found for {algo_key} (seed {seed})")
        
        if not algo_images:
            print(f"  No algorithm images found for seed {seed}, skipping...")
            continue
        
        # Create the comparison figure
        # Layout proportions from LaTeX: seed=0.18, gap=0.04, algorithms=0.78 (split into 3)
        # Width ratios: [0.18, 0.04 (gap), 0.26, 0.26, 0.26]
        fig = plt.figure(figsize=(28, 10))
        gs = gridspec.GridSpec(2, 5, figure=fig, 
                              width_ratios=[0.18, 0.04, 0.26, 0.26, 0.26],
                              wspace=0.0, hspace=0.15, 
                              top=0.98, bottom=0.02, left=0.01, right=0.99)
        
        # Load and display seed image with prompt (spans 2 rows)
        ax_seed = fig.add_subplot(gs[:, 0])
        seed_img = Image.open(initial_image_path)
        ax_seed.imshow(seed_img)
        
        # Wrap prompt text if too long
        import textwrap
        # Prompt on top, "Seed Image" centered on bottom
        wrapped_prompt = "\n".join(textwrap.wrap(prompt, width=18))
        ax_seed.set_title(wrapped_prompt, fontsize=28, fontweight='bold', pad=10, loc='left')
        ax_seed.text(0.5, -0.01, "Seed Image", transform=ax_seed.transAxes,
                    ha='center', va='top', fontsize=24)
        ax_seed.axis('off')
        
        # Gap column (column 1) - leave empty
        
        # Display algorithm grids in 2x3 layout (columns 2, 3, 4)
        for idx, (img_path, label) in enumerate(zip(algo_images[:6], algo_labels[:6])):
            row = idx // 3
            col = idx % 3 + 2  # +2 because columns 0 is seed, 1 is gap
            
            ax = fig.add_subplot(gs[row, col])
            algo_img = Image.open(img_path)
            ax.imshow(algo_img)
            ax.axis('off')
            
            # Place label below the image
            wrapped_label = "\n".join(textwrap.wrap(label, width=25))
            ax.text(0.5, -0.02, wrapped_label, transform=ax.transAxes,
                   ha='center', va='top', fontsize=24, fontweight='bold')
        
        # Adjust layout to minimize white space
        plt.subplots_adjust(wspace=0.0, hspace=0.15)
        
        # Save the figure
        output_filename = f"comparison_table_seed_{seed}_{metric}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {output_path}")
    
    print(f"\n✓ Comparison tables saved to: {output_dir}")


def main():
    """
    Example usage when running the script directly.
    """
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table images and code"
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
        default="fitness",
        choices=["fitness", "dns", "mixed"],
        help="Metric to use"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs='+',
        default=[1, 2, 3, 4],
        help="List of seeds to process (space-separated)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_output/comparison_tables",
        help="Output directory for comparison table figures"
    )
    
    args = parser.parse_args()
    
    # Import prompts
    try:
        from prompts import prompts_dict
    except ImportError:
        print("Warning: Could not import prompts_dict from prompts.py")
        prompts_dict = None
    
    # Example algorithm dictionary - modify this to match your experiments
    algorithm_dicts = {
        "dns_svd": {
            "path": "dns/multi-hidden-layer-multi-svd-alpha-10",
            "label": "Spectral-Guided QD-Diffusion (Ours)",
        },  
        "cvt-map-elites": {
            "path": "cvt-map-elites/multi-hidden-layer-multi-svd-alpha-10",
            "label": "CVT MAP-Elites",
        },
        "ablation": {   
            "path": "dns/multi-hidden-layer-multi-svd-threshold-0-alpha-10",
            "label": "Fitness Threshold Ablation",
        },
        "cma-es": {
            "path": "dns/multi-sep-cma-es",
            "label": "Sep-CMA-ES",
        },
        "adam": {
            "path": "dns/multi-adam",
            "label": "Adam",
        },
        "qdhf": {
            "path": "qdhf/multi-hidden-layer-multi-svd-alpha-10",
            "label": "QDHF",
        },
    }
    
    generate_latex_table_images(
        root_dir=args.root_dir,
        algorithm_dicts=algorithm_dicts,
        seeds=args.seeds,
        metric=args.metric,
        output_dir=args.output_dir,
        prompts_dict=prompts_dict,
    )


if __name__ == "__main__":
    main()
