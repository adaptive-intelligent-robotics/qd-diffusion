import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from typing import List, Dict

import config

def get_experiment_subdir(base_path: str) -> str:
    """Finds the timestamped folder inside the algo path."""
    if not os.path.exists(base_path): return None
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs: return None
    subdirs.sort() 
    return os.path.join(base_path, subdirs[-1])

def load_final_metrics(root_dir, algo_dicts, metrics, seeds):
    """
    Returns a dictionary: { metric: { algo_key: [val_seed1, val_seed2...] } }
    """
    final_data = {m: {k: [] for k in algo_dicts} for m in metrics}

    print("Loading final metrics for statistical analysis...")

    for algo_key, algo_config in algo_dicts.items():
        print(f"\nProcessing Algorithm: {algo_config['label']} ({algo_key})")
        valid_count = 0
        for seed in seeds:
            initial_path = os.path.join(root_dir, str(seed), algo_config["path"])
            exp_path = get_experiment_subdir(initial_path)
            
            if exp_path is None: continue

            metrics_file = os.path.join(exp_path, "metrics", "mapelites-logs.csv")
            
            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)
                    if not df.empty:
                        last_row = df.iloc[-1]
                        for metric in metrics:
                            if metric == 'num_solutions':
                                # Calculate num_solutions as coverage * 64
                                if 'coverage' in last_row:
                                    val = last_row['coverage'] * 64
                                    if not np.isnan(val):
                                        final_data[metric][algo_key].append(val)
                            elif metric in last_row:
                                val = last_row[metric]
                                if not np.isnan(val):
                                    final_data[metric][algo_key].append(val)
                        valid_count += 1
                except Exception as e:
                    print(f"     Warning: Failed to load seed {seed}: {e}")
                    pass
        print(f"  -> Processed {valid_count} valid seeds.")
        for metric in metrics:
            count = len(final_data[metric][algo_key])
            print(f"     Metric '{metric}': {count} values")
    return final_data

def holm_bonf_correction(p_values):
    """
    Manual implementation of Holm-Bonferroni correction.
    """
    p_values = np.array(p_values)
    
    # Handle NaNs (if any)
    non_nan_mask = ~np.isnan(p_values)
    non_nan_p_values = p_values[non_nan_mask]

    # Order non-nan p values
    ordered_p_indices = np.argsort(non_nan_p_values)
    n_hypotheses = len(non_nan_p_values) 
    
    corrected_p_values_non_nan = np.zeros_like(non_nan_p_values)

    for i in range(len(ordered_p_indices)):
        original_idx = ordered_p_indices[i]
        p_val = non_nan_p_values[original_idx]
        
        # Correct p value: p * (m - rank + 1)
        corrected_p = p_val * (n_hypotheses - i)
        
        # Cap at 1.0
        corrected_p = min(corrected_p, 1.0)
        
        # Enforce monotonicity (step-down procedure)
        if i > 0:
            prev_idx = ordered_p_indices[i-1]
            corrected_p = max(corrected_p, corrected_p_values_non_nan[prev_idx])
            
        corrected_p_values_non_nan[original_idx] = corrected_p
    
    # Reconstruct full array with NaNs
    final_corrected = np.empty_like(p_values)
    final_corrected[non_nan_mask] = corrected_p_values_non_nan
    final_corrected[~non_nan_mask] = np.nan

    return final_corrected

def perform_statistical_analysis(
    root_dir: str,
    seeds: List[int],
    algorithm_dicts: Dict,
    metrics_list: List[str],
    main_algo_key: str,
    output_dir: str
):
    """
    Main function to run the full analysis and save results.
    """
    print("\n-------------------------------------------------------------------------")
    print("                       Running Statistical Analysis                      ")
    print("-------------------------------------------------------------------------")

    if main_algo_key not in algorithm_dicts:
        print(f"!! Error: Main algorithm key '{main_algo_key}' not found.")
        return

    # 1. Load Data
    final_data = load_final_metrics(root_dir, algorithm_dicts, metrics_list, seeds)

    # 2. Check Data Availability
    if not final_data[metrics_list[0]].get(main_algo_key):
        print(f"!! Error: No data found for main algorithm '{main_algo_key}'.")
        return

    results_table = []

    # 3. Iterate per METRIC to apply correction per family of hypotheses
    for metric in metrics_list:
        p_values = []
        comparisons = [] # List of algo_keys corresponding to p_values
        
        # Collect raw p-values
        for algo_key in algorithm_dicts:
            if algo_key == main_algo_key: continue

            main_vals = np.array(final_data[metric][main_algo_key])
            comp_vals = np.array(final_data[metric].get(algo_key, []))
            
            # Ensure we have matching lengths for paired Wilcoxon
            # We truncate to the minimum length (assuming seeds align across algorithms)
            min_len = min(len(main_vals), len(comp_vals))
            
            if min_len >= 2:
                try:
                    # Paired Wilcoxon signed-rank test
                    stat, p = wilcoxon(main_vals[:min_len], comp_vals[:min_len])
                    p_values.append(p)
                    comparisons.append(algo_key)
                except ValueError:
                    # Occurs if all differences are zero
                    p_values.append(1.0)
                    comparisons.append(algo_key)
            else:
                pass # Not enough data

        if not p_values:
            continue

        # Apply Holm-Bonferroni correction
        corrected_pvals = holm_bonf_correction(p_values)

        # Store results
        for i, algo_key in enumerate(comparisons):
            # Find or create row
            row = next((r for r in results_table if r["Algorithm"] == algorithm_dicts[algo_key]["label"]), None)
            if row is None:
                row = {"Algorithm": algorithm_dicts[algo_key]["label"]}
                results_table.append(row)
            
            corr_p = corrected_pvals[i]
            
            # Formatting
            sig_mark = "*" if corr_p < 0.05 else ""
            if corr_p < 0.001:
                row[metric] = f"{corr_p:.2e}{sig_mark}"
            else:
                row[metric] = f"{corr_p:.4f}{sig_mark}"

    # 4. Save and Print
    if results_table:
        df_results = pd.DataFrame(results_table)
        
        # Reorder columns
        cols = ["Algorithm"] + [col for col in df_results.columns if col != "Algorithm"]
        df_results = df_results[cols]

        # Display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(f"\nBaseline: {algorithm_dicts[main_algo_key]['label']}")
        print(df_results.to_string(index=False))
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "stats_holm_correction.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\nStats saved to: {csv_path}")
    else:
        print("No statistical results generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wilcoxon tests with Holm-Bonferroni correction on final metrics.")
    config.add_model_argument(parser)
    args = parser.parse_args()

    cfg = config.get_model_config(args.model)
    perform_statistical_analysis(
        root_dir=cfg["root_dir"],
        seeds=cfg["seeds"],
        algorithm_dicts=cfg["algorithms"],
        metrics_list=["avg_pairwise_dist", "total_fitnesses", "num_solutions"],
        main_algo_key=config.BASE_ALGO_KEY,
        output_dir=cfg["output_dir"],
    )
