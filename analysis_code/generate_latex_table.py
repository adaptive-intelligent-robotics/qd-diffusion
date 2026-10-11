import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from typing import List, Dict, Optional

import config

def get_experiment_subdir(base_path: str) -> Optional[str]:
    """
    Finds the inner timestamped directory inside the algorithm folder.
    """
    if not os.path.exists(base_path):
        return None
        
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    if not subdirs:
        return None
    
    subdirs.sort() 
    return os.path.join(base_path, subdirs[-1])


def load_all_seed_metrics(
    root_dir: str,
    algorithm_dicts: Dict,
    metrics_list: List[str],
    seeds: List[int],
    target_evaluations: Optional[int] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Loads the metric values for each algorithm at a given evaluation count
    (or the final row if target_evaluations is None), returning all seed values
    (not averaged).
    
    Returns:
        Dict mapping algorithm key -> {metric: [value_seed1, value_seed2, ...]}
    """
    
    seed_values = {}
    
    print("\n-------------------------------------------------------------------------")
    if target_evaluations is None:
        print("                    Loading Metrics for LaTeX Table                     ")
    else:
        print(f"           Loading Metrics at evaluations={target_evaluations}           ")
    print("-------------------------------------------------------------------------")
    
    for algo_key, algo_config in algorithm_dicts.items():
        print(f"\nProcessing Algorithm: {algo_config['label']} ({algo_key})")
        algo_seed_values = {m: [] for m in metrics_list}
        
        valid_seeds_count = 0

        for seed in seeds:
            initial_path = os.path.join(root_dir, str(seed), algo_config["path"])
            exp_path = get_experiment_subdir(initial_path)
            
            if exp_path is None:
                continue

            metrics_file = os.path.join(exp_path, "metrics", "mapelites-logs.csv")
            
            if not os.path.exists(metrics_file):
                continue
                
            try:
                df = pd.read_csv(metrics_file)
                
                if df.empty:
                    continue
                
                if target_evaluations is None:
                    selected_row = df.iloc[-1]
                else:
                    if "evaluations" not in df.columns:
                        print("     Warning: 'evaluations' column missing; skipping this run")
                        continue
                    matches = df[df["evaluations"] == target_evaluations]
                    if matches.empty:
                        print(f"     Warning: No row with evaluations={target_evaluations}; skipping this run")
                        continue
                    selected_row = matches.iloc[-1]
                
                for metric in metrics_list:
                    if metric == 'num_solutions':
                        # Calculate num_solutions as coverage * 64
                        if 'coverage' in selected_row:
                            val = selected_row['coverage'] * 64
                            if not np.isnan(val):
                                algo_seed_values[metric].append(val)
                    elif metric == 'total_fitnesses':
                        # Calculate total_fitnesses as avg_fitness * coverage * 64
                        if 'avg_fitness' in selected_row and 'coverage' in selected_row:
                            val = selected_row['avg_fitness'] * selected_row['coverage'] * 64
                            if not np.isnan(val):
                                algo_seed_values[metric].append(val)
                    elif metric in selected_row:
                        val = selected_row[metric]
                        if not np.isnan(val):
                            algo_seed_values[metric].append(val)
                
                valid_seeds_count += 1
                
            except Exception as e:
                print(f"     Warning: Failed to load seed {seed}: {e}")
                pass

        print(f"  -> Processed {valid_seeds_count} valid seeds.")
        for metric in metrics_list:
            count = len(algo_seed_values[metric])
            print(f"     Metric '{metric}': {count} values")
        
        seed_values[algo_key] = algo_seed_values

    return seed_values


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


def calculate_statistical_significance(
    seed_values: Dict[str, Dict[str, List[float]]],
    metrics_list: List[str],
    maximize_metrics: Optional[Dict[str, bool]] = None,
    base_algo_key: Optional[str] = None,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    """
    Calculate p-values comparing each algorithm to a reference algorithm.

    If base_algo_key is given, all algorithms are compared against that fixed
    baseline (Wilcoxon signed-rank + Holm-Bonferroni correction).  Otherwise,
    the best-performing algorithm for each metric is used as the reference.

    Args:
        seed_values: Dict mapping algo_key -> {metric: [values]}
        metrics_list: List of metrics to analyze
        maximize_metrics: Dict indicating whether each metric should be maximized
        base_algo_key: Key of the algorithm to use as the fixed baseline.
                       When None, the best performer per metric is used.

    Returns:
        Tuple of:
        - Dict mapping metric -> {algo_key: p_value}
        - Dict mapping metric -> reference_algo_key
    """
    if maximize_metrics is None:
        maximize_metrics = {m: True for m in metrics_list}

    p_values = {m: {} for m in metrics_list}
    best_algos = {}

    print("\n-------------------------------------------------------------------------")
    if base_algo_key:
        print(f"        Calculating Statistical Significance vs '{base_algo_key}'         ")
    else:
        print("                    Calculating Statistical Significance                 ")
        print("                         (One-vs-Many Comparison)                        ")
    print("-------------------------------------------------------------------------")

    for metric in metrics_list:
        print(f"\nMetric: {metric}")

        if base_algo_key is not None:
            ref_key = base_algo_key
        else:
            means = {k: np.mean(v[metric]) for k, v in seed_values.items() if len(v[metric]) > 0}
            if not means:
                continue
            if maximize_metrics[metric]:
                ref_key = max(means.keys(), key=lambda k: means[k])
            else:
                ref_key = min(means.keys(), key=lambda k: means[k])

        best_algos[metric] = ref_key
        ref_vals = np.array(seed_values[ref_key][metric])
        ref_mean = np.mean(ref_vals) if len(ref_vals) > 0 else float("nan")
        print(f"  Reference algorithm: {ref_key} (mean = {ref_mean:.4f})")

        # The reference gets p=0.0 (it is the baseline by definition)
        p_values[metric][ref_key] = 0.0

        raw_p_values = []
        algo_keys_for_correction = []

        for algo_key in seed_values.keys():
            if algo_key == ref_key:
                continue

            algo_vals = np.array(seed_values[algo_key][metric])
            min_len = min(len(ref_vals), len(algo_vals))

            if min_len < 2:
                p_values[metric][algo_key] = np.nan
                continue

            try:
                _, p = wilcoxon(ref_vals[:min_len], algo_vals[:min_len])
                raw_p_values.append(p)
                algo_keys_for_correction.append(algo_key)
                print(f"    {algo_key}: raw p-value = {p:.4f}")
            except ValueError:
                raw_p_values.append(1.0)
                algo_keys_for_correction.append(algo_key)

        if raw_p_values:
            corrected = holm_bonf_correction(raw_p_values)
            for i, algo_key in enumerate(algo_keys_for_correction):
                p_values[metric][algo_key] = corrected[i]
                print(f"    {algo_key}: corrected p-value = {corrected[i]:.4f}")

    return p_values, best_algos


def save_p_values_csv(
    p_values: Dict[str, Dict[str, float]],
    algorithm_dicts: Dict,
    metrics_list: List[str],
    base_algo_key: str,
    output_path: str,
) -> None:
    """Save Holm-Bonferroni-corrected p-values to a CSV file.

    The reference algorithm is labelled "baseline".  A trailing '*' marks
    p < 0.05 for all other algorithms.
    """
    rows = []
    for algo_key, algo_config in algorithm_dicts.items():
        row: Dict = {"Algorithm": algo_config["label"]}
        for metric in metrics_list:
            if algo_key == base_algo_key:
                row[metric] = "baseline"
            else:
                p = p_values.get(metric, {}).get(algo_key, np.nan)
                if np.isnan(p):
                    row[metric] = "N/A"
                else:
                    sig = "*" if p < 0.05 else ""
                    row[metric] = f"{p:.2e}{sig}"
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nP-values saved to: {output_path}")
    print(df.to_string(index=False))


def generate_latex_table(
    seed_values: Dict[str, Dict[str, List[float]]],
    algorithm_dicts: Dict,
    metrics_list: List[str],
    metrics_labels: Dict[str, str],
    p_values: Dict[str, Dict[str, float]],
    best_algos: Dict[str, str],
    output_path: str,
    maximize_metrics: Optional[Dict[str, bool]] = None,
    significance_level: float = 0.05,
    show_std: bool = True
) -> None:
    """
    Generate a LaTeX table with means, standard deviations, and bold formatting.
    
    Bold formatting is applied to:
    - The best performing algorithm for each metric
    - Any other algorithms that are NOT significantly worse than the best (p >= significance_level)
    
    Args:
        seed_values: Dict mapping algo_key -> {metric: [values]}
        algorithm_dicts: Dict with algorithm configurations
        metrics_list: List of metrics to include
        metrics_labels: Dict mapping metric keys to display labels
        p_values: Dict mapping metric -> {algo_key: p_value}
        best_algos: Dict mapping metric -> best_algo_key
        output_path: Path to save the LaTeX table
        maximize_metrics: Dict indicating whether each metric should be maximized
        significance_level: P-value threshold for significance
        show_std: Whether to show standard deviation
    """
    if maximize_metrics is None:
        maximize_metrics = {m: True for m in metrics_list}
    
    print("\n-------------------------------------------------------------------------")
    print("                       Generating LaTeX Table                            ")
    print("-------------------------------------------------------------------------")
    
    # Calculate statistics
    stats = {}
    for algo_key in algorithm_dicts.keys():
        stats[algo_key] = {}
        for metric in metrics_list:
            vals = seed_values[algo_key][metric]
            if len(vals) > 0:
                stats[algo_key][metric] = {
                    'mean': np.mean(vals),
                    'std': np.std(vals, ddof=1) if len(vals) > 1 else 0,
                    'n': len(vals)
                }
            else:
                stats[algo_key][metric] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'n': 0
                }
    
    # Find best for each metric
    best_values = {}
    for metric in metrics_list:
        valid_means = {k: stats[k][metric]['mean'] 
                      for k in algorithm_dicts.keys() 
                      if not np.isnan(stats[k][metric]['mean'])}
        if valid_means:
            if maximize_metrics[metric]:
                best_values[metric] = max(valid_means.values())
            else:
                best_values[metric] = min(valid_means.values())
        else:
            best_values[metric] = np.nan
    
    # Build LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Performance comparison across algorithms. Bold values indicate the best performer or algorithms not significantly worse than the best (p $\\geq$ 0.05).}")
    latex_lines.append("\\label{tab:results}")
    
    # Table format: l for algorithm name, c for each metric
    col_format = "l" + "c" * len(metrics_list)
    latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex_lines.append("\\toprule")
    
    # Header row
    header = "Algorithm"
    for metric in metrics_list:
        label = metrics_labels.get(metric, metric)
        header += f" & {label}"
    header += " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # Data rows
    for algo_key, algo_config in algorithm_dicts.items():
        row = algo_config['label']
        
        for metric in metrics_list:
            mean_val = stats[algo_key][metric]['mean']
            std_val = stats[algo_key][metric]['std']
            
            if np.isnan(mean_val):
                row += " & ---"
                continue
            
            # Format the value
            if show_std:
                cell_text = f"{mean_val:.3f} $\\pm$ {std_val:.3f}"
            else:
                cell_text = f"{mean_val:.3f}"
            
            # Check if this is the best algorithm for this metric
            is_best = (algo_key == best_algos.get(metric))
            
            # Check if NOT significantly worse than the best
            # (i.e., p >= significance_level means no significant difference)
            not_significantly_worse = False
            if algo_key in p_values[metric]:
                p_val = p_values[metric][algo_key]
                # p_val == 0.0 is the best (assigned in calculation)
                # p_val >= significance_level means not significantly different
                if p_val == 0.0 or (not np.isnan(p_val) and p_val >= significance_level):
                    not_significantly_worse = True
            
            # Bold if best OR not significantly worse than best
            if is_best or not_significantly_worse:
                cell_text = f"\\textbf{{{cell_text}}}"
            
            row += f" & {cell_text}"
        
        row += " \\\\"
        latex_lines.append(row)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"\nLaTeX table saved to: {output_path}")
    print("\nTable preview:")
    print('\n'.join(latex_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the results LaTeX table and p-values.")
    config.add_model_argument(parser)
    args = parser.parse_args()

    cfg = config.get_model_config(args.model)
    algo_dicts = cfg["algorithms"]
    output_dir = cfg["output_dir"]

    metrics_labels = {
        "avg_fitness": "Avg. Fitness",
        "avg_pairwise_dist": "Avg. Pairwise Dist.",
        "total_variance": "Total Variance",
        "coverage": "Coverage",
        "num_solutions": "Num. Solutions",
        "total_fitnesses": "Total Fitness",
    }

    # Whether each metric is maximized (True) or minimized (False)
    maximize_metrics = {
        "avg_fitness": True,
        "avg_pairwise_dist": True,
        "coverage": True,
        "num_solutions": True,
        "total_fitnesses": True,
        "total_variance": True,
    }

    metrics_list = ["avg_pairwise_dist", "total_fitnesses", "num_solutions"]

    seed_values = load_all_seed_metrics(
        root_dir=cfg["root_dir"],
        algorithm_dicts=algo_dicts,
        metrics_list=metrics_list,
        seeds=cfg["seeds"],
    )
    p_values, best_algos = calculate_statistical_significance(
        seed_values=seed_values,
        metrics_list=metrics_list,
        maximize_metrics=maximize_metrics,
        base_algo_key=config.BASE_ALGO_KEY,
    )
    save_p_values_csv(
        p_values=p_values,
        algorithm_dicts=algo_dicts,
        metrics_list=metrics_list,
        base_algo_key=config.BASE_ALGO_KEY,
        output_path=os.path.join(output_dir, "p_values.csv"),
    )
    generate_latex_table(
        seed_values=seed_values,
        algorithm_dicts=algo_dicts,
        metrics_list=metrics_list,
        metrics_labels=metrics_labels,
        p_values=p_values,
        best_algos=best_algos,
        output_path=os.path.join(output_dir, "latex_results_table.txt"),
        maximize_metrics=maximize_metrics,
        significance_level=0.05,
        show_std=True,
    )
