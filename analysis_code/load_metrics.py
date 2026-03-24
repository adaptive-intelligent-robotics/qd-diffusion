import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple

def get_experiment_subdir(base_path: str) -> str:
    """
    Finds the inner timestamped directory inside the algorithm folder.
    E.g. turns '.../seed/algo' into '.../seed/algo/2026-01-01_12345'
    """
    if not os.path.exists(base_path):
        return None
        
    # List all subdirectories in the base path
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # If there are no subdirectories, we can't proceed
    if not subdirs:
        return None
    
    # If there's more than one, we usually pick the one that actually looks like a Hydra output 
    # or just the first one if you expect only one run per seed folder.
    # Here we pick the first one found, or you could sort by name/date if needed.
    # Sorting ensures determinism.
    subdirs.sort() 
    return os.path.join(base_path, subdirs[-1]) # Takes the last one (alphabetically/latest usually)

def load_metrics(
    root_dir: str,
    algorithm_dicts: Dict,
    metrics_list: List[str],
    seeds: range,
) -> Tuple[Dict, Dict, Dict]:
    """
    Loads metrics for each algorithm.
    Assumes all seeds for a SINGLE algorithm share the exact same 'evaluations' logging steps.
    """
    
    # Structure: {metric: {algo_label: DataFrame}}
    all_medians = {m: {} for m in metrics_list}
    all_lqs = {m: {} for m in metrics_list}
    all_uqs = {m: {} for m in metrics_list}

    for algo_key, algo_config in algorithm_dicts.items():
        print(f"\nProcessing Algorithm: {algo_config['label']} ({algo_key})")
        
        algo_path_name = algo_config["path"]
        
        # We need to capture the 'evaluations' column from the first valid seed
        ref_evaluations = None
        
        # Store raw values: {metric: [ [val_seed1], [val_seed2], ... ] }
        algo_raw_values = {m: [] for m in metrics_list}
        
        valid_seeds_count = 0

        for seed in seeds:
            # 1. Construct the path to the Algorithm/Emitter folder
            initial_path = os.path.join(root_dir, str(seed), algo_path_name)
            
            # 2. Find the timestamped subdirectory (e.g., 2026-01-01_...)
            experiment_path = get_experiment_subdir(initial_path)
            
            if experiment_path is None:
                # print(f"  !! Missing experiment folder for seed {seed} in {initial_path}")
                continue

            # 3. Construct paths to actual files
            metrics_path = os.path.join(experiment_path, "metrics", "mapelites-logs.csv")
            check_file_path = os.path.join(experiment_path, "repertoire", "final_fitnesses.pt")

            # Check for successful run file
            if not os.path.exists(check_file_path):
                print(f"  !! WARNING: Seed {seed} missing 'final_fitnesses.pt' - Excluding.")
                print(f"     Checked: {check_file_path}")
                continue
            
            # Check if there are multiple timestamped subdirectories
            subdirs = [d for d in os.listdir(initial_path) if os.path.isdir(os.path.join(initial_path, d))]
            if len(subdirs) > 1:
                print(f"  !! WARNING: Seed {seed} has multiple experiment subdirectories ({len(subdirs)} found). Using: {os.path.basename(experiment_path)}")
            
            if not os.path.exists(metrics_path):
                print(f"  !! WARNING: Seed {seed} missing metrics file - Excluding.")
                continue

            try:
                # Load Data
                df = pd.read_csv(metrics_path)
                
                # Check evaluations alignment
                current_evals = df['evaluations'].values
                
                if ref_evaluations is None:
                    ref_evaluations = current_evals
                else:
                    # Sanity check: ensure this seed has same log frequency as the first seed
                    if not np.array_equal(ref_evaluations, current_evals):
                        print(f"  !! Mismatch in log length/frequency for Seed {seed}. Expected {len(ref_evaluations)}, got {len(current_evals)}. Skipping.")
                        continue

                # Store Metric Data
                for metric in metrics_list:
                    if metric in df.columns:
                        algo_raw_values[metric].append(df[metric].values)
                    elif metric == "num_solutions":
                        if 'coverage' in df.columns:
                            algo_raw_values[metric].append(df['coverage'].values * 64)
                    elif metric == "total_fitness":
                        if 'coverage' in df.columns and 'avg_fitness' in df.columns:
                            algo_raw_values[metric].append(df['coverage'].values * 64 * df['avg_fitness'].values)
                    else:
                        print(f"    !! Metric {metric} missing in CSV for seed {seed}")

                valid_seeds_count += 1

            except Exception as e:
                print(f"  !! Error processing seed {seed}: {e}")

        print(f"  -> Processed {valid_seeds_count} valid seeds.")

        # Compute Stats for this Algorithm
        if valid_seeds_count > 0 and ref_evaluations is not None:
            for metric in metrics_list:
                data_list = algo_raw_values[metric]
                
                if not data_list:
                    continue

                # Stack: rows=seeds, cols=evaluations
                stacked_data = np.vstack(data_list)
                
                median = np.nanmedian(stacked_data, axis=0)
                lq = np.nanpercentile(stacked_data, 25, axis=0)
                uq = np.nanpercentile(stacked_data, 75, axis=0)
                
                # Store results
                df_stats = pd.DataFrame({
                    'evaluations': ref_evaluations,
                    'median': median,
                    'lq': lq,
                    'uq': uq
                })
                
                # Use evaluations as index for easier plotting later
                df_stats.set_index('evaluations', inplace=True)
                
                all_medians[metric][algo_key] = df_stats['median']
                all_lqs[metric][algo_key] = df_stats['lq']
                all_uqs[metric][algo_key] = df_stats['uq']

    return all_medians, all_lqs, all_uqs