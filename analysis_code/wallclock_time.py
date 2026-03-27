import os
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import wilcoxon
from typing import List, Dict, Optional

# --- CONFIG ---
# Path from your example
ROOT_DIR = "/workspace/src/hpc_results/timing_results/sdxl-turbo" 
REFERENCE_ALGO_KEY = "dns_svd"
SEEDS = [1, 4, 7, 10, 13, 16, 22, 25, 28, 31, 34] 

# Updated to match the actual folder names in your example path
ALGORITHM_DICTS = {
    "dns_svd": {"path": "dns/multi-hidden-layer-multi-svd", "label": "Spectral-Guided (Ours)"},
    "cvt-map-elites": {"path": "cvt-map-elites/multi-hidden-layer-multi-svd", "label": "CVT MAP-Elites"},
    "ablation": {"path": "dns/multi-hidden-layer-multi-svd-threshold-0", "label": "Threshold Ablation"},
    "cma-es": {"path": "dns/multi-sep-cma-es", "label": "Sep-CMA-ES"},
    "adam": {"path": "dns/multi-adam", "label": "Adam"},
    "qdhf": {"path": "qdhf/multi-hidden-layer-multi-svd", "label": "QDHF"},
    "dns_random": {"path": "dns/random-seeds", "label": "Random Seeds"},
    "dns_hidden_gaussian": {"path": "dns/multi-hidden-layer-gaussian", "label": "Intermed.-layer Gaussian"},
    "dns_mixed": {"path": "dns/mixed-all", "label": "ImageBreeder"},
    "dns_gaussian": {"path": "dns/multi-gaussian", "label": "Gaussian"},
    "dns_isoline": {"path": "dns/multi-isoline", "label": "Isoline-DD"},
}

METRICS_MAP = {
    "max_fitness": "Max Fitness",
    "qd_score": "QD Score",
    "avg_pairwise_dist": "Avg Descriptor Distance",
    "num_solutions": "Num. Solutions"
}

# --- THE CRAWLER ---

def crawl_for_experiment_files(base_path: str):
    """Recursively crawls to find the folder containing main.log and metrics/mapelites-logs.csv."""
    if not os.path.exists(base_path):
        return None, None

    for root, dirs, files in os.walk(base_path):
        log_path = os.path.join(root, "main.log")
        csv_path = os.path.join(root, "metrics", "mapelites-logs.csv")
        
        if os.path.exists(log_path) and os.path.exists(csv_path):
            return log_path, csv_path
            
    return None, None

def get_runtime_from_log(log_path: str) -> float:
    try:
        timestamps = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "[" in line and "]" in line:
                    try:
                        # Extracting '2026-02-18 14:51:59'
                        ts_str = line.split(']')[0][1:].split(',')[0]
                        timestamps.append(datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"))
                    except: continue
        if len(timestamps) < 2: return None
        return (timestamps[-1] - timestamps[0]).total_seconds()
    except: return None

def get_row_at_timestamp(df: pd.DataFrame, target_time: float) -> pd.Series:
    if 'emitter_time' not in df.columns or df.empty:
        return df.iloc[-1]
    df['emitter_time'] = pd.to_numeric(df['emitter_time'], errors='coerce')
    matched = df[df['emitter_time'] <= target_time]
    return matched.iloc[-1] if not matched.empty else df.iloc[0]

# --- MAIN RUNNER ---

def run_analysis():
    results_data = {algo: {m: [] for m in METRICS_MAP} for algo in ALGORITHM_DICTS if algo != REFERENCE_ALGO_KEY}
    ref_data = {algo: {m: [] for m in METRICS_MAP} for algo in ALGORITHM_DICTS if algo != REFERENCE_ALGO_KEY}
    matched_times = {algo: [] for algo in ALGORITHM_DICTS if algo != REFERENCE_ALGO_KEY}

    print(f"--- Starting Analysis in: {ROOT_DIR} ---")

    for seed in SEEDS:
        seed_str = str(seed)
        # Resolve Reference
        ref_path = os.path.join(ROOT_DIR, seed_str, ALGORITHM_DICTS[REFERENCE_ALGO_KEY]["path"])
        ref_log, ref_csv = crawl_for_experiment_files(ref_path)

        if not (ref_log and ref_csv):
            # Try once more without the alpha-10 suffix if path was wrong
            print(f" [!] Seed {seed_str}: Reference (dns_svd) not found in {ref_path}")
            continue

        ref_runtime = get_runtime_from_log(ref_log)
        df_ref = pd.read_csv(ref_csv)

        for algo_key, config in ALGORITHM_DICTS.items():
            if algo_key == REFERENCE_ALGO_KEY: continue
            
            comp_path = os.path.join(ROOT_DIR, seed_str, config["path"])
            comp_log, comp_csv = crawl_for_experiment_files(comp_path)

            if not (comp_log and comp_csv): 
                continue

            comp_runtime = get_runtime_from_log(comp_log)
            df_comp = pd.read_csv(comp_csv)

            common_time = min(ref_runtime, comp_runtime)
            matched_times[algo_key].append(common_time)

            row_ref = get_row_at_timestamp(df_ref, common_time)
            row_comp = get_row_at_timestamp(df_comp, common_time)

            mapping = {"max_fitness": "max_fitness", "qd_score": "qd_score", 
                       "avg_pairwise_dist": "avg_pairwise_dist", "num_solutions": "coverage"}

            for m_key, col in mapping.items():
                r_v, c_v = row_ref[col], row_comp[col]
                if m_key == "num_solutions": r_v, c_v = r_v * 64, c_v * 64
                ref_data[algo_key][m_key].append(r_v)
                results_data[algo_key][m_key].append(c_v)

        print(f" [+] Seed {seed_str} processed.")

    # --- OUTPUT TABLES ---
    found_any = False
    for algo_key, config in ALGORITHM_DICTS.items():
        if algo_key == REFERENCE_ALGO_KEY or not ref_data[algo_key]["max_fitness"]:
            continue
        
        found_any = True
        avg_t = np.mean(matched_times[algo_key])
        print(f"\n--- Comparison: {REFERENCE_ALGO_KEY} vs {algo_key} (Avg Time: {avg_t:.1f}s) ---")
        
        latex = ["\\begin{table}[htbp]", "\\centering", 
                 f"\\caption{{Matched at {avg_t:.1f}s: Ours vs {config['label']}}}",
                 "\\begin{tabular}{lccc}", "\\toprule", 
                 "Metric & Ours & " + config['label'] + " & p-value \\\\", "\\midrule"]

        for m_key, label in METRICS_MAP.items():
            rv, cv = ref_data[algo_key][m_key], results_data[algo_key][m_key]
            rm, rs, cm, cs = np.mean(rv), np.std(rv), np.mean(cv), np.std(cv)

            try:
                p_val = 1.0 if np.array_equal(rv, cv) else wilcoxon(rv, cv).pvalue
                p_str = f"\\textbf{{{p_val:.4f}}}" if p_val < 0.05 else f"{p_val:.4f}"
            except: p_str = "N/A"

            r_txt = f"{rm:.3f} $\\pm$ {rs:.3f}"
            c_txt = f"{cm:.3f} $\\pm$ {cs:.3f}"
            if rm > cm: r_txt = f"\\textbf{{{r_txt}}}"
            elif cm > rm: c_txt = f"\\textbf{{{c_txt}}}"
            latex.append(f"{label} & {r_txt} & {c_txt} & {p_str} \\\\")

        latex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        print("\n".join(latex))

if __name__ == "__main__":
    run_analysis()