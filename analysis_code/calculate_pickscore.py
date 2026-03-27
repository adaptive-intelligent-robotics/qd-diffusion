import os
import torch
import numpy as np
import pandas as pd
import warnings  # <--- NEW IMPORT
from PIL import Image
from transformers import AutoProcessor, AutoModel
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple

# --- SIMPLE SCRIPT CONFIG ---
# Set these values and run this file directly if you want the
# PATH/SEED/ALGO/EXP_DATE/repertoire/final_images.pt analysis.
ROOT_PATH = "/workspace/src/hpc_results/output/sdxl-turbo/"  
ALGORITHM_DICTS: Dict = {
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

# Optional explicit seed list, e.g. [1, 2, 3].
# Keep as None to auto-discover numeric seed folders in ROOT_PATH.
SEEDS_OVERRIDE = None

# --- SINGLETON MODEL LOADER ---
_pickscore_model = None
_pickscore_processor = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pickscore():
    global _pickscore_model, _pickscore_processor, _device
    
    if _pickscore_model is None:
        print("  [PickScore] Loading model (this happens once)...")
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"
        
        _pickscore_processor = AutoProcessor.from_pretrained(processor_path)
        _pickscore_model = AutoModel.from_pretrained(model_path).eval().to(_device)
        print("  [PickScore] Model loaded.")
        
    return _pickscore_processor, _pickscore_model

def get_pick_scores_batch(prompt: str, image_tensors: torch.Tensor, batch_size=32) -> np.ndarray:
    """
    Calculates RAW PickScore logits for a set of images against a SINGLE specific prompt.
    """
    processor, model = load_pickscore()
    
    all_scores = []
    num_images = image_tensors.shape[0]
    
    # Pre-process text specific to this seed/prompt
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(_device)
    
    with torch.no_grad():
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # Loop through images in batches
        for i in range(0, num_images, batch_size):
            batch_tensors = image_tensors[i : i + batch_size]
            
            # Convert tensors to PIL
            pil_images = []
            for img in batch_tensors:
                img = img.detach().cpu().float()
                # Normalize 0-1
                img = (img - img.min()) / (img.max() - img.min())
                # (C, H, W) -> (H, W, C)
                img_np = img.permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            
            image_inputs = processor(
                images=pil_images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(_device)
            
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            
            # Raw PickScore (Logit)
            scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            all_scores.append(scores.cpu().numpy())

    return np.concatenate(all_scores)

def calculate_dns_scores(fitnesses: np.ndarray, descriptors: np.ndarray, k_neighbors: int = 15) -> np.ndarray:
    """
    Calculates DNS scores: Mean distance to the k-nearest FITTER neighbors.
    Individuals with no fitter neighbors (true peaks) get infinity.
    """
    num_all = len(fitnesses)
    
    # 1. Pairwise Matrix: fitter[i, j] is True if j is fitter than i
    fitter = fitnesses[:, None] < fitnesses[None, :]
    np.fill_diagonal(fitter, False)

    # 2. Pairwise Distances (Euclidean)
    distance = cdist(descriptors, descriptors, metric="euclidean")

    # 3. Mask non-fitter individuals (set dist to infinity)
    distance = np.where(fitter, distance, np.inf)

    # 4. Find K nearest
    k_to_find = min(k_neighbors, num_all - 1)
    
    if k_to_find <= 0:
        return np.full(num_all, np.inf)

    # Partition to get smallest distances efficiently
    # We want indices of the k smallest distances in each row
    k_nearest_indices = np.argpartition(distance, k_to_find, axis=1)[:, :k_to_find]
    k_nearest_dists = np.take_along_axis(distance, k_nearest_indices, axis=1)

    # If distance is inf, it means there weren't enough fitter neighbors
    k_nearest_dists[np.isinf(k_nearest_dists)] = np.nan
    
    # DNS Score is mean distance to fitter neighbors
    # --- FIX: Use standard warnings instead of np.warnings ---
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        dns_score = np.nanmean(k_nearest_dists, axis=1)

    # Revert NaNs (peaks) to Infinity so they are sorted first
    dns_score[np.isnan(dns_score)] = np.inf
    
    return dns_score

def get_experiment_subdir(base_path: str) -> str:
    if not os.path.exists(base_path): return None
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs: return None
    subdirs.sort() 
    return os.path.join(base_path, subdirs[-1])

def analyze_pickscores(
    root_dir: str, 
    algorithm_dicts: Dict, 
    seeds: range, 
    prompts_dict: Dict[int, str],
    top_k: int = 16,
    metric: str = "fitness"
) -> pd.DataFrame:
    
    print("\n-------------------------------------------------------------------------")
    print("                    Calculating PickScore Stats                          ")
    print(f"      (Filtered by Top-{top_k} {metric.upper()} per Seed)")
    print("-------------------------------------------------------------------------")

    results = []

    for algo_key, algo_config in algorithm_dicts.items():
        print(f"Processing {algo_config['label']}...")
        
        # We calculate statistics on the *subset* of top-k images
        stats = {
            "subset_avg": [],  # Average PickScore of the top-k images
            "subset_max": [],  # Best PickScore within the top-k images
        }
        
        for seed in seeds:
            if seed not in prompts_dict: continue
            current_prompt = prompts_dict[seed]
            
            initial_path = os.path.join(root_dir, str(seed), algo_config["path"])
            exp_path = get_experiment_subdir(initial_path)
            if exp_path is None: continue
            
            rep_dir = os.path.join(exp_path, "repertoire")
            img_path = os.path.join(rep_dir, "final_images.pt")
            fit_path = os.path.join(rep_dir, "final_fitnesses.pt")
            desc_path = os.path.join(rep_dir, "final_descriptors.pt")

            # Check required files
            if not os.path.exists(img_path) or not os.path.exists(fit_path):
                continue
            
            try:
                # 1. Load Data (Explicitly set weights_only=False)
                images = torch.load(img_path, map_location="cpu", weights_only=False)
                fitnesses = torch.load(fit_path, map_location="cpu", weights_only=False)
                
                if isinstance(images, np.ndarray): images = torch.from_numpy(images)
                if isinstance(fitnesses, np.ndarray): fitnesses = torch.from_numpy(fitnesses)
                
                if len(images) == 0 or len(fitnesses) == 0: continue
                if len(images) != len(fitnesses):
                    print(f"  !! Warning: Size mismatch seed {seed} (Img: {len(images)}, Fit: {len(fitnesses)}). Skipping.")
                    continue

                # 2. Determine Scores for Sorting
                if metric == "dns":
                    if not os.path.exists(desc_path):
                        print(f"  !! Missing descriptors for seed {seed}. Skipping.")
                        continue
                    
                    descriptors = torch.load(desc_path, map_location="cpu", weights_only=False)
                    if isinstance(descriptors, np.ndarray): descriptors = torch.from_numpy(descriptors)
                    
                    # Calculate DNS
                    scores_np = calculate_dns_scores(fitnesses.numpy(), descriptors.numpy())
                    scores = torch.from_numpy(scores_np)
                else:
                    # Default to fitness
                    scores = fitnesses

                # 3. Sort Descending (Inf/High Fitness is better) and Select Top-K
                k = min(len(scores), top_k)
                sorted_indices = torch.argsort(scores, descending=True)
                top_indices = sorted_indices[:k]
                
                top_images = images[top_indices]
                
                # 4. Calculate PickScores ONLY on these top K images
                pick_scores = get_pick_scores_batch(current_prompt, top_images)
                
                # 5. Record Stats
                stats["subset_avg"].append(np.mean(pick_scores))
                stats["subset_max"].append(np.max(pick_scores))
                    
            except Exception as e:
                print(f"  Error reading seed {seed}: {e}")
        
        if stats["subset_avg"]:
            row = {
                "Algorithm": algo_config["label"],
                f"PickScore (Mean Top-{top_k} {metric})": np.mean(stats["subset_avg"]),
                f"PickScore (Max Top-{top_k} {metric})": np.mean(stats["subset_max"]),
                "Std Error": np.std(stats["subset_avg"]) / np.sqrt(len(stats["subset_avg"])),
            }
            results.append(row)
            val = row[f"PickScore (Mean Top-{top_k} {metric})"]
            print(f"  -> Mean Score: {val:.2f}")

    return pd.DataFrame(results)


def _discover_numeric_seeds(root_dir: str) -> List[int]:
    if not os.path.exists(root_dir):
        return []
    seeds = []
    for name in os.listdir(root_dir):
        full_path = os.path.join(root_dir, name)
        if os.path.isdir(full_path) and name.isdigit():
            seeds.append(int(name))
    return sorted(seeds)


def analyze_final_images_by_algo(
    root_dir: str,
    algorithm_dicts: Dict,
    prompts_by_seed: Dict[int, str],
    seeds: List[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates PickScore means from final_images.pt for each seed and algorithm,
    then averages across seeds.

    Expected structure:
      root_dir/SEED/ALGO/EXP_DATE/repertoire/final_images.pt
    """
    if seeds is None:
        seeds = _discover_numeric_seeds(root_dir)

    per_seed_rows = []
    summary_rows = []

    print("\n------------------------------------------------------------")
    print(" PickScore from final images (per seed and mean over seeds)")
    print("------------------------------------------------------------")

    for algo_key, algo_config in algorithm_dicts.items():
        algo_path = algo_config.get("path")
        algo_label = algo_config.get("label", algo_key)
        if not algo_path:
            print(f"Processing algorithm: {algo_label}")
            print("  -> Missing 'path' in config. Skipping.")
            continue

        algo_seed_means = []
        print(f"Processing algorithm: {algo_label}")

        for seed in seeds:
            if seed not in prompts_by_seed:
                continue

            prompt = prompts_by_seed[seed]
            base_path = os.path.join(root_dir, str(seed), algo_path)
            exp_path = get_experiment_subdir(base_path)
            if exp_path is None:
                continue

            img_path = os.path.join(exp_path, "repertoire", "final_images.pt")
            if not os.path.exists(img_path):
                continue

            try:
                images = torch.load(img_path, map_location="cpu", weights_only=False)
                if isinstance(images, np.ndarray):
                    images = torch.from_numpy(images)
                if len(images) == 0:
                    continue

                pick_scores = get_pick_scores_batch(prompt, images)
                seed_mean = float(np.mean(pick_scores))
                algo_seed_means.append(seed_mean)

                per_seed_rows.append(
                    {
                        "Algorithm": algo_label,
                        "Algorithm Key": algo_key,
                        "Algorithm Path": algo_path,
                        "Seed": seed,
                        "Num Images": int(len(images)),
                        "PickScore Mean": seed_mean,
                    }
                )
            except Exception as e:
                print(f"  Error processing seed {seed} for {algo_label}: {e}")

        if algo_seed_means:
            summary_rows.append(
                {
                    "Algorithm": algo_label,
                    "Algorithm Key": algo_key,
                    "Algorithm Path": algo_path,
                    "Mean PickScore Across Seeds": float(np.mean(algo_seed_means)),
                    "Std Error": float(np.std(algo_seed_means) / np.sqrt(len(algo_seed_means))),
                    "Seeds Used": int(len(algo_seed_means)),
                }
            )
            print(f"  -> Mean across seeds: {np.mean(algo_seed_means):.4f}")
        else:
            print("  -> No valid seed runs found.")

    return pd.DataFrame(per_seed_rows), pd.DataFrame(summary_rows)


if __name__ == "__main__":
    if not ROOT_PATH:
        raise ValueError("Set ROOT_PATH at the top of this file before running.")
    if not ALGORITHM_DICTS:
        raise ValueError("Set ALGORITHM_DICTS (dict with 'path' and 'label') at the top of this file before running.")

    from prompts import prompts_dict

    seeds_to_use = SEEDS_OVERRIDE if SEEDS_OVERRIDE is not None else None

    per_seed_df, summary_df = analyze_final_images_by_algo(
        root_dir=ROOT_PATH,
        algorithm_dicts=ALGORITHM_DICTS,
        prompts_by_seed=prompts_dict,
        seeds=seeds_to_use,
    )

    print("\nPer-seed results:")
    print(per_seed_df)
    print("\nPer-algorithm summary:")
    print(summary_df)

    if not per_seed_df.empty:
        per_seed_df.to_csv("pickscore_per_seed.csv", index=False)
    if not summary_df.empty:
        summary_df.to_csv("pickscore_summary.csv", index=False)