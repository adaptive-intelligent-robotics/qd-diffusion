import os
import torch
import numpy as np
import pandas as pd
import warnings
from PIL import Image
from typing import List, Dict

# --- SAFE IMPORT ---
try:
    import ImageReward
    IMAGE_REWARD_AVAILABLE = True
except ImportError as e:
    print(f"!! WARNING: Could not import ImageReward: {e}")
    print("!! Ensure you ran: pip install image-reward")
    IMAGE_REWARD_AVAILABLE = False

# --- SINGLETON LOADER ---
_ir_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_imagereward():
    global _ir_model, _device, IMAGE_REWARD_AVAILABLE
    
    if not IMAGE_REWARD_AVAILABLE:
        return None

    if _ir_model is None:
        print("  [ImageReward] Loading model (this happens once)...")
        try:
            # Downloads/loads ImageReward-v1.0 to cache
            _ir_model = ImageReward.load("ImageReward-v1.0", device=_device)
            print("  [ImageReward] Model loaded.")
        except Exception as e:
            print(f"  !! Error loading ImageReward model: {e}")
            IMAGE_REWARD_AVAILABLE = False # Disable to stop retrying
            return None
        
    return _ir_model

def get_imagereward_scores_batch(prompt: str, image_tensors: torch.Tensor, batch_size=16) -> np.ndarray:
    """
    Calculates ImageReward scores for a batch of images.
    """
    model = load_imagereward()
    
    # If model failed to load or lib missing, return NaNs
    if model is None:
        return np.full(len(image_tensors), np.nan)

    all_scores = []
    num_images = image_tensors.shape[0]

    for i in range(0, num_images, batch_size):
        batch_tensors = image_tensors[i : i + batch_size]
        
        # Convert to PIL
        pil_images = []
        for img in batch_tensors:
            img = img.detach().cpu().float()
            if img.max() > 1.0: img /= 255.0
            img = (img - img.min()) / (img.max() - img.min())
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
        
        with torch.no_grad():
            try:
                # ImageReward score() returns a float or list of floats
                scores = model.score(prompt, pil_images)
                if isinstance(scores, float): scores = [scores]
                all_scores.extend(scores)
            except Exception as e:
                print(f"  !! ImageReward inference error: {e}")
                all_scores.extend([np.nan] * len(pil_images))

    return np.array(all_scores)

# --- REUSED HELPER ---
def get_experiment_subdir(base_path: str) -> str:
    if not os.path.exists(base_path): return None
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs: return None
    subdirs.sort() 
    return os.path.join(base_path, subdirs[-1])

def analyze_imagereward(
    root_dir: str, 
    algorithm_dicts: Dict, 
    seeds: range, 
    prompts_dict: Dict[int, str],
    top_k: int = 16,
    metric: str = "fitness"
) -> pd.DataFrame:
    
    if not IMAGE_REWARD_AVAILABLE:
        print("\n[ImageReward] Skipping analysis (library not available).")
        return pd.DataFrame()

    print("\n-------------------------------------------------------------------------")
    print("                    Calculating ImageReward Scores                       ")
    print(f"      (Filtered by Top-{top_k} {metric.upper()})")
    print("-------------------------------------------------------------------------")

    results = []

    for algo_key, algo_config in algorithm_dicts.items():
        print(f"Processing {algo_config['label']}...")
        stats = {"subset_avg": [], "subset_max": []}
        
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

            if not os.path.exists(img_path) or not os.path.exists(fit_path): continue
            
            try:
                images = torch.load(img_path, map_location="cpu", weights_only=False)
                fitnesses = torch.load(fit_path, map_location="cpu", weights_only=False)
                
                if isinstance(images, np.ndarray): images = torch.from_numpy(images)
                if isinstance(fitnesses, np.ndarray): fitnesses = torch.from_numpy(fitnesses)
                if len(images) == 0: continue

                # Sorting Logic
                if metric == "dns":
                    if os.path.exists(desc_path):
                         # Lazy import DNS logic
                         from analysis_code.calculate_pickscore import calculate_dns_scores
                         descriptors = torch.load(desc_path, map_location="cpu", weights_only=False)
                         if isinstance(descriptors, np.ndarray): descriptors = torch.from_numpy(descriptors)
                         scores = calculate_dns_scores(fitnesses.numpy(), descriptors.numpy())
                         scores = torch.from_numpy(scores)
                    else:
                        scores = fitnesses
                else:
                    scores = fitnesses # Default
                
                # --- SORTING ---
                k = min(len(scores), top_k)
                sorted_indices = torch.argsort(scores, descending=True)
                top_indices = sorted_indices[:k]
                top_images = images[top_indices]
                
                # --- CALC IMAGEREWARD ---
                ir_scores = get_imagereward_scores_batch(current_prompt, top_images)
                
                # Filter NaNs
                ir_scores = ir_scores[~np.isnan(ir_scores)]
                
                if len(ir_scores) > 0:
                    stats["subset_avg"].append(np.mean(ir_scores))
                    stats["subset_max"].append(np.max(ir_scores))

            except Exception as e:
                print(f"  Error {seed}: {e}")

        if stats["subset_avg"]:
            row = {
                "Algorithm": algo_config["label"],
                f"ImageReward (Mean Top-{top_k} {metric})": np.mean(stats["subset_avg"]),
                f"ImageReward (Max Top-{top_k} {metric})": np.mean(stats["subset_max"]),
                "Std Error": np.std(stats["subset_avg"]) / np.sqrt(len(stats["subset_avg"])),
            }
            results.append(row)
            print(f"  -> Mean ImageReward: {row[f'ImageReward (Mean Top-{top_k} {metric})']:.2f}")

    return pd.DataFrame(results)