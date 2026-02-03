import os
import torch
import numpy as np
import math
import warnings  # <--- NEW IMPORT
from PIL import Image
from scipy.spatial.distance import cdist
from typing import List

# --- VISUAL SETTINGS ---
THUMBNAIL_SIZE = (512, 512) 
GRID_PADDING = 10
BACKGROUND_COLOR = "white"

def get_experiment_subdir(base_path: str) -> str:
    if not os.path.exists(base_path): return None
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs: return None
    subdirs.sort() 
    return os.path.join(base_path, subdirs[-1])

def tensor_to_pil(tensor):
    img = tensor.detach().cpu().float()
    if img.min() < 0 or img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min())
    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)

def create_grid_image(images: List[Image.Image], rows, cols, padding=GRID_PADDING, bg_color=BACKGROUND_COLOR):
    w, h = THUMBNAIL_SIZE
    grid_w = (w * cols) + (padding * (cols - 1))
    grid_h = (h * rows) + (padding * (rows - 1))
    grid = Image.new('RGB', (grid_w, grid_h), bg_color)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        if img.size != THUMBNAIL_SIZE: img = img.resize(THUMBNAIL_SIZE)
        x = c * (w + padding)
        y = r * (h + padding)
        grid.paste(img, (x, y))
    return grid

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

def generate_top_k_grids(root_dir, seeds, algorithm_dicts, k=16, metric="fitness"):
    print("\n-------------------------------------------------------------------------")
    print(f"            Generating Top-{k} Grid Plots (Metric: {metric.upper()})      ")
    print("-------------------------------------------------------------------------")

    rows = int(math.sqrt(k))
    cols = math.ceil(k / rows)

    for algo_key, algo_config in algorithm_dicts.items():
        print(f"Processing {algo_config['label']}...")
        
        for seed in seeds:
            initial_path = os.path.join(root_dir, str(seed), algo_config["path"])
            exp_path = get_experiment_subdir(initial_path)
            if exp_path is None: continue
            
            rep_dir = os.path.join(exp_path, "repertoire")
            images_output_dir = os.path.join(exp_path, "images")
            os.makedirs(images_output_dir, exist_ok=True)
            
            fit_path = os.path.join(rep_dir, "final_fitnesses.pt")
            img_path = os.path.join(rep_dir, "final_images.pt")
            desc_path = os.path.join(rep_dir, "final_descriptors.pt")
            
            if os.path.exists(fit_path) and os.path.exists(img_path):
                try:
                    fitnesses = torch.load(fit_path, map_location="cpu", weights_only=False)
                    images_tensor = torch.load(img_path, map_location="cpu", weights_only=False)
                    
                    if isinstance(fitnesses, np.ndarray): fitnesses = torch.from_numpy(fitnesses)
                    if isinstance(images_tensor, np.ndarray): images_tensor = torch.from_numpy(images_tensor)

                    # Determine sorting scores
                    if metric == "dns":
                        if not os.path.exists(desc_path):
                            print(f"  !! Missing descriptors for seed {seed}. Skipping.")
                            continue
                        
                        descriptors = torch.load(desc_path, map_location="cpu", weights_only=False)
                        if isinstance(descriptors, np.ndarray): descriptors = torch.from_numpy(descriptors)
                        
                        scores_np = calculate_dns_scores(fitnesses.numpy(), descriptors.numpy())
                        scores = torch.from_numpy(scores_np)
                    elif metric == "mixed":
                        # First, get top half by fitness
                        if not os.path.exists(desc_path):
                            print(f"  !! Missing descriptors for seed {seed}. Skipping.")
                            continue
                        
                        descriptors = torch.load(desc_path, map_location="cpu", weights_only=False)
                        if isinstance(descriptors, np.ndarray): descriptors = torch.from_numpy(descriptors)
                        
                        # Sort by fitness and take top half
                        fitness_sorted_indices = torch.argsort(fitnesses, descending=True)
                        half_size = len(fitnesses) // 2
                        top_half_indices = fitness_sorted_indices[:half_size]
                        
                        # Calculate DNS scores only on top half
                        top_half_fitnesses = fitnesses[top_half_indices].numpy()
                        top_half_descriptors = descriptors[top_half_indices].numpy()
                        dns_scores_np = calculate_dns_scores(top_half_fitnesses, top_half_descriptors)
                        
                        # Sort by DNS and get top k indices (relative to top_half)
                        dns_scores = torch.from_numpy(dns_scores_np)
                        dns_sorted_indices = torch.argsort(dns_scores, descending=True)
                        top_k_in_half = dns_sorted_indices[:k]
                        
                        # Map back to original indices
                        top_indices = top_half_indices[top_k_in_half]
                    else:
                        # Default to fitness
                        scores = fitnesses

                    # Sort Descending (Inf/High Fitness is better) - only if not "mixed"
                    if metric != "mixed":
                        sorted_indices = torch.argsort(scores, descending=True)
                        top_indices = sorted_indices[:k]
                    
                    top_images_tensors = images_tensor[top_indices]
                    pil_images = [tensor_to_pil(t) for t in top_images_tensors]
                    
                    while len(pil_images) < k:
                        pil_images.append(Image.new('RGB', THUMBNAIL_SIZE, "black"))

                    grid_img = create_grid_image(pil_images, rows, cols)
                    
                    filename = f"top_{k}_{metric}_grid.png"
                    save_path = os.path.join(images_output_dir, filename)
                    grid_img.save(save_path)
                    
                except Exception as e:
                    print(f"  Error processing seed {seed}: {e}")

    print("Grid generation complete.")