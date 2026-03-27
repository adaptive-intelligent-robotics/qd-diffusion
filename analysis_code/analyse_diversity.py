import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Callable
from functools import partial
from scipy.stats import wilcoxon
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
from dreamsim import dreamsim

# --- CONFIG ---
ROOT_PATH = "/workspace/src/hpc_results/output/sdxl-turbo/"
DESCRIPTOR_TYPE = "dino-v2" # Options: "dino-v2", "dreamsim", "clip"
BASELINE_ALGO_KEY = "dns_svd" 

ALGORITHM_DICTS: Dict = {
    "dns_svd": {"path": "dns/multi-hidden-layer-multi-svd-alpha-10", "label": "Spectral-Guided QD-Diffusion (Ours)"},
    "cvt-map-elites": {"path": "cvt-map-elites/multi-hidden-layer-multi-svd-alpha-10", "label": "CVT MAP-Elites"},
    "ablation": {"path": "dns/multi-hidden-layer-multi-svd-threshold-0-alpha-10", "label": "Fitness Threshold Ablation"},
    "cma-es": {"path": "dns/multi-sep-cma-es", "label": "Sep-CMA-ES"},
    "adam": {"path": "dns/multi-adam", "label": "Adam"},
    "qdhf": {"path": "qdhf/multi-hidden-layer-multi-svd-alpha-10", "label": "QDHF"},
    "dns_random": {"path": "dns/random-seeds", "label": "Random Seeds"},
    "dns_hidden_gaussian": {"path": "dns/multi-hidden-layer-gaussian", "label": "Intermed.-layer Gaussian"},
    "dns_mixed": {"path": "dns/mixed-all", "label": "ImageBreeder"},
    "dns_gaussian": {"path": "dns/multi-gaussian", "label": "Gaussian"},
    "dns_isoline": {"path": "dns/multi-isoline", "label": "Isoline-DD"},
}

_device = "cuda" if torch.cuda.is_available() else "cpu"

# --- FEATURE EXTRACTORS ---

def generic_extractor(image_np, processor, model, device, mode="hf"):
    """Standardises image preprocessing and embedding extraction."""
    # Ensure HWC format for PIL
    if image_np.ndim == 3 and image_np.shape[0] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))
    
    # Normalise to [0, 255] uint8
    if image_np.dtype != np.uint8:
        if image_np.min() < 0: # Assume [-1, 1]
            image_np = ((image_np + 1.0) / 2.0 * 255).clip(0, 255)
        else: # Assume [0, 1]
            image_np = (image_np * 255).clip(0, 255)
        image_np = image_np.astype(np.uint8)

    image_pil = Image.fromarray(image_np, mode='RGB')
    
    with torch.no_grad():
        if mode == "dreamsim":
            inputs = processor(image_pil).to(device)
            emb = model.embed(inputs).squeeze(0)
        elif mode == "clip":
            inputs = processor(images=image_pil, return_tensors="pt").to(device)
            emb = model.get_image_features(**inputs).squeeze(0)
        else: # dinov2
            inputs = processor(images=image_pil, return_tensors="pt").to(device)
            output = model(**inputs)
            emb = output.last_hidden_state.mean(dim=1).squeeze(0)
            
    return emb.cpu().numpy()

def get_descriptor_fun(name: str):
    print(f"  [Init] Loading {name}...")
    if name == "dino-v2":
        m = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(_device)
        p = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        return partial(generic_extractor, processor=p, model=m, device=_device, mode="hf")
    elif name == "clip":
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(_device)
        p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return partial(generic_extractor, processor=p, model=m, device=_device, mode="clip")
    elif name == "dreamsim":
        m, p = dreamsim(pretrained=True, dreamsim_type="dino_vitb16")
        m.eval().to(_device)
        return partial(generic_extractor, processor=p, model=m, device=_device, mode="dreamsim")
    raise ValueError(f"Unknown: {name}")

# --- MATH & STATS ---

def calculate_avg_pairwise_distance(images, extract_fn: Callable) -> float:
    """Calculates diversity by averaging Euclidean distances in embedding space."""
    embeddings = []
    
    for i in range(len(images)):
        # Handle the Error 19: Check if current item is Tensor or Numpy
        img_item = images[i]
        if torch.is_tensor(img_item):
            img_np = img_item.detach().cpu().numpy()
        else:
            img_np = img_item # It is already a numpy array
            
        emb = extract_fn(img_np)
        embeddings.append(torch.from_numpy(emb))
    
    embs_tensor = torch.stack(embeddings).to(_device)
    # Pairwise Euclidean Distance
    dist_matrix = torch.cdist(embs_tensor, embs_tensor, p=2)
    
    n = embs_tensor.shape[0]
    if n < 2: return 0.0
    
    # Mean of upper triangle (excluding diagonal)
    sum_dist = torch.triu(dist_matrix, diagonal=1).sum()
    num_pairs = n * (n - 1) / 2
    return (sum_dist / num_pairs).item()

def holm_bonferroni(p_vals):
    p_vals = np.array(p_vals)
    sort_idx = np.argsort(p_vals)
    n = len(p_vals)
    corrected = np.zeros(n)
    for i, idx in enumerate(sort_idx):
        corrected[idx] = min(p_vals[idx] * (n - i), 1.0)
    for i in range(1, n):
        corrected[sort_idx[i]] = max(corrected[sort_idx[i]], corrected[sort_idx[i-1]])
    return corrected

# --- RUNNER ---

def run_analysis():
    extract_fn = get_descriptor_fun(DESCRIPTOR_TYPE)
    seeds = sorted([d for d in os.listdir(ROOT_PATH) if d.isdigit()], key=int)
    
    raw_scores = {k: [] for k in ALGORITHM_DICTS}

    for algo_key, config in ALGORITHM_DICTS.items():
        print(f"\nProcessing {config['label']}...")
        for seed in tqdm(seeds):
            path = os.path.join(ROOT_PATH, str(seed), config["path"])
            if not os.path.exists(path): continue
            
            subdirs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
            if not subdirs: continue
            
            img_file = os.path.join(path, subdirs[-1], "repertoire", "final_images.pt")
            if not os.path.exists(img_file): continue
            
            try:
                # Load images; can be Tensor or Numpy array
                imgs = torch.load(img_file, map_location="cpu", weights_only=False)
                if len(imgs) < 5: continue
                
                # Subsample for speed
                if len(imgs) > 400:
                    indices = np.random.choice(len(imgs), 400, replace=False)
                    imgs = [imgs[i] for i in indices]
                
                score = calculate_avg_pairwise_distance(imgs, extract_fn)
                raw_scores[algo_key].append(score)
            except Exception as e:
                print(f"  ! Seed {seed} failed: {e}")

    # Statistical Comparison
    baseline_vals = raw_scores.get(BASELINE_ALGO_KEY, [])
    if not baseline_vals:
        print("Error: No baseline data found.")
        return

    comparison_keys = [k for k in ALGORITHM_DICTS if k != BASELINE_ALGO_KEY]
    p_values_raw = []
    
    for k in comparison_keys:
        comp_vals = raw_scores[k]
        # Align lengths based on shared seeds/min count
        min_len = min(len(baseline_vals), len(comp_vals))
        if min_len >= 5:
            _, p = wilcoxon(baseline_vals[:min_len], comp_vals[:min_len])
            p_values_raw.append(p)
        else:
            p_values_raw.append(1.0)

    corrected_ps = holm_bonferroni(p_values_raw)
    
    # Format Results
    results = []
    # Add Baseline first
    results.append({
        "Algorithm": f"{ALGORITHM_DICTS[BASELINE_ALGO_KEY]['label']} (Baseline)",
        "Avg Dist": f"{np.mean(baseline_vals):.4f}",
        "Seeds": len(baseline_vals),
        "p-val (adj)": "-"
    })

    for i, k in enumerate(comparison_keys):
        scores = raw_scores[k]
        p_val = corrected_ps[i]
        sig = "*" if p_val < 0.05 else ""
        results.append({
            "Algorithm": ALGORITHM_DICTS[k]["label"],
            "Avg Dist": f"{np.mean(scores):.4f}" if scores else "N/A",
            "Seeds": len(scores),
            "p-val (adj)": f"{p_val:.2e}{sig}" if scores else "N/A"
        })

    df = pd.DataFrame(results)
    print("\n--- Final Diversity Results ---")
    print(df.to_string(index=False))
    df.to_csv(f"analysis_output/diversity_stats_{DESCRIPTOR_TYPE}.csv", index=False)

if __name__ == "__main__":
    run_analysis()