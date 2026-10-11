import argparse
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import Callable
from functools import partial
from scipy.stats import wilcoxon
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
from dreamsim import dreamsim

import config

DESCRIPTOR_TYPE = "clip"  # Options: "dino-v2", "dreamsim", "clip"

_device = "cuda" if torch.cuda.is_available() else "cpu"

# FEATURE EXTRACTORS

def generic_extractor(image_np, processor, model, device, mode="hf"):
    """Standardises image preprocessing and embedding extraction."""
    if image_np.ndim == 3 and image_np.shape[0] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))
    
    if image_np.dtype != np.uint8:
        if image_np.min() < 0: 
            image_np = ((image_np + 1.0) / 2.0 * 255).clip(0, 255)
        else: 
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
        else: 
            inputs = processor(images=image_pil, return_tensors="pt").to(device)
            output = model(**inputs)
            emb = output.last_hidden_state.mean(dim=1).squeeze(0)
            
    return emb.cpu().numpy()

def get_descriptor_fun(name: str):
    print(f"  [Init] Loading {name}...")
    if name == "dino-v2":
        # use_safetensors avoids the torch.load vulnerability check
        m = AutoModel.from_pretrained("facebook/dinov2-base", use_safetensors=True).eval().to(_device)
        p = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        return partial(generic_extractor, processor=p, model=m, device=_device, mode="hf")
    elif name == "clip":
        # use_safetensors avoids the torch.load vulnerability check
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).eval().to(_device)
        p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return partial(generic_extractor, processor=p, model=m, device=_device, mode="clip")
    elif name == "dreamsim":
        m, p = dreamsim(pretrained=True, dreamsim_type="dino_vitb16")
        m.eval().to(_device)
        return partial(generic_extractor, processor=p, model=m, device=_device, mode="dreamsim")
    raise ValueError(f"Unknown: {name}")

# MATH & STATS

def calculate_avg_pairwise_distance(images, extract_fn: Callable) -> float:
    """Calculates diversity by averaging Euclidean distances in embedding space."""
    embeddings = []
    
    for i in range(len(images)):
        img_item = images[i]
        if torch.is_tensor(img_item):
            img_np = img_item.detach().cpu().numpy()
        else:
            img_np = img_item 
            
        emb = extract_fn(img_np)
        embeddings.append(torch.from_numpy(emb))
    
    embs_tensor = torch.stack(embeddings).to(_device)
    dist_matrix = torch.cdist(embs_tensor, embs_tensor, p=2)
    
    n = embs_tensor.shape[0]
    if n < 2: return 0.0
    
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

# RUNNER

def run_analysis(model: str):
    cfg = config.get_model_config(model)
    root_dir = cfg["root_dir"]
    output_dir = cfg["output_dir"]
    algorithm_dicts = cfg["algorithms"]
    baseline_key = config.BASE_ALGO_KEY

    extract_fn = get_descriptor_fun(DESCRIPTOR_TYPE)
    seeds = cfg["seeds"]
    
    raw_scores = {k: [] for k in algorithm_dicts}

    for algo_key, algo_config in algorithm_dicts.items():
        print(f"\nProcessing {algo_config['label']}...")
        for seed in tqdm(seeds):
            path = os.path.join(root_dir, str(seed), algo_config["path"])
            if not os.path.exists(path): continue
            
            subdirs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
            if not subdirs: continue
            
            img_file = os.path.join(path, subdirs[-1], "repertoire", "final_images.pt")
            if not os.path.exists(img_file): continue
            
            try:
                imgs = torch.load(img_file, map_location="cpu", weights_only=False)
                if len(imgs) < 5: continue
                
                if len(imgs) > 400:
                    indices = np.random.choice(len(imgs), 400, replace=False)
                    imgs = [imgs[i] for i in indices]
                
                score = calculate_avg_pairwise_distance(imgs, extract_fn)
                raw_scores[algo_key].append(score)
            except Exception as e:
                print(f"  ! Seed {seed} failed: {e}")

    baseline_vals = raw_scores.get(baseline_key, [])
    if not baseline_vals:
        print("Error: No baseline data found.")
        return

    comparison_keys = [k for k in algorithm_dicts if k != baseline_key]
    p_values_raw = []
    
    for k in comparison_keys:
        comp_vals = raw_scores[k]
        min_len = min(len(baseline_vals), len(comp_vals))
        if min_len >= 5:
            _, p = wilcoxon(baseline_vals[:min_len], comp_vals[:min_len])
            p_values_raw.append(p)
        else:
            p_values_raw.append(1.0)

    corrected_ps = holm_bonferroni(p_values_raw)
    
    results = []
    results.append({
        "Algorithm": f"{algorithm_dicts[baseline_key]['label']} (Baseline)",
        "Avg Dist": f"{np.mean(baseline_vals):.4f}",
        "Seeds": len(baseline_vals),
        "p-val (adj)": "-"
    })

    for i, k in enumerate(comparison_keys):
        scores = raw_scores[k]
        p_val = corrected_ps[i]
        sig = "*" if p_val < 0.05 else ""
        results.append({
            "Algorithm": algorithm_dicts[k]["label"],
            "Avg Dist": f"{np.mean(scores):.4f}" if scores else "N/A",
            "Seeds": len(scores),
            "p-val (adj)": f"{p_val:.2e}{sig}" if scores else "N/A"
        })

    df = pd.DataFrame(results)
    print("\nFinal Diversity Results")
    print(df.to_string(index=False))
    
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"diversity_stats_{DESCRIPTOR_TYPE}.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare archive diversity in a learned embedding space.")
    config.add_model_argument(parser)
    args = parser.parse_args()
    run_analysis(args.model)