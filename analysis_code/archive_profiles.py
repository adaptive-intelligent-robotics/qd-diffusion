import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict
from transformers import CLIPModel, CLIPTokenizerFast

try:
    from prompts import prompts_dict
except ImportError:
    prompts_dict = {}

# --- CONFIG ---
ROOT_PATH = "/workspace/src/hpc_results/output/sdxl-turbo/"
OUTPUT_DIR = "analysis_output/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_ARCHIVE_CELLS = 4096  # Change based on your grid resolution

ALGORITHM_DICTS: Dict = {
    "dns_svd": {"path": "dns/multi-hidden-layer-multi-svd-alpha-10", "label": "Spectral-Guided QD-Diffusion (Ours)", "color": "#d62728"},
    # "cvt-map-elites": {"path": "cvt-map-elites/multi-hidden-layer-multi-svd-alpha-10", "label": "CVT MAP-Elites", "color": "#1f77b4"},
    # "ablation": {"path": "dns/multi-hidden-layer-multi-svd-threshold-0-alpha-10", "label": "Fitness Threshold Ablation", "color": "#ff7f0e"},
    # "cma-es": {"path": "dns/multi-sep-cma-es", "label": "Sep-CMA-ES", "color": "#2ca02c"},
    # "adam": {"path": "dns/multi-adam", "label": "Adam", "color": "#9467bd"},
    # "qdhf": {"path": "qdhf/multi-hidden-layer-multi-svd-alpha-10", "label": "QDHF", "color": "#8c564b"},
    # Add or uncomment others as needed
    "dns_random": {"path": "dns/random-seeds", "label": "Random Seeds", "color": "#1f77b4"},
    "dns_hidden_gaussian": {"path": "dns/multi-hidden-layer-gaussian", "label": "Intermed.-layer Gaussian", "color": "#ff7f0e"},
    "dns_mixed": {"path": "dns/mixed-all", "label": "ImageBreeder", "color": "#2ca02c"},
    "dns_gaussian": {"path": "dns/multi-gaussian", "label": "Gaussian", "color": "#9467bd"},
    "dns_isoline": {"path": "dns/multi-isoline", "label": "Isoline-DD", "color": "#8c564b"},
}

# --- CLIP SETUP ---
print(f"Initialising CLIP on {DEVICE}...")
clip_tokenizer = CLIPTokenizerFast.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer", cache_dir="/var/tmp/hf")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True, cache_dir="/var/tmp/hf").to(DEVICE).eval()

def get_experiment_subdir(base_path: str) -> str:
    if not os.path.exists(base_path): return None
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs: return None
    subdirs.sort() 
    return os.path.join(base_path, subdirs[-1])

def compute_clip_fitness(images_tensor, prompt):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(DEVICE)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(DEVICE)
    
    # Ensure tensor type and move to device
    images_tensor = images_tensor.to(DEVICE).float()
    
    norm_imgs = F.interpolate(images_tensor, size=224, mode='bilinear')
    norm_imgs = (norm_imgs - mean[None, :, None, None]) / std[None, :, None, None]
    
    text_inputs = clip_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

    with torch.no_grad():
        image_features = clip_model.get_image_features(norm_imgs)
        text_features = clip_model.get_text_features(**text_inputs)

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    return (image_features @ text_features.T).squeeze().cpu().numpy()

def load_data():
    seeds = sorted([d for d in os.listdir(ROOT_PATH) if d.isdigit()], key=int)
    results = {k: [] for k in ALGORITHM_DICTS}

    for algo_key, config in ALGORITHM_DICTS.items():
        print(f"\nLoading {config['label']}...")
        for seed_str in tqdm(seeds):
            seed = int(seed_str)
            path = get_experiment_subdir(os.path.join(ROOT_PATH, seed_str, config["path"]))
            if not path: continue
            
            rep_dir = os.path.join(path, "repertoire")
            try:
                if algo_key == "qdhf":
                    img_path = os.path.join(rep_dir, "final_images.pt")
                    if not os.path.exists(img_path): continue
                    
                    data_in = torch.load(img_path, map_location="cpu", weights_only=False)
                    
                    # Robust conversion to Tensor
                    if isinstance(data_in, list):
                        raw_imgs = torch.stack([torch.as_tensor(i) for i in data_in])
                    elif isinstance(data_in, np.ndarray):
                        raw_imgs = torch.from_numpy(data_in)
                    else:
                        raw_imgs = data_in # Already a tensor
                    
                    # Normalise uint8 to [0, 1]
                    if raw_imgs.dtype == torch.uint8:
                        raw_imgs = raw_imgs.float() / 255.0
                    
                    prompt = prompts_dict.get(seed, "a high quality image")
                    batch_size, scores = 32, []
                    
                    for i in range(0, len(raw_imgs), batch_size):
                        batch = raw_imgs[i : i+batch_size]
                        # Final safety check for the batch itself
                        if isinstance(batch, np.ndarray):
                            batch = torch.from_numpy(batch)
                            
                        batch_scores = compute_clip_fitness(batch, prompt)
                        if batch_scores.ndim == 0: batch_scores = np.array([batch_scores])
                        scores.append(batch_scores)
                    fits = np.concatenate(scores)
                else:
                    fit_path = os.path.join(rep_dir, "final_fitnesses.pt")
                    if not os.path.exists(fit_path): continue
                    fits = torch.load(fit_path, map_location="cpu", weights_only=False)
                    if torch.is_tensor(fits): fits = fits.detach().cpu().numpy().flatten()
                    
                    fits = fits[np.isfinite(fits)]
                    if len(fits) > 0:
                        # Only keep filled cells (better than the initialisation placeholder)
                        fits = fits[fits > fits.min()]

                if len(fits) > 0:
                    results[algo_key].append(fits)
            except Exception as e:
                print(f"  ! Seed {seed} error: {e}")
    return results

def run_analysis(data):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_vals = np.concatenate([np.concatenate(r) for r in data.values() if r])
    x_range = np.linspace(all_vals.min(), all_vals.max(), 250)
    stats = []

    for algo_key, runs in data.items():
        if not runs: continue
        
        profiles = np.array([[np.sum(r >= x) for x in x_range] for r in runs])
        mean_p, std_p = np.mean(profiles, axis=0), np.std(profiles, axis=0)
        
        cfg = ALGORITHM_DICTS[algo_key]
        ax.plot(x_range, mean_p, label=cfg["label"], color=cfg["color"], lw=2.5)
        ax.fill_between(x_range, mean_p - std_p, mean_p + std_p, color=cfg["color"], alpha=0.15)
        
        # Calculate summary metrics
        cov_pct = (np.mean([len(r) for r in runs]) / TOTAL_ARCHIVE_CELLS) * 100
        avg_fit = np.mean([np.mean(r) for r in runs])
        stats.append({
            "Algorithm": cfg["label"],
            "Avg Coverage": f"{cov_pct:.2f}%",
            "Avg Avg Fitness": f"{avg_fit:.4f}",
            "Max Fitness": f"{np.max([np.max(r) for r in runs]):.4f}",
            "QD-Score": f"{np.mean([np.sum(r) for r in runs]):.2f}"
        })

    ax.set_title("Archive Performance Profiles (Aligned CLIP Fitness)", fontsize=14)
    ax.set_xlabel("Fitness Threshold (CLIP Similarity)")
    ax.set_ylabel("Number of Solutions in Repertoire")
    ax.legend(facecolor='white', loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "archive_profiles.png"), dpi=300)
    
    print("\n--- Summary Statistics ---")
    print(pd.DataFrame(stats).to_string(index=False))

if __name__ == "__main__":
    processed_data = load_data()
    run_analysis(processed_data)