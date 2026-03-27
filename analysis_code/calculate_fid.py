import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from scipy import linalg
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from typing import List, Dict, Tuple

# --- CONFIG ---
ROOT_PATH = "/workspace/src/hpc_results/output/sdxl-turbo/"  
REAL_IMAGES_PATH = "/workspace/src/coco/val2017" # Folder containing COCO .jpg files
BATCH_SIZE = 64

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
_inception_model = None

# --- DATASET LOADER FOR REAL IMAGES ---

class FolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.files[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img

# --- UTILS ---

def load_inception():
    global _inception_model
    if _inception_model is None:
        print("  [Inception] Loading model...")
        weights = Inception_V3_Weights.DEFAULT
        _inception_model = inception_v3(weights=weights, transform_input=False).to(_device)
        _inception_model.fc = torch.nn.Identity() 
        _inception_model.eval()
    return _inception_model

def get_features_from_tensors(images: torch.Tensor, batch_size=BATCH_SIZE) -> np.ndarray:
    model = load_inception()
    features = []
    upsampler = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(_device).float()
        batch = (batch - batch.min()) / (batch.max() - batch.min() + 1e-6)
        batch = (batch * 2) - 1 
        batch = upsampler(batch)
        with torch.no_grad():
            feat = model(batch)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)

def get_features_from_folder(folder_path, batch_size=BATCH_SIZE) -> np.ndarray:
    model = load_inception()
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Scales to [-1, 1]
    ])
    dataset = FolderDataset(folder_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    features = []
    print(f"  -> Extracting features from {len(dataset)} real images...")
    for batch in tqdm(loader):
        batch = batch.to(_device)
        with torch.no_grad():
            feat = model(batch)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)

def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))

def get_experiment_subdir(base_path: str) -> str:
    if not os.path.exists(base_path): return None
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs: return None
    subdirs.sort() 
    return os.path.join(base_path, subdirs[-1])

# --- MAIN RUNNER ---

def run_full_fid_analysis():
    # 1. Get Real Stats
    if not os.path.exists(REAL_IMAGES_PATH):
        print(f"Error: REAL_IMAGES_PATH not found at {REAL_IMAGES_PATH}")
        return

    print("Step 1: Calculating Real Image Statistics (COCO)...")
    real_features = get_features_from_folder(REAL_IMAGES_PATH)
    real_mu = np.mean(real_features, axis=0)
    real_sigma = np.cov(real_features, rowvar=False)

    # 2. Process Algorithms
    seeds = [d for d in os.listdir(ROOT_PATH) if d.isdigit()]
    summary_results = []

    print("\nStep 2: Processing Algorithm Archives...")
    for algo_key, algo_config in ALGORITHM_DICTS.items():
        print(f"\nAnalyzing: {algo_config['label']}")
        algo_all_feats = []

        for seed in seeds:
            base_path = os.path.join(ROOT_PATH, seed, algo_config["path"])
            exp_path = get_experiment_subdir(base_path)
            if not exp_path: continue

            img_path = os.path.join(exp_path, "repertoire", "final_images.pt")
            if not os.path.exists(img_path): continue

            try:
                images = torch.load(img_path, map_location="cpu", weights_only=False)
                if isinstance(images, np.ndarray): images = torch.from_numpy(images)
                if len(images) < 2: continue
                
                feats = get_features_from_tensors(images)
                algo_all_feats.append(feats)
            except Exception as e:
                print(f"  ! Error seed {seed}: {e}")

        if algo_all_feats:
            all_feats = np.concatenate(algo_all_feats, axis=0)
            gen_mu = np.mean(all_feats, axis=0)
            gen_sigma = np.cov(all_feats, rowvar=False)
            
            score = calculate_fid(gen_mu, gen_sigma, real_mu, real_sigma)
            summary_results.append({"Algorithm": algo_config['label'], "FID": score, "Images": len(all_feats)})
            print(f"  Done. FID: {score:.4f} (Total Images: {len(all_feats)})")

    # Output
    df = pd.DataFrame(summary_results)
    df.to_csv("fid_analysis_results.csv", index=False)
    print("\nFinal Results Table:")
    print(df)

if __name__ == "__main__":
    run_full_fid_analysis()