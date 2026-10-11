import argparse
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
from sklearn.neighbors import NearestNeighbors
from typing import Tuple

import config

BATCH_SIZE = 64

_device = "cuda" if torch.cuda.is_available() else "cpu"
_inception_model = None

# --- DATASETS & FEATURE UTILS ---

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

def load_inception():
    global _inception_model
    if _inception_model is None:
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
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
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

def calculate_precision_recall(real_features: np.ndarray, gen_features: np.ndarray, k=3) -> Tuple[float, float]:
    if len(gen_features) <= k or len(real_features) <= k: return np.nan, np.nan
    nn_real = NearestNeighbors(n_neighbors=k+1, metric='euclidean', n_jobs=-1).fit(real_features)
    dist_real, _ = nn_real.kneighbors(real_features)
    radii_real = dist_real[:, -1]

    nn_gen = NearestNeighbors(n_neighbors=k+1, metric='euclidean', n_jobs=-1).fit(gen_features)
    dist_gen, _ = nn_gen.kneighbors(gen_features)
    radii_gen = dist_gen[:, -1]

    nn_real_1 = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1).fit(real_features)
    dist_gen_to_real, idx_gen_to_real = nn_real_1.kneighbors(gen_features)
    precision = np.mean(dist_gen_to_real.flatten() <= radii_real[idx_gen_to_real.flatten()])

    nn_gen_1 = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1).fit(gen_features)
    dist_real_to_gen, idx_real_to_gen = nn_gen_1.kneighbors(real_features)
    recall = np.mean(dist_real_to_gen.flatten() <= radii_gen[idx_real_to_gen.flatten()])
    return float(precision), float(recall)

def get_experiment_subdir(base_path: str) -> str:
    if not os.path.exists(base_path): return None
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs: return None
    subdirs.sort() 
    return os.path.join(base_path, subdirs[-1])

# --- FORMATTING & OUTPUT UTILS ---

def generate_latex_table(df: pd.DataFrame, output_path: str):
    """Generates a LaTeX table, bolding the best values per column."""
    
    # Define which columns should be minimized vs maximized
    minimize_cols = ['FID_inf'] # FID is lower-is-better
    maximize_cols = ['P_1000', 'R_1000', 'P_2000', 'R_2000'] # Precision/Recall is higher-is-better
    
    # Identify best values
    best_vals = {}
    for col in minimize_cols:
        if col in df.columns: best_vals[col] = df[col].min()
    for col in maximize_cols:
        if col in df.columns: best_vals[col] = df[col].max()

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Fidelity and Diversity Metrics across combined archives. Metrics are evaluated on subsampled archives to prevent sample size bias. Best results are bolded.}",
        "\\label{tab:combined_metrics}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l c | c | c c | c c}",
        "\\toprule",
        "Algorithm & Total Images & FID$_\\infty$ ($\\downarrow$) & P@1000 ($\\uparrow$) & R@1000 ($\\uparrow$) & P@2000 ($\\uparrow$) & R@2000 ($\\uparrow$) \\\\",
        "\\midrule"
    ]

    for _, row in df.iterrows():
        line = f"{row['Algorithm']} & {int(row['Total_Images'])} "
        
        cols_to_format = ['FID_inf', 'P_1000', 'R_1000', 'P_2000', 'R_2000']
        for col in cols_to_format:
            val = row[col]
            if pd.isna(val):
                line += "& --- "
            else:
                formatted_val = f"{val:.4f}" if col != 'FID_inf' else f"{val:.2f}"
                # Bold if it matches the best value
                if abs(val - best_vals.get(col, -999)) < 1e-5:
                    line += f"& \\textbf{{{formatted_val}}} "
                else:
                    line += f"& {formatted_val} "
        
        line += "\\\\"
        latex_lines.append(line)

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}"
    ])

    with open(output_path, 'w') as f:
        f.write("\n".join(latex_lines))
    print(f"  -> LaTeX table successfully saved to {output_path}")

# --- MAIN RUNNER ---

def run_combined_analysis(model: str):
    cfg = config.get_model_config(model)
    root_dir = cfg["root_dir"]
    output_dir = cfg["output_dir"]
    real_images_path = str(config.COCO_VAL_DIR)

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(real_images_path):
        raise FileNotFoundError(f"Real images not found at {real_images_path}")

    print("Step 1: Calculating Real Image Statistics (COCO)...")
    real_features = get_features_from_folder(real_images_path)
    real_mu = np.mean(real_features, axis=0)
    real_sigma = np.cov(real_features, rowvar=False)

    seeds = [str(seed) for seed in cfg["seeds"]]
    summary_results = []
    
    target_ns_fid = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    target_ns_pr = [700, 1000, 2000]

    print("\nStep 2: Processing Combined Algorithm Archives...")
    for algo_key, algo_config in cfg["algorithms"].items():
        print(f"\nAnalysing: {algo_config['label']}")
        algo_all_feats = []

        for seed in seeds:
            base_path = os.path.join(root_dir, seed, algo_config["path"])
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
                print(f"  ! Seed {seed} failed: {e}")

        if not algo_all_feats:
            continue

        all_feats = np.concatenate(algo_all_feats, axis=0)
        total_images = len(all_feats)
        
        res = {"Algorithm": algo_config['label'], "Total_Images": total_images}
        
        valid_fid_ns = []
        valid_fids = []

        # Subsample for FID
        for n in target_ns_fid:
            if total_images >= n:
                np.random.seed(42)
                idx = np.random.choice(total_images, n, replace=False)
                sub_feats = all_feats[idx]
                gen_mu, gen_sigma = np.mean(sub_feats, axis=0), np.cov(sub_feats, rowvar=False)
                score = calculate_fid(gen_mu, gen_sigma, real_mu, real_sigma)
                
                res[f"FID_{n}"] = score
                valid_fid_ns.append(n)
                valid_fids.append(score)
            else:
                res[f"FID_{n}"] = np.nan

        # FID_inf extrapolation
        if len(valid_fid_ns) >= 2:
            inv_ns = 1.0 / np.array(valid_fid_ns)
            A = np.vstack([inv_ns, np.ones(len(inv_ns))]).T
            _, c = np.linalg.lstsq(A, valid_fids, rcond=None)[0]
            res["FID_inf"] = c
        else:
            res["FID_inf"] = np.nan

        # Subsample for Precision/Recall
        for n in target_ns_pr:
            if total_images >= n:
                np.random.seed(42)  # Fixed seed so every algorithm uses a comparable subset
                idx = np.random.choice(total_images, n, replace=False)
                sub_feats = all_feats[idx]
                
                precision, recall = calculate_precision_recall(real_features, sub_feats)
                res[f"P_{n}"] = precision
                res[f"R_{n}"] = recall
            else:
                res[f"P_{n}"] = np.nan
                res[f"R_{n}"] = np.nan

        summary_results.append(res)
        print(f"  Done. Total: {total_images} | FID_inf: {res['FID_inf']:.2f} | P@1000: {res.get('P_1000', np.nan):.4f} | R@1000: {res.get('R_1000', np.nan):.4f}")

    df = pd.DataFrame(summary_results)
    
    csv_path = os.path.join(output_dir, "fid_results.csv")
    df.to_csv(csv_path, index=False)
    
    tex_path = os.path.join(output_dir, "latex_fid_results.txt")
    generate_latex_table(df, tex_path)
    
    print(f"\nResults successfully written to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID and precision/recall of generated archives against COCO.")
    config.add_model_argument(parser)
    args = parser.parse_args()
    run_combined_analysis(args.model)