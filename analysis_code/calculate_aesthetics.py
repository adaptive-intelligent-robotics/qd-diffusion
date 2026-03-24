import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import urllib.request
import warnings
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from typing import List, Dict

# --- SINGLETON LOADER ---
_clip_model = None
_clip_processor = None
_aesthetic_head = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

# --- MLP ARCHITECTURE (for robust loading) ---
class AestheticMLP(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Alternative MLP architecture (no dropout, just Linear+ReLU)
class AestheticMLP_NoDropout(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),  # 0
            nn.ReLU(),                    # 1
            nn.Linear(1024, 128),         # 2
            nn.ReLU(),                    # 3
            nn.Linear(128, 64),           # 4
            nn.ReLU(),                    # 5
            nn.Linear(64, 16),            # 6
            nn.ReLU(),                    # 7
            nn.Linear(16, 1)              # 8
        )

    def forward(self, x):
        return self.layers(x)

# Dynamic MLP builder from state dict
def build_mlp_from_state_dict(state_dict):
    """Build an MLP that matches the structure in the state dict."""
    # Extract linear layer info
    linear_layers = []
    for key in sorted(state_dict.keys()):
        if '.weight' in key and 'layers.' in key:
            layer_idx = int(key.split('.')[1])
            weight_shape = state_dict[key].shape  # (out_features, in_features)
            linear_layers.append((layer_idx, weight_shape[1], weight_shape[0]))
    
    if not linear_layers:
        return None
    
    # Build Sequential model
    modules = []
    last_idx = -1
    for idx, in_features, out_features in linear_layers:
        # Add Linear layer at correct index
        while len(modules) < idx:
            modules.append(nn.ReLU())  # Fill gaps with ReLU
        modules.append(nn.Linear(in_features, out_features))
        last_idx = idx
    
    return nn.Sequential(*modules)

def load_aesthetic_model():
    global _clip_model, _clip_processor, _aesthetic_head, _device
    
    if _clip_model is None:
        print("  [Aesthetics] Loading CLIP + Aesthetic Head...")
        
        # 1. Load CLIP (ViT-L/14 is standard for LAION-Aesthetics)
        model_id = "openai/clip-vit-large-patch14"
        try:
            _clip_processor = CLIPProcessor.from_pretrained(model_id)
            _clip_model = CLIPModel.from_pretrained(model_id).eval().to(_device)
        except Exception as e:
            print(f"!! Error loading CLIP: {e}")
            return None, None, None
        
        # 2. Prepare Weights File
        save_path = "analysis_code/aesthetic_predictor.pth"
        # Default URL (Linear) - but we support whatever is on disk
        url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
        
        if not os.path.exists(save_path):
            print("  Downloading aesthetic predictor weights...")
            try:
                urllib.request.urlretrieve(url, save_path)
            except Exception as e:
                print(f"!! Error downloading aesthetic predictor: {e}")
                return None, None, None
            
        # 3. Load Weights & Determine Architecture
        try:
            state_dict = torch.load(save_path, map_location="cpu")
        except Exception as e:
             print(f"!! Error reading state dict: {e}")
             return None, None, None

        # Debug: Print state dict structure
        layer_keys = [k for k in state_dict.keys() if 'weight' in k]
        print(f"  [Debug] Found {len(layer_keys)} weight layers: {layer_keys[:5]}...")
        
        # Check architecture and try multiple model configurations
        model_loaded = False
        
        if "layers.0.weight" in state_dict:
            print("  [Aesthetics] Detected MLP architecture.")
            
            # Inspect the actual architecture from state dict
            linear_layers = []
            for key in sorted(state_dict.keys()):
                if '.weight' in key and 'layers.' in key:
                    layer_idx = int(key.split('.')[1])
                    weight_shape = state_dict[key].shape
                    linear_layers.append((layer_idx, weight_shape))
            
            print(f"  [Debug] Linear layers found: {linear_layers}")
            
            # First try: Build dynamic model from state dict
            try:
                print("  [Aesthetics] Attempting dynamic architecture matching...")
                dynamic_layers = build_mlp_from_state_dict(state_dict)
                if dynamic_layers is not None:
                    class DynamicAestheticMLP(nn.Module):
                        def __init__(self, layers):
                            super().__init__()
                            self.layers = layers
                        def forward(self, x):
                            return self.layers(x)
                    
                    _aesthetic_head = DynamicAestheticMLP(dynamic_layers)
                    _aesthetic_head.load_state_dict(state_dict, strict=True)
                    _aesthetic_head.eval()
                    _aesthetic_head.to(_device)
                    print(f"  [Aesthetics] Dynamic model loaded successfully on {_device}.")
                    model_loaded = True
            except Exception as e:
                print(f"  [Debug] Dynamic loading failed: {e}")
            
            # Fallback: Try different MLP architectures (order matters - try most common first)
            if not model_loaded:
                architectures = [
                    ("No-Dropout MLP", AestheticMLP_NoDropout),
                    ("Standard MLP", AestheticMLP),
                ]
                
                for arch_name, arch_class in architectures:
                    try:
                        _aesthetic_head = arch_class(input_dim=768)
                        # Try strict loading first
                        _aesthetic_head.load_state_dict(state_dict, strict=True)
                        _aesthetic_head.eval()
                        _aesthetic_head.to(_device)
                        # Verify model is on correct device
                        for param in _aesthetic_head.parameters():
                            assert param.device.type == _device.split(':')[0], f"Model param on {param.device}, expected {_device}"
                            break  # Just check first param
                        print(f"  [Aesthetics] {arch_name} loaded successfully (strict) on {_device}.")
                        model_loaded = True
                        break
                    except Exception as e:
                        # Try non-strict loading
                        try:
                            _aesthetic_head = arch_class(input_dim=768) 
                            missing_keys, unexpected_keys = _aesthetic_head.load_state_dict(state_dict, strict=False)
                            if len(missing_keys) == 0:  # Only accept if no missing keys
                                _aesthetic_head.eval()
                                _aesthetic_head.to(_device)
                                # Verify model is on correct device
                                for param in _aesthetic_head.parameters():
                                    assert param.device.type == _device.split(':')[0], f"Model param on {param.device}, expected {_device}"
                                    break  # Just check first param
                                print(f"  [Aesthetics] {arch_name} loaded successfully (non-strict) on {_device}.")
                                if unexpected_keys:
                                    print(f"    Warning: Ignored keys {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")
                                model_loaded = True
                                break
                        except Exception:
                            continue
                        
            if not model_loaded:
                print(f"!! Error loading MLP: Could not match any architecture")
        else:
            # It's Linear
            print("  [Aesthetics] Detected Linear architecture.")
            try:
                _aesthetic_head = nn.Linear(768, 1)
                _aesthetic_head.load_state_dict(state_dict)
                _aesthetic_head.eval()
                _aesthetic_head.to(_device)
                print(f"  [Aesthetics] Linear model loaded successfully on {_device}.")
                model_loaded = True
            except Exception as e:
                print(f"!! Error loading Linear model: {e}")
        
        if not model_loaded:
            # Last resort: delete file to force re-download next time
            try: os.remove(save_path) 
            except: pass
            return None, None, None
        
    return _clip_processor, _clip_model, _aesthetic_head

def get_aesthetic_scores_batch(image_tensors: torch.Tensor, batch_size=32) -> np.ndarray:
    """
    Calculates Aesthetic Score (1-10) for a set of images.
    """
    processor, clip_model, aesthetic_head = load_aesthetic_model()
    
    # Safety check if loading failed
    if clip_model is None or aesthetic_head is None: 
        return np.full(len(image_tensors), np.nan)
    
    all_scores = []
    num_images = image_tensors.shape[0]
    
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch_tensors = image_tensors[i : i + batch_size]
            
            # Convert tensors to PIL
            pil_images = []
            for img in batch_tensors:
                img = img.detach().cpu().float()
                # Normalize 0-1
                if img.max() > 1.0: img = img / 255.0 
                img = (img - img.min()) / (img.max() - img.min())
                img_np = img.permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            
            # Get CLIP Embeddings
            # Ensure inputs are on the correct device
            inputs = processor(images=pil_images, return_tensors="pt", padding=True)
            # Move all input tensors to device
            inputs = {k: v.to(_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            outputs = clip_model.get_image_features(**inputs)
            image_embeds = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
            # Predict Score
            # Ensure everything is on the same device and float32
            image_embeds = image_embeds.float().to(_device)
            scores = aesthetic_head(image_embeds)
            all_scores.append(scores.detach().cpu().numpy().flatten())

    return np.concatenate(all_scores)

def get_experiment_subdir(base_path: str) -> str:
    if not os.path.exists(base_path): return None
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs: return None
    subdirs.sort() 
    return os.path.join(base_path, subdirs[-1])

def analyze_aesthetics(
    root_dir: str, 
    algorithm_dicts: Dict, 
    seeds: range, 
    prompts_dict: Dict[int, str],
    top_k: int = 16,
    metric: str = "fitness" 
) -> pd.DataFrame:
    
    print("\n-------------------------------------------------------------------------")
    print("                    Calculating Aesthetic Scores (LAION)                 ")
    print(f"      (Filtered by Top-{top_k} {metric.upper()})")
    print("-------------------------------------------------------------------------")

    results = []

    for algo_key, algo_config in algorithm_dicts.items():
        print(f"Processing {algo_config['label']}...")
        stats = {"subset_avg": [], "subset_max": []}
        
        for seed in seeds:
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
                         from analysis_code.calculate_pickscore import calculate_dns_scores
                         descriptors = torch.load(desc_path, map_location="cpu", weights_only=False)
                         if isinstance(descriptors, np.ndarray): descriptors = torch.from_numpy(descriptors)
                         scores = calculate_dns_scores(fitnesses.numpy(), descriptors.numpy())
                         scores = torch.from_numpy(scores)
                    else:
                        scores = fitnesses
                else:
                    scores = fitnesses
                
                # Sort Descending
                k = min(len(scores), top_k)
                sorted_indices = torch.argsort(scores, descending=True)
                top_indices = sorted_indices[:k]
                top_images = images[top_indices]
                
                # --- CALC AESTHETICS ---
                aes_scores = get_aesthetic_scores_batch(top_images)
                
                # Filter NaNs
                aes_scores = aes_scores[~np.isnan(aes_scores)]
                
                if len(aes_scores) > 0:
                    stats["subset_avg"].append(np.mean(aes_scores))
                    stats["subset_max"].append(np.max(aes_scores))

            except Exception as e:
                print(f"  Error {seed}: {e}")

        if stats["subset_avg"]:
            row = {
                "Algorithm": algo_config["label"],
                f"Aesthetic (Mean Top-{top_k} {metric})": np.mean(stats["subset_avg"]),
                f"Aesthetic (Max Top-{top_k} {metric})": np.mean(stats["subset_max"]),
                "Std Error": np.std(stats["subset_avg"]) / np.sqrt(len(stats["subset_avg"])),
            }
            results.append(row)
            print(f"  -> Mean Aesthetic: {row[f'Aesthetic (Mean Top-{top_k} {metric})']:.2f}")

    return pd.DataFrame(results)