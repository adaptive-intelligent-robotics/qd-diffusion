import os
import sys

# Mock tkinter module completely to prevent import errors in headless environments
class MockCallable:
    """A mock object that can be called, instantiated, and have attributes"""
    def __init__(self, name='mock'):
        self._name = name
    
    def __call__(self, *args, **kwargs):
        return MockCallable(f'{self._name}()')
    
    def __getattr__(self, name):
        return MockCallable(f'{self._name}.{name}')
    
    def __bool__(self):
        return False
    
    def __str__(self):
        return f'<Mock {self._name}>'
    
    def __mro_entries__(self, bases):
        # Return empty tuple to allow usage as a base class
        return (object,)
    
    def __iter__(self):
        # Return empty iterator
        return iter([])
    
    def __len__(self):
        return 0

class MockModule:
    """A flexible mock module that returns mock objects for any attribute access"""
    def __init__(self, name='mock'):
        self.__name__ = name
        self.__file__ = f'<mock {name}>'
        self.__path__ = []
    
    def __getattr__(self, name):
        return MockCallable(f'{self.__name__}.{name}')

# Install the mock before any imports
sys.modules['tkinter'] = MockModule('tkinter')
sys.modules['tkinter.simpledialog'] = MockModule('tkinter.simpledialog')
sys.modules['tkinter.filedialog'] = MockModule('tkinter.filedialog')
sys.modules['tkinter.messagebox'] = MockModule('tkinter.messagebox')

# Fix matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict
import shutil
import gzip
import builtins

# Copy the BPE vocabulary file to a temporary location
bpe_source = '/workspace/src/analysis_code/hpsv2_data/bpe_simple_vocab_16e6.txt.gz'
bpe_temp_dir = '/tmp/hpsv2_patch'
os.makedirs(bpe_temp_dir, exist_ok=True)
bpe_temp = os.path.join(bpe_temp_dir, 'bpe_simple_vocab_16e6.txt.gz')

if os.path.exists(bpe_source) and not os.path.exists(bpe_temp):
    shutil.copy(bpe_source, bpe_temp)

# Monkey-patch builtins.open to redirect the specific file
_original_open = builtins.open

def patched_open(filename, *args, **kwargs):
    # If trying to open the missing file, redirect to our copy
    if isinstance(filename, str) and 'bpe_simple_vocab_16e6.txt.gz' in filename:
        if not os.path.exists(filename) and os.path.exists(bpe_temp):
            filename = bpe_temp
    return _original_open(filename, *args, **kwargs)

builtins.open = patched_open

import hpsv2

# Keep the patch active - don't restore

def get_hpsv2_scores_batch(prompt: str, image_tensors: torch.Tensor, batch_size=16) -> np.ndarray:
    """
    Calculates HPS v2.1 scores.
    """
    all_scores = []
    num_images = image_tensors.shape[0]

    for i in range(0, num_images, batch_size):
        batch_tensors = image_tensors[i : i + batch_size]
        
        pil_images = []
        for img in batch_tensors:
            img = img.detach().cpu().float()
            if img.max() > 1.0: img /= 255.0
            img = (img - img.min()) / (img.max() - img.min())
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
        
        # HPSv2 handles the model loading internally/lazily
        # It expects a list of images and a single prompt
        try:
            # Result is a list of floats
            scores = hpsv2.score(pil_images, prompt, hps_version="v2.1")
            all_scores.extend(scores)
        except Exception as e:
            print(f"  HPSv2 Error batch {i}: {e}")
            # Pad with NaNs if it fails
            all_scores.extend([np.nan] * len(pil_images))

    return np.array(all_scores)

# --- REUSED HELPER ---
def get_experiment_subdir(base_path: str) -> str:
    if not os.path.exists(base_path): return None
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs: return None
    subdirs.sort() 
    return os.path.join(base_path, subdirs[-1])

def analyze_hpsv2(
    root_dir: str, 
    algorithm_dicts: Dict, 
    seeds: range, 
    prompts_dict: Dict[int, str],
    top_k: int = 16,
    metric: str = "fitness"
) -> pd.DataFrame:
    
    print("\n-------------------------------------------------------------------------")
    print("                    Calculating HPS v2.1 Scores                          ")
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

            if not os.path.exists(img_path) or not os.path.exists(fit_path): continue
            
            try:
                images = torch.load(img_path, map_location="cpu", weights_only=False)
                fitnesses = torch.load(fit_path, map_location="cpu", weights_only=False)
                
                if isinstance(images, np.ndarray): images = torch.from_numpy(images)
                if isinstance(fitnesses, np.ndarray): fitnesses = torch.from_numpy(fitnesses)
                if len(images) == 0: continue

                # Sort Logic
                scores_for_sort = fitnesses
                k = min(len(scores_for_sort), top_k)
                sorted_indices = torch.argsort(scores_for_sort, descending=True)
                top_indices = sorted_indices[:k]
                top_images = images[top_indices]
                
                # --- CALC HPSv2 ---
                hps_scores = get_hpsv2_scores_batch(current_prompt, top_images)
                
                # Filter NaNs if any failed
                hps_scores = hps_scores[~np.isnan(hps_scores)]
                
                if len(hps_scores) > 0:
                    stats["subset_avg"].append(np.mean(hps_scores))
                    stats["subset_max"].append(np.max(hps_scores))

            except Exception as e:
                print(f"  Error {seed}: {e}")

        if stats["subset_avg"]:
            row = {
                "Algorithm": algo_config["label"],
                f"HPSv2 (Mean Top-{top_k})": np.mean(stats["subset_avg"]),
                f"HPSv2 (Max Top-{top_k})": np.mean(stats["subset_max"]),
                "Std Error": np.std(stats["subset_avg"]) / np.sqrt(len(stats["subset_avg"])),
            }
            results.append(row)
            print(f"  -> Mean HPSv2: {row[f'HPSv2 (Mean Top-{top_k})']:.2f}")

    return pd.DataFrame(results)