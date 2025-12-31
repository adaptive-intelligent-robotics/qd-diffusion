
import csv
import itertools
import numpy as np

from scipy.spatial.distance import pdist
from skimage.metrics import structural_similarity as ssim
from typing import Dict, List


class CSVLogger:
    """Logger to save metrics of an experiment in a csv file
    during the training process.
    """

    def __init__(self, filename: str, header: List) -> None:
        """Create the csv logger, create a file and write the
        header.

        Args:
            filename: path to which the file will be saved.
            header: header of the csv file.
        """
        self._filename = filename
        self._header = header
        with open(self._filename, "w") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write the header
            writer.writeheader()

    def log(self, metrics: Dict[str, float]) -> None:
        """Log new metrics to the csv file.

        Args:
            metrics: A dictionary containing the metrics that
                need to be saved.
        """
        with open(self._filename, "a") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write new metrics in a raw
            writer.writerow(metrics)
            

def me_metrics_function(
    repertoire,
    iteration,
    config,
):
    """
    Calculate MAP-Elites metrics including Diversity and SSIM.
    
    Updates:
    - Calculates Descriptor Distance/Variance for both Whole Archive AND Top-K.
    - Calculates SSIM ONLY on the Top-K solutions.
    """
    
    # 1. Basic Setup
    current_metrics = {}
    current_metrics["iteration"] = iteration
    
    if config.emitter.type == "adam":
        if config.emitter.genotype != "multi":
            current_metrics["evaluations"] = iteration
        else:
            if config.model.name == "sd-1-5":
                current_metrics["evaluations"] = iteration * 2
            elif config.model.name == "sdxl-turbo":
                current_metrics["evaluations"] = iteration * 3
    else:
        current_metrics["evaluations"] = iteration * config.batch_size
    
    # 2. Extract Data & Handle Empty Archive
    data = repertoire.data()
    num_elites = repertoire.stats.num_elites
    
    # Need at least 2 elites for pairwise calc
    if num_elites < 2:
        # Zero out metrics if archive is empty/tiny
        keys = ["qd_score", "max_fitness", "coverage", "avg_pairwise_dist", "total_variance", 
                f"top_{config.top_k}_avg_dissimilarity_ssim", f"top_{config.top_k}_avg_pairwise_dist", 
                f"top_{config.top_k}_total_variance", f"top_{config.top_k}_fitness_avg"]
        for k in keys: current_metrics[k] = 0.0
        return current_metrics
    
    if config.algorithm.name == "qdhf":
        
        objectives = data["solution"]["dns_fitness"]
        descriptors = data["solution"]["dns_descriptor"]
        current_metrics["qd_score"] = np.sum(objectives)
        current_metrics["avg_fitness"] = np.mean(objectives)
        current_metrics["max_fitness"] = np.max(objectives)
        
    else:
        descriptors = data["measures"]
        objectives = data["objective"]
        
        current_metrics["qd_score"] = repertoire.stats.qd_score
        current_metrics["avg_fitness"] = repertoire.stats.obj_mean
        current_metrics["max_fitness"] = repertoire.stats.obj_max

    current_metrics["coverage"] = repertoire.stats.coverage

    # --- 4. Identify Top-K Elites ---
    k = config.top_k
    sorted_indices = np.argsort(objectives)
    top_k_indices = sorted_indices[-k:] 
    
    top_k_descriptors = descriptors[top_k_indices]
    top_k_objectives = objectives[top_k_indices]

    # --- 5. Descriptor Space Metrics ---
    try:
        dists_all = pdist(descriptors, metric='cosine')
        current_metrics["avg_pairwise_dist"] = np.nanmean(dists_all)
    except ValueError:
        current_metrics["avg_pairwise_dist"] = 0.0 # Handle edge cases
        
    current_metrics["total_variance"] = np.sum(np.var(descriptors, axis=0))

    # B) Top-K Subset
    # Note: If num_elites < 2 (handled above), we don't get here.
    # But if 2 <= num_elites < k, top_k metrics == whole archive metrics.
    if len(top_k_descriptors) >= 2:
        dists_top_k = pdist(top_k_descriptors, metric='cosine')
        current_metrics[f"top_{k}_avg_pairwise_dist"] = np.mean(dists_top_k)
        current_metrics[f"top_{k}_total_variance"] = np.sum(np.var(top_k_descriptors, axis=0))
    else:
        current_metrics[f"top_{k}_avg_pairwise_dist"] = 0.0
        current_metrics[f"top_{k}_total_variance"] = 0.0

    current_metrics[f"top_{k}_fitness_avg"] = np.mean(top_k_objectives)

    # --- 5. Image Space Metrics (SSIM on Top-K only) ---
    
    # Check if images exist
    has_images = False
    images = None
    
    # Robust extraction depending on PyRibs version/config
    if isinstance(data.get("solution"), dict) and "image" in data["solution"]:
        has_images = True
        # Extract only the top K images
        images = data["solution"]["image"][top_k_indices] 
    elif "image" in data: 
        has_images = True
        images = data["image"][top_k_indices]

    if has_images and len(images) >= 2:
        # We do FULL pairwise comparison on the Top-K subset.
        # Since K is usually small (e.g., 10 or 50), N^2 is negligible here.
        num_top = len(images)
        pairs_to_compute = itertools.combinations(range(num_top), 2)
        
        ssim_scores = []
        for i, j in pairs_to_compute:
            score = ssim(
                images[i],
                images[j],
                channel_axis=0,  # (C, H, W)
                data_range=2.0   # Assumes images are in range [-1, 1]
            )
            ssim_scores.append(score)

        if ssim_scores:
            avg_ssim = np.mean(ssim_scores)
            # Dissimilarity: -1.0 (Opposite) to 1.0 (Identical)
            # We negate so Higher = More Diverse
            current_metrics[f"top_{k}_avg_dissimilarity_ssim"] = -avg_ssim
        else:
            current_metrics[f"top_{k}_avg_dissimilarity_ssim"] = 0.0
    else:
        current_metrics[f"top_{k}_avg_dissimilarity_ssim"] = 0.0
    return current_metrics