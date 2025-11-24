
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
    ssim_sample_size=500  # Set to -1 for full calculation
):
    """
    Calculate MAP-Elites metrics including Diversity and SSIM.
    
    Args:
        repertoire: The archive.
        iteration: Current iteration number.
        config: Configuration object.
        ssim_sample_size (int or -1): Number of pairs to sample for SSIM calculation.
                                        If -1, calculates all pairs (O(N^2)).
    """
    
    # 1. Basic Setup
    current_metrics = {}
    current_metrics["iteration"] = iteration
    current_metrics["evaluations"] = (iteration + 1) * config.batch_size
    
    # Archive Stats
    current_metrics["coverage"] = repertoire.stats.coverage
    current_metrics["qd_score"] = repertoire.stats.qd_score
    current_metrics["max_fitness"] = repertoire.stats.obj_max

    # 2. Extract Data
    # We need raw arrays for pairwise comparisons
    data = repertoire.data()
    num_elites = repertoire.stats.num_elites
    
    # Safety check: Need at least 2 elites to calculate pairwise metrics
    if num_elites < 2:
        current_metrics["avg_pairwise_dist"] = 0.0
        current_metrics["total_variance"] = 0.0
        current_metrics["avg_dissimilarity_ssim"] = 0.0
        current_metrics["top_10_fitness_avg"] = 0.0
        return current_metrics

    # Access arrays
    descriptors = data["measures"]     # (N, measure_dim)
    objectives = data["objective"]     # (N,)

    # --- 3. Latent Space Metrics ---
    
    # A) Average Pairwise Distance (Euclidean)
    # pdist returns a condensed distance matrix (upper triangle)
    dists = pdist(descriptors, metric='euclidean')
    current_metrics["avg_pairwise_dist"] = np.mean(dists)

    # B) Total Variance
    # Sum of variances along each dimension
    variances = np.var(descriptors, axis=0)
    current_metrics["total_variance"] = np.sum(variances)

    # --- 4. Image Space Metrics (SSIM) ---
    
    # Check if images exist in the solution dictionary
    has_images = False
    if isinstance(data["solution"], dict) and "image" in data["solution"]:
        has_images = True
        images = data["solution"]["image"]
    elif "image" in data: # Fallback if data() structure changes
        has_images = True
        images = data["image"]

    if has_images:
        total_possible_pairs = num_elites * (num_elites - 1) // 2
        
        # Determine pairs to compare
        pairs_to_compute = []
        
        if ssim_sample_size != -1 and total_possible_pairs > ssim_sample_size:
            # --- RANDOM SAMPLING ---
            rng = np.random.default_rng()
            # Generate random indices, repeat until we have enough valid pairs
            # (This is a simple way to get unique pairs without generating all combinations)
            while len(pairs_to_compute) < ssim_sample_size:
                remaining = ssim_sample_size - len(pairs_to_compute)
                idx1 = rng.integers(0, num_elites, size=remaining)
                idx2 = rng.integers(0, num_elites, size=remaining)
                
                for i, j in zip(idx1, idx2):
                    if i != j:
                        pairs_to_compute.append((i, j))
        else:
            # --- FULL PAIRWISE ---
            pairs_to_compute = itertools.combinations(range(num_elites), 2)

        # Calculate SSIM
        ssim_scores = []
        for i, j in pairs_to_compute:
            score = ssim(
                images[i],
                images[j],
                channel_axis=0,  # (C, H, W)
                data_range=2.0   # Images are [-1, 1]
            )
            ssim_scores.append(score)

        if ssim_scores:
            avg_ssim = np.mean(ssim_scores)
            # We want Dissimilarity: Negate the SSIM (1.0 is identical, -1.0 is opposite)
            # Result: Lower is identical, Higher is diverse
            current_metrics["avg_dissimilarity_ssim"] = -avg_ssim
        else:
            current_metrics["avg_dissimilarity_ssim"] = 0.0
    else:
        current_metrics["avg_dissimilarity_ssim"] = 0.0

    # --- 5. Top-K Fitness ---
    k = 10
    if num_elites >= k:
        # Sort ascending, take last k
        top_k_objs = np.sort(objectives)[-k:]
        current_metrics[f"top_{k}_fitness_avg"] = np.mean(top_k_objs)
    else:
        current_metrics[f"top_{k}_fitness_avg"] = np.mean(objectives)

    return current_metrics