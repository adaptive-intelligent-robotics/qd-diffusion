import numpy as np
import torch
from ribs.emitters._emitter_base import EmitterBase
from typing import Dict, Optional, Union

class IsoLineDictEmitter(EmitterBase):
    """
    An IsoLineEmitter that works with Dictionary-based solutions.
    
    It leverages correlations between existing elites. To generate a new solution,
    it selects a pair of elites x_i and x_j and samples from:
    
        x_new = x_i + sigma_iso * N(0, I) + sigma_line * (x_j - x_i) * N(0, 1)
        
    Args:
        archive: The archive object.
        genotype_key (str): The key in the solution dict to mutate (e.g., "latent").
        initial_solution (dict): The seed solution.
        iso_sigma (float): Scale factor for isotropic Gaussian noise.
        line_sigma (float): Scale factor for the line distribution (direction).
        bounds: Bounds for the solution space.
        batch_size (int): Number of solutions to return.
        seed (int): Random seed.
    """
    def __init__(
        self,
        archive,
        *,
        genotype_key: str,
        initial_solution: Dict[str, Union[np.ndarray, torch.Tensor]],
        iso_sigma: float = 0.01,
        line_sigma: float = 0.2,
        bounds = None,
        batch_size: int = 64,
        seed: int = None,
    ):
        self._solution_schema = archive.solution_dim 
        
        super().__init__(
            archive=archive,
            solution_dim=(),
            bounds=bounds,
            upper_bounds=None,
            lower_bounds=None,
        )
        
        self.genotype_key = genotype_key
        self.batch_size = batch_size
        self.iso_sigma = iso_sigma
        self.line_sigma = line_sigma
        self._rng = np.random.default_rng(seed)

        if initial_solution is None:
            raise ValueError("initial_solution must be provided.")

        # --- Sanitize Seed: Convert Tensors to NumPy ---
        self._seed_sol = {}
        for k, v in initial_solution.items():
            if isinstance(v, torch.Tensor):
                self._seed_sol[k] = v.detach().cpu().numpy()
            else:
                self._seed_sol[k] = np.array(v)
        
        if self.genotype_key not in self._seed_sol:
            raise ValueError(f"genotype_key '{self.genotype_key}' not found in initial_solution.")

    def ask(self):
        """
        Generates solutions using IsoLine logic on the genotype_key.
        """
        # 1. Select Parents (2x Batch Size needed for pairs)
        if self.archive.empty:
            # --- GEN 0: Broadcasting the Seed ---
            # P1 and P2 will be identical, so direction (P2 - P1) will be 0.
            # This reduces to pure Gaussian noise (iso_sigma) for the first Gen.
            parents_flat = {}
            for k, v in self._seed_sol.items():
                if v.ndim == len(self._solution_schema[k]):
                    v = v[np.newaxis, ...]
                # Repeat 2x batch size to simulate having selected pairs
                parents_flat[k] = np.repeat(v, 2 * self.batch_size, axis=0)
        else:
            # --- GEN 1+: Sampling Elites ---
            # Sample 2 * batch_size to get pairs
            parents_flat = self.archive.sample_elites(2 * self.batch_size)["solution"]

        # 2. Separate into P1 (Elites) and P2 (Partners)
        # We need to construct the result batch.
        final_batch = {}
        
        # Extract Genotypes
        genotype_arr = parents_flat[self.genotype_key] # Shape: (2*B, ...)
        
        # Reshape to (2, B, ...) to separate pairs
        # We perform this reshaping logic on the 0-th dimension
        input_shape = genotype_arr.shape[1:] # e.g. (4, 64, 64)
        
        parents_reshaped = genotype_arr.reshape(2, self.batch_size, *input_shape)
        
        x_i = parents_reshaped[0] # Parent 1 (Batch, ...)
        x_j = parents_reshaped[1] # Parent 2 (Batch, ...)
        
        # 3. Apply IsoLine Math
        # Direction vector
        directions = x_j - x_i
        
        # Isotropic Noise: N(0, I) -> Shape matches genotype
        iso_noise = self._rng.normal(
            scale=self.iso_sigma,
            size=x_i.shape
        ).astype(x_i.dtype)
        
        # Line Noise: N(0, 1) -> Scalar per individual
        # Shape (Batch, 1)
        line_noise_scalar = self._rng.normal(
            scale=self.line_sigma,
            size=(self.batch_size, 1)
        ).astype(x_i.dtype)
        
        # Broadcast line_noise to match genotype dimensions
        # If genotype is (B, 4, 64, 64), line_noise needs to be (B, 1, 1, 1)
        view_shape = [self.batch_size] + [1] * len(input_shape)
        line_noise_broadcast = line_noise_scalar.reshape(view_shape)
        
        # Formula: x_i + iso + line * (x_j - x_i)
        new_genotypes = x_i + iso_noise + (line_noise_broadcast * directions)
        
        final_batch[self.genotype_key] = new_genotypes

        # 4. Handle Non-Genotype Keys (Copy from Parent 1)
        for k in parents_flat.keys():
            if k != self.genotype_key:
                # Extract 2*B array
                arr_flat = parents_flat[k]
                # Reshape to get Parent 1 only
                # (2, B, ...) -> Take index 0
                arr_reshaped = arr_flat.reshape(2, self.batch_size, *arr_flat.shape[1:])
                final_batch[k] = arr_reshaped[0]

        return self._clip_if_needed(final_batch)

    def _clip_if_needed(self, solution_dict):
        """Clips the genotype_key if bounds were provided."""
        if self.lower_bounds is None or self.upper_bounds is None:
            return solution_dict
            
        data = solution_dict[self.genotype_key]
        solution_dict[self.genotype_key] = np.clip(
            data, self.lower_bounds, self.upper_bounds
        )
        return solution_dict
    
    def tell(self, solution, objective, measures, add_info, **fields):
        pass