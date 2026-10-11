import numpy as np
import torch
from ribs.emitters._emitter_base import EmitterBase
from typing import Dict, Optional

class GaussianDictEmitter(EmitterBase):
    """
    A GaussianEmitter for Dictionary-based solutions.
    """
    def __init__(
        self,
        archive,
        *,
        genotype_key: str,
        initial_solution: Dict[str, np.ndarray],
        sigma: float = 0.1,
        bounds = None,
        batch_size: int = 64,
        seed: int = None,
    ):
        self._solution_schema = archive.solution_dim 
        
        super().__init__(
            archive=archive,
            solution_dim=(),
            bounds=bounds,
            lower_bounds=None,
            upper_bounds=None,
        )
        
        self.genotype_key = genotype_key
        self.batch_size = batch_size
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

        if initial_solution is None:
            raise ValueError("initial_solution must be provided as the seed.")

        # --- Sanitize Seed: Convert Tensors to NumPy ---
        self._seed_sol = {}
        for k, v in initial_solution.items():
            if isinstance(v, torch.Tensor):
                # Convert to float32 to handle bfloat16 (numpy doesn't support bfloat16)
                self._seed_sol[k] = v.detach().cpu().float().numpy()
            else:
                # Ensure it's a numpy array
                self._seed_sol[k] = np.array(v)
        
        if self.genotype_key not in self._seed_sol:
            raise ValueError(f"genotype_key '{self.genotype_key}' not found in initial_solution.")

    def ask(self):
        """
        Creates solutions by adding Gaussian noise to the 'genotype_key'.
        """
        is_first_generation = False

        # 1. Select Parents
        if self.archive.empty:
            is_first_generation = True
            # --- GEN 0: Broadcasting the Seed ---
            parents = {}
            for k, v in self._seed_sol.items():
                # If input is (C, H, W), wrap to (1, C, H, W)
                if v.ndim == len(self._solution_schema[k]):
                    v = v[np.newaxis, ...]
                
                # Repeat to fill batch_size
                parents[k] = np.repeat(v, self.batch_size, axis=0)
        else:
            # --- GEN 1+: Sampling Elites (Already NumPy) ---
            parents = self.archive.sample_elites(self.batch_size)["solution"]

        # 2. Create Children
        children = {}

        for key, parent_batch in parents.items():
            if key == self.genotype_key:
                # --- Apply Noise ---
                # Now parent_batch is guaranteed to be NumPy, so .dtype works
                noise = self._rng.normal(
                    scale=self.sigma,
                    size=parent_batch.shape
                ).astype(parent_batch.dtype)
                
                # Force first item to be exact copy (0 noise) in Gen 0
                if is_first_generation:
                    noise[0] = 0.0
                
                children[key] = parent_batch + noise
            else:
                children[key] = parent_batch.copy()

        return self._clip_if_needed(children)

    def _clip_if_needed(self, solution_dict):
        if self.lower_bounds is None or self.upper_bounds is None:
            return solution_dict
            
        data = solution_dict[self.genotype_key]
        solution_dict[self.genotype_key] = np.clip(
            data, self.lower_bounds, self.upper_bounds
        )
        return solution_dict
    
    def tell(self, solution, objective, measures, add_info, **fields):
        pass