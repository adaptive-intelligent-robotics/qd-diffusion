import numpy as np
import torch
from ribs.emitters._emitter_base import EmitterBase

class DiverseDiffusionMaxEmitter(EmitterBase):
    """
    Implements Algorithm 2 (Max setting) from the Diverse Diffusion paper.
    Generates a batch of latents where each member is chosen to maximize 
    the distance from all other members in the current batch.
    """
    def __init__(
        self,
        archive,
        *,
        initial_solution: dict,
        n_max: int = 10,  # Number of candidates to check for each batch slot
        batch_size: int = 64,
        seed: int = None,
        bounds = None,
    ):
        self._solution_schema = archive.solution_dim 
        super().__init__(
            archive=archive,
            solution_dim=(),
            bounds=bounds,
            lower_bounds=None,
            upper_bounds=None,
        )
        
        self.batch_size = batch_size
        self.n_max = n_max
        self._rng = np.random.default_rng(seed)
        
        # Store initial solution to get the correct shapes/prompt embeddings
        self._seed_sol = {}
        for k, v in initial_solution.items():
            if isinstance(v, torch.Tensor):
                self._seed_sol[k] = v.detach().cpu().float().numpy()
            else:
                self._seed_sol[k] = np.array(v)

    def _get_min_dist(self, candidate, existing_set):
        """Calculates the minimum L2 distance from candidate to any vector in existing_set."""
        if len(existing_set) == 0:
            return 0.0
        # Flatten for distance calculation: (C, H, W) -> (D,)
        c_flat = candidate.ravel()
        # existing_set is List of (C, H, W)
        dists = [np.linalg.norm(c_flat - e.ravel()) for e in existing_set]
        return min(dists)

    def ask(self):
        # 1. Start with the template from our seed (prompts, etc.)
        # In a 1-step Turbo setting, we usually keep the prompt constant 
        # for the whole batch.
        final_batch = {}
        for k, v in self._seed_sol.items():
            # Ensure template is repeated for the batch
            if v.ndim == len(self._solution_schema[k]):
                v = v[np.newaxis, ...]
            final_batch[k] = np.repeat(v, self.batch_size, axis=0)

        # 2. Diverse Latent Generation (Algorithm 2)
        latent_shape = self._seed_sol["latent"].shape
        if latent_shape[0] == 1: # remove batch dim if present in seed
             latent_shape = latent_shape[1:]
             
        diverse_latents = []

        for b in range(self.batch_size):
            # Algorithm 2, Line 3: Initial farthest candidate
            v_farthest = self._rng.standard_normal(latent_shape).astype(np.float32)
            
            # Algorithm 2, Line 4: Iteratively find a 'farther' vector
            for n in range(self.n_max - 1):
                v_new = self._rng.standard_normal(latent_shape).astype(np.float32)
                
                if self._get_min_dist(v_new, diverse_latents) > \
                   self._get_min_dist(v_farthest, diverse_latents):
                    v_farthest = v_new
            
            diverse_latents.append(v_farthest)

        # 3. Update the batch with our optimized latents
        final_batch["latent"] = np.stack(diverse_latents)

        return final_batch

    def tell(self, solution, objective, measures, add_info, **fields):
        pass