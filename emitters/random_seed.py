import numpy as np
import torch
from ribs.emitters._emitter_base import EmitterBase
from typing import Dict, Optional, Union

class RandomSeedEmitter(EmitterBase):
    """
    An emitter that generates purely random solutions at every generation.
    
    It ignores parent performance. 
    - For the 'latent_key', it generates fresh standard Gaussian noise.
    - For all other keys (e.g. embeddings), it repeats the data found in 'initial_solution'.
    
    Compatible with both SD-1.5 and SDXL Turbo data structures.
    """
    def __init__(
        self,
        archive,
        *,
        initial_solution: Dict[str, Union[np.ndarray, torch.Tensor]],
        latent_key: str = "latent",
        batch_size: int = 64,
        seed: int = None,
        bounds = None,
    ):
        """
        Args:
            archive: The archive object (used for solution_dim).
            initial_solution: A dictionary containing the shapes and fixed embeddings 
                              (e.g. prompt_embeddings) to use.
            latent_key: The dictionary key representing the noise/latent to randomize 
                        (default: "latent").
            batch_size: Number of solutions to emit per ask().
            seed: Random seed.
        """
        self._solution_schema = archive.solution_dim 
        
        # Initialize Base (pass empty tuple to bypass standard checks)
        super().__init__(
            archive=archive,
            solution_dim=(),
            bounds=bounds,
            lower_bounds=None,
            upper_bounds=None,
        )
        
        self.latent_key = latent_key
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)

        # --- Sanitize Initial Solution (Tensor -> Numpy) ---
        if initial_solution is None:
            raise ValueError("initial_solution must be provided to determine shapes and fixed embeddings.")

        self._seed_sol = {}
        for k, v in initial_solution.items():
            if isinstance(v, torch.Tensor):
                # Convert to float32 to handle bfloat16 (numpy doesn't support bfloat16)
                self._seed_sol[k] = v.detach().cpu().float().numpy()
            else:
                self._seed_sol[k] = np.array(v)
        
        # Validation
        if self.latent_key not in self._seed_sol:
            raise ValueError(f"latent_key '{self.latent_key}' not found in initial_solution.")
            
        # Store the shape of the latent (removing batch dim 1 if present)
        self._latent_shape = self._seed_sol[self.latent_key].shape
        if self._latent_shape[0] == 1 and len(self._latent_shape) > 1:
             self._latent_shape = self._latent_shape[1:]

    def ask(self):
        """
        Generates a batch of solutions.
        Latents are Random Noise. Embeddings are repeated from seed.
        """
        batch = {}
        
        # 1. Generate Random Latents
        # shape: (batch_size, channels, height, width)
        noise_shape = (self.batch_size, *self._latent_shape)
        
        batch[self.latent_key] = self._rng.standard_normal(
            size=noise_shape, 
            dtype=self._seed_sol[self.latent_key].dtype
        )
        
        # 2. Repeat Fixed Embeddings
        for key, val in self._seed_sol.items():
            if key == self.latent_key:
                continue
            
            # Ensure input is (1, ...) before repeating
            if val.shape[0] != 1:
                # If seed was provided as (C, ...), wrap to (1, C, ...)
                val_to_repeat = val[np.newaxis, ...]
            else:
                val_to_repeat = val
                
            batch[key] = np.repeat(val_to_repeat, self.batch_size, axis=0)
            
        return batch

    def tell(self, solution, objective, measures, add_info, **fields):
        # Random emitter does not learn from feedback
        pass