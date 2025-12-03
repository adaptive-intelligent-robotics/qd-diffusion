import numpy as np
import torch
from ribs.emitters._emitter_base import EmitterBase
from typing import Dict, Optional, Union, List

class ImageBreederEmitter(EmitterBase):
    """
    An emitter based on the crossover operators from the ImageBreeder paper:
    https://dl.acm.org/doi/pdf/10.1145/3712256.3726439

    It performs pairwise crossover on Latents.
    Embeddings (Prompts) are kept constant (inherited from Parent 1), as the paper 
    only describes mutating the image/latent representations.

    Supported Mutation Types (mapped to paper functions):
    - 'latent-blend': _basic_latent_blend_crossover
    - 'latent-blend-with-score': _fitness_latent_blend_crossover
    - 'channel-cut': _channel_cut_crossover
    - 'horizontal-cut': _horizontal_cut_crossover
    - 'vertical-cut': _vertical_cut_crossover
    - 'mixed-cut': _random_cut_crossover
    - 'mixed-all': _any_crossover (excluding pixel ops)
    """
    def __init__(
        self,
        archive,
        *,
        initial_solution: Dict[str, np.ndarray],
        mutation_type: str = "latent-blend",
        batch_size: int = 64,
        seed: int = None,
        device: str = "cuda",
        bounds = None,
    ):
        self._solution_schema = archive.solution_dim 
        super().__init__(archive=archive, solution_dim=(), bounds=bounds)
        
        self.mutation_type = mutation_type
        self.batch_size = batch_size
        self._device = device
        self._rng = np.random.default_rng(seed)

        # Sanitize Seed
        if initial_solution is None:
            raise ValueError("initial_solution must be provided.")
        
        self._seed_sol = {}
        for k, v in initial_solution.items():
            if isinstance(v, torch.Tensor):
                self._seed_sol[k] = v.detach().cpu().numpy()
            else:
                self._seed_sol[k] = np.array(v)

        self._available_mutations = [
            "latent-blend", "latent-blend-with-score",
            "channel-cut", "horizontal-cut", "vertical-cut",
            "mixed-cut", "mixed-all"
        ]
        
        if self.mutation_type not in self._available_mutations:
            raise ValueError(f"Unknown mutation_type: {self.mutation_type}")

    # =========================================================================
    # PAPER OPERATORS (Adapted for NumPy Batches)
    # =========================================================================

    def _basic_latent_blend_crossover(self, latent1, latent2, blend_alpha=0.5):
        """
        Corresponds to: basic_latent_blend_crossover
        latentchild = (1 - alpha) * latent1 + alpha * latent2
        """
        return (1 - blend_alpha) * latent1 + blend_alpha * latent2

    def _fitness_latent_blend_crossover(self, latent1, latent2, score1, score2, _):
        """
        Corresponds to: fitness_latent_blend_crossover
        Calculates alpha based on fitness scores.
        """
        # Logic: score1 = individ1[1] if individ1[1] > 0 else 0
        s1 = max(0, score1)
        s2 = max(0, score2)
        
        if s1 + s2 == 0:
            blend_alpha = 0.5
        else:
            blend_alpha = s2 / (s1 + s2)
            
        return self._basic_latent_blend_crossover(latent1, latent2, blend_alpha)

    def _channel_cut_crossover(self, latent1, latent2, crossover_point=0.5):
        """
        Corresponds to: channel_cut_crossover
        Cuts latent on channel dim (axis 1 in NCHW, but here inputs are CHW or NCHW).
        Assuming input is (C, H, W).
        """
        if crossover_point >= 0.75:
            channel_cut = 3
        elif crossover_point <= 0.25:
            channel_cut = 1
        else:
            channel_cut = 2

        # Create offspring candidates
        # latent1[:cut] + latent2[cut:]
        offspring1 = np.concatenate((latent1[:channel_cut], latent2[channel_cut:]), axis=0)
        # latent2[:cut] + latent1[cut:]
        offspring2 = np.concatenate((latent2[:channel_cut], latent1[channel_cut:]), axis=0)

        # "Randomly selects one of the two resulting outcomes"
        return offspring1 if self._rng.random() < 0.5 else offspring2

    def _horizontal_cut_crossover(self, latent1, latent2, crossover_point=0.5):
        """
        Corresponds to: horizontal_cut_crossover
        Cuts on Height (axis 1 in CHW).
        """
        # cut_point = int(crossover_point * 64)
        height = latent1.shape[1]
        cut_point = int(crossover_point * height)
        
        # Clamp for safety
        cut_point = max(1, min(cut_point, height - 1))

        offspring1 = np.concatenate((latent1[:, :cut_point, :], latent2[:, cut_point:, :]), axis=1)
        offspring2 = np.concatenate((latent2[:, :cut_point, :], latent1[:, cut_point:, :]), axis=1)

        return offspring1 if self._rng.random() < 0.5 else offspring2

    def _vertical_cut_crossover(self, latent1, latent2, crossover_point=0.5):
        """
        Corresponds to: vertical_cut_crossover
        Cuts on Width (axis 2 in CHW).
        """
        width = latent1.shape[2]
        cut_point = int(crossover_point * width)
        
        # Clamp for safety
        cut_point = max(1, min(cut_point, width - 1))

        offspring1 = np.concatenate((latent1[:, :, :cut_point], latent2[:, :, cut_point:]), axis=2)
        offspring2 = np.concatenate((latent2[:, :, :cut_point], latent1[:, :, cut_point:]), axis=2)

        return offspring1 if self._rng.random() < 0.5 else offspring2

    def _random_cut_crossover(self, latent1, latent2, crossover_point=0.5):
        """
        Corresponds to: random_cut_crossover
        Randomly chooses between channel, horizontal, vertical.
        """
        methods = [
            self._channel_cut_crossover,
            self._horizontal_cut_crossover,
            self._vertical_cut_crossover
        ]
        used_crossover = self._rng.choice(methods)
        return used_crossover(latent1, latent2, crossover_point)

    def _any_crossover(self, latent1, latent2, score1, score2, crossover_point=0.5):
        """
        Corresponds to: any_crossover
        Randomly chooses between all applicable methods.
        (Wraps methods to uniform signature)
        """
        # Helper wrappers to handle signature differences
        def wrap_basic(l1, l2, pt): return self._basic_latent_blend_crossover(l1, l2, pt)
        def wrap_fitness(l1, l2, pt): return self._fitness_latent_blend_crossover(l1, l2, score1, score2, pt)
        def wrap_channel(l1, l2, pt): return self._channel_cut_crossover(l1, l2, pt)
        def wrap_horiz(l1, l2, pt): return self._horizontal_cut_crossover(l1, l2, pt)
        def wrap_vert(l1, l2, pt): return self._vertical_cut_crossover(l1, l2, pt)

        methods = [wrap_basic, wrap_fitness, wrap_channel, wrap_horiz, wrap_vert]
        used_crossover = self._rng.choice(methods)
        
        return used_crossover(latent1, latent2, crossover_point)

    # =========================================================================
    # EMITTER LOGIC
    # =========================================================================

    def _get_parents(self):
        if self.archive.empty:
            # Check if seed has at least 2 items
            seed_batch_size = self._seed_sol["latent"].shape[0]
            
            p1_batch = {}
            p2_batch = {}
            
            if seed_batch_size >= 2:
                # We have 2+ seeds. Use Index 0 as P1, Index 1 as P2.
                for k, v in self._seed_sol.items():
                    # v is (Batch, C, H, W)
                    item0 = v[0] # (C, H, W)
                    item1 = v[1] # (C, H, W)
                    
                    # Expand to (1, C, H, W) then repeat to (BatchSize, ...)
                    if item0.ndim == len(self._solution_schema[k]):
                        item0 = item0[np.newaxis, ...]
                        item1 = item1[np.newaxis, ...]
                        
                    p1_batch[k] = np.repeat(item0, self.batch_size, axis=0)
                    p2_batch[k] = np.repeat(item1, self.batch_size, axis=0)
            else:
                # We only have 1 seed. P1 = Seed, P2 = Seed (Logic in ask() adds noise)
                for k, v in self._seed_sol.items():
                    item0 = v[0] if v.shape[0] == 1 else v
                    if item0.ndim == len(self._solution_schema[k]):
                        item0 = item0[np.newaxis, ...]
                    
                    p1_batch[k] = np.repeat(item0, self.batch_size, axis=0)
                    p2_batch[k] = p1_batch[k].copy() # Identical parent

            return p1_batch, p2_batch, None, None

        elites = self.archive.sample_elites(self.batch_size * 2)
        p1, p2 = {}, {}
        
        for k, v in elites["solution"].items():
            p1[k] = v[:self.batch_size]
            p2[k] = v[self.batch_size:]
            
        objs = elites["objective"]
        obj1 = objs[:self.batch_size]
        obj2 = objs[self.batch_size:]
        
        return p1, p2, obj1, obj2

    def ask(self):
        p1, p2, fit1, fit2 = self._get_parents()
        
        # Handle Gen 0 (Empty Archive) -> Stochastic Noise
        if self.archive.empty:
            batch = {}
            for k, v in p1.items():
                if k == "latent":
                    noise = self._rng.standard_normal(v.shape, dtype=v.dtype) * 0.05
                    batch[k] = v + noise
                else:
                    batch[k] = v
            return batch

        # Prepare Random Split Points (Crossover Points)
        # "expect ... a random split point"
        crossover_points = self._rng.random(size=self.batch_size).astype(np.float32)

        final_batch = {}
        
        # Initialize result arrays with same shape as P1
        for k in p1.keys():
            final_batch[k] = np.empty_like(p1[k])

        # Perform Crossover row-by-row
        for i in range(self.batch_size):
            # Extract scalar score (if available)
            s1 = fit1[i] if fit1 is not None else 0
            s2 = fit2[i] if fit2 is not None else 0
            pt = crossover_points[i]

            # 1. Handle LATENTS (Mutate)
            l1 = p1["latent"][i]
            l2 = p2["latent"][i]
            
            if self.mutation_type == "latent-blend":
                res_latent = self._basic_latent_blend_crossover(l1, l2, pt)
            elif self.mutation_type == "latent-blend-with-score":
                res_latent = self._fitness_latent_blend_crossover(l1, l2, s1, s2, pt)
            elif self.mutation_type == "channel-cut":
                res_latent = self._channel_cut_crossover(l1, l2, pt)
            elif self.mutation_type == "horizontal-cut":
                res_latent = self._horizontal_cut_crossover(l1, l2, pt)
            elif self.mutation_type == "vertical-cut":
                res_latent = self._vertical_cut_crossover(l1, l2, pt)
            elif self.mutation_type == "mixed-cut":
                res_latent = self._random_cut_crossover(l1, l2, pt)
            elif self.mutation_type == "mixed-all":
                res_latent = self._any_crossover(l1, l2, s1, s2, pt)
            else:
                raise ValueError(f"Unknown type: {self.mutation_type}")
            
            final_batch["latent"][i] = res_latent

            # 2. Handle PROMPTS (Keep Constant)
            # "paper only mutates the latent, not the prompt"
            for k in p1.keys():
                if k != "latent":
                    # Just inherit from Parent 1
                    final_batch[k][i] = p1[k][i]

        return final_batch

    def tell(self, solution, objective, measures, add_info, **fields):
        pass