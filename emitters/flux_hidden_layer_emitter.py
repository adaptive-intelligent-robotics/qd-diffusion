import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from ribs.emitters._emitter_base import EmitterBase
from typing import Tuple, Dict, Optional, Union

# =============================================================================
# 1. FLUX CORE HELPERS
# =============================================================================

def get_flux_mu(pipe, latents):
    """Calculates density shift (mu) based on latent sequence length."""
    return calculate_shift(
        latents.shape[1],
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )

def _get_target_layer_flux(transformer, layer_name):
    """Fetches a specific submodule from the Flux Transformer."""
    target = transformer
    for part in layer_name.split("."):
        if part.isdigit():
            target = target[int(part)]
        else:
            target = getattr(target, part)
    return target

@torch.no_grad()
def flux_forward(pipe, latents, prompt_embeds, pooled_prompt_embeds, 
                 text_ids, latent_img_ids, num_steps, hook_idx, guidance_scale):
    """Rolls forward until hook_idx using manual Euler steps for reversibility."""
    device = latents.device
    curr = latents.clone()
    
    # Sync scheduler with correct mu for this resolution
    mu = get_flux_mu(pipe, curr)
    pipe.scheduler.set_timesteps(num_steps, device=device, mu=mu)
    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas
    
    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0]) if pipe.transformer.config.guidance_embeds else None

    for i in range(hook_idx):
        t = timesteps[i]
        with pipe.transformer.cache_context("cond"):
            v = pipe.transformer(
                hidden_states=curr,
                timestep=t.expand(curr.shape[0]).to(curr.dtype) / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_img_ids,
                return_dict=False,
            )[0]
        # Euler step: x = x + dt * v
        curr += (sigmas[i+1] - sigmas[i]) * v
        
    return curr, timesteps[hook_idx], sigmas, timesteps

@torch.no_grad()
def flux_invert(pipe, latents, prompt_embeds, pooled_prompt_embeds, 
                text_ids, latent_img_ids, sigmas, timesteps, hook_idx, guidance_scale):
    """Inverts the ODE from hook point back to noise space (T=0)."""
    device = latents.device
    curr = latents.clone()
    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0]) if pipe.transformer.config.guidance_embeds else None

    for i in range(hook_idx - 1, -1, -1):
        t = timesteps[i]
        with pipe.transformer.cache_context("cond"):
            v = pipe.transformer(
                hidden_states=curr,
                timestep=t.expand(curr.shape[0]).to(curr.dtype) / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_img_ids,
                return_dict=False,
            )[0]
        # Reverse Euler step: move back toward sigma=1.0
        curr += (sigmas[i] - sigmas[i+1]) * v
        
    return curr

# =============================================================================
# 2. SVD SENSITIVITY CALCULATION
# =============================================================================

def compute_flux_svd_basis(pipe, x_t, t_val, prompt_embeds, pooled_prompt_embeds,
                           text_ids, latent_img_ids, mutation_key, layer_name,
                           guidance_scale, k=5, iters=5, eps=1e-3, device="cuda"):
    """
    Approximates the top-k right singular vectors of d(hidden)/d(target) via a
    randomised range-finder (Halko et al. 2011). Uses only no_grad forward passes —
    no backward through the quantised transformer, so no bitsandbytes OOM.

    Algorithm:
      1. Sample n_sketch random input directions (Omega).
      2. Compute Y = J * Omega^T column-by-column via finite differences.
      3. SVD of Y gives left singular vectors U (in sketch space).
      4. Input-space basis = Omega^T @ U[:, :k]  (combinations of random dirs).
    """
    target_module = _get_target_layer_flux(pipe.transformer, layer_name)
    t = torch.as_tensor(t_val, device=device).view(1)

    l_base = x_t.detach()
    p_base = prompt_embeds.detach()
    pp_base = pooled_prompt_embeds.detach()

    def get_hidden(l, p, pp):
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(l.shape[0])
        feat = []
        def hook(m, i, o): feat.append(o[0] if isinstance(o, tuple) else o)
        h = target_module.register_forward_hook(hook)
        try:
            with pipe.transformer.cache_context("cond"):
                pipe.transformer(
                    hidden_states=l, timestep=t/1000, guidance=guidance,
                    pooled_projections=pp, encoder_hidden_states=p,
                    txt_ids=text_ids, img_ids=latent_img_ids, return_dict=False
                )
            return feat[0].reshape(-1)
        finally:
            h.remove()

    if mutation_key == "latent": target_init = l_base
    elif mutation_key == "prompt_embedding": target_init = p_base
    else: target_init = pp_base

    n_sketch = k * iters  # same forward-pass budget as the old power-iteration
    rng = torch.Generator(device=device).manual_seed(42)

    with torch.no_grad():
        y0 = get_hidden(l_base, p_base, pp_base)

        # Omega: (n_sketch x input_dim) — normalised random input directions
        Omega = torch.randn(n_sketch, target_init.numel(), device=device,
                            dtype=target_init.dtype, generator=rng)
        Omega = Omega / (Omega.norm(dim=1, keepdim=True) + 1e-8)

        # Y: (n_sketch x output_dim) — image of each random direction under J
        Y_rows = []
        for i in range(n_sketch):
            v_i = Omega[i].reshape(target_init.shape)
            l, p, pp = l_base.clone(), p_base.clone(), pp_base.clone()
            if mutation_key == "latent": l = l + eps * v_i
            elif mutation_key == "prompt_embedding": p = p + eps * v_i
            else: pp = pp + eps * v_i
            torch.cuda.empty_cache()
            Y_rows.append((get_hidden(l, p, pp) - y0) / eps)
            del l, p, pp, v_i

        Y = torch.stack(Y_rows, dim=0)  # (n_sketch x output_dim)
        del Y_rows

        # Left singular vectors of Y reveal which sketch directions span
        # the dominant output subspace; project back to input space via Omega.
        U, _, _ = torch.linalg.svd(Y.float(), full_matrices=False)  # svd requires fp32
        del Y

        basis = []
        for i in range(k):
            b = (U[:, i].to(Omega.dtype) @ Omega).reshape(target_init.shape)
            basis.append(b / (b.norm() + 1e-8))

    return basis

# =============================================================================
# 3. FLUX HIDDEN LAYER EMITTER
# =============================================================================
            
class FluxHiddenLayerEmitter(EmitterBase):
    """
    Emitter for Flux-1 pipelines in a QD loop.
    
    Supports randomized hook points to explore multiple levels of the 
    Transformer hierarchy (structural vs. detail mutations).
    """
    def __init__(
        self, archive, initial_solution, pipe, text_ids, latent_img_ids,
        mutation_key: str = "latent", 
        mutation_type: str = "svd", 
        batch_size: int = 4,
        seed: int = None,
        sigma: float = 0.1,
        alpha_min: float = -5.0,
        alpha_max: float = 5.0,
        num_inference_steps: int = 50,
        hook_proportion: Union[float, Tuple[float, float]] = (0.1, 0.4),
        svd_layer: str = "transformer_blocks.5",
        svd_k: int = 5,
        svd_iters: int = 5,
        svd_fd_eps: float = 1e-1,
        guidance_scale: float = 3.5,
        device: str = "cuda"
    ):
        super().__init__(
                    archive=archive, 
                    solution_dim=(), 
                    bounds=None, 
                    lower_bounds=None,
                    upper_bounds=None
                )
        self._seed_sol = initial_solution
        self.pipe = pipe
        self.text_ids = text_ids
        self.latent_img_ids = latent_img_ids
        
        self.mutation_key = mutation_key
        self.mutation_type = mutation_type
        self.batch_size = batch_size
        self.sigma = sigma
        self.alpha_min, self.alpha_max = alpha_min, alpha_max
        self.num_steps = num_inference_steps
        self.hook_prop = hook_proportion
        self.svd_layer, self.svd_k, self.svd_iters, self.svd_fd_eps = svd_layer, svd_k, svd_iters, svd_fd_eps
        self.guidance_scale, self.device = guidance_scale, device
        
        self._rng = np.random.default_rng(seed)

    def ask(self):
        # 1. Parent Selection and Standardization
        if self.archive.empty:
            # Seed solution is CPU-bound; ensure it's on device and correct dtype
            parent_sol = {k: v.to(self.device) for k, v in self._seed_sol.items()}
        else:
            # Sample from archive (numpy float32); skip "image" — not needed for mutation
            sampled = self.archive.sample_elites(1)["solution"]
            _mutation_keys = {"latent", "prompt_embedding", "pooled_prompt_embedding"}
            parent_sol = {
                k: torch.from_numpy(v[0]).to(device=self.device, dtype=self.pipe.transformer.dtype)
                for k, v in sampled.items() if k in _mutation_keys
            }

        # Ensure Parent shapes are [1, Seq, Dim] for the Transformer
        p_latent = parent_sol["latent"]
        if p_latent.ndim == 2: p_latent = p_latent.unsqueeze(0)
        
        p_embeds = parent_sol["prompt_embedding"]
        if p_embeds.ndim == 2: p_embeds = p_embeds.unsqueeze(0)
        
        p_pooled = parent_sol["pooled_prompt_embedding"]
        if p_pooled.ndim == 1: p_pooled = p_pooled.unsqueeze(0)

        # 2. Randomize Hook Point
        prop = self._rng.uniform(*self.hook_prop) if isinstance(self.hook_prop, tuple) else self.hook_prop
        hook_idx = int(prop * self.num_steps)

        # 3. Forward Roll (Parent Only)
        # This gives us x_t at the hook point for inversion later
        x_t, t_val, sigmas, timesteps = flux_forward(
            self.pipe, 
            p_latent,
            p_embeds,
            p_pooled,
            self.text_ids, 
            self.latent_img_ids,
            self.num_steps, 
            hook_idx, 
            self.guidance_scale
        )

        # 4. SVD Calculation — all no_grad forward passes, no backward needed
        basis = None
        if self.mutation_type == "svd":
            was_cp = self.pipe.transformer.is_gradient_checkpointing
            if was_cp:
                self.pipe.transformer.disable_gradient_checkpointing()
            torch.cuda.empty_cache()
            try:
                basis = compute_flux_svd_basis(
                    self.pipe, x_t, t_val, p_embeds, p_pooled,
                    self.text_ids, self.latent_img_ids,
                    self.mutation_key, self.svd_layer, self.guidance_scale,
                    k=self.svd_k, iters=self.svd_iters, eps=self.svd_fd_eps, device=self.device
                )
            finally:
                if was_cp:
                    self.pipe.transformer.enable_gradient_checkpointing()

        # 5. Sequential Generation of the Mutated Batch
        new_batch = {"latent": [], "prompt_embedding": [], "pooled_prompt_embedding": []}
        
        def prepare_for_archive(t):
            """Ensures tensor is [Seq, Dim] and float32 for archive storage."""
            arr = t.detach().cpu().float().numpy()
            if arr.ndim > 0 and arr.shape[0] == 1:
                return np.squeeze(arr, axis=0)
            return arr

        for i in range(self.batch_size):
            # Gen 0 Preservation Logic
            if self.archive.empty and i == 0:
                for k in ["latent", "prompt_embedding", "pooled_prompt_embedding"]:
                    new_batch[k].append(prepare_for_archive(parent_sol[k]))
                continue

            alpha = self._rng.uniform(self.alpha_min, self.alpha_max)
            
            # Select Direction (SVD Basis or Random Gaussian)
            if self.mutation_type == "svd":
                direction_idx = self._rng.integers(len(basis))
                direction = basis[direction_idx]
            else:
                # Direction matches the shape of the mutation target
                target_shape = x_t.shape if self.mutation_key == "latent" else p_embeds.shape if self.mutation_key == "prompt_embedding" else p_pooled.shape
                direction = torch.randn(target_shape, device=self.device, dtype=self.pipe.transformer.dtype)
                direction_idx = None

            # Local copies for this batch element
            curr_p = p_embeds.clone()
            curr_pp = p_pooled.clone()

            if self.mutation_key == "latent":
                # ODE Inversion Mutation
                x_t_star = x_t + alpha * (direction if self.mutation_type == "svd" else self.sigma * direction)
                mut_latent = flux_invert(
                    self.pipe, x_t_star, curr_p, curr_pp, 
                    self.text_ids, self.latent_img_ids, 
                    sigmas, timesteps, hook_idx, self.guidance_scale
                )
                new_batch["latent"].append(prepare_for_archive(mut_latent))
                new_batch["prompt_embedding"].append(prepare_for_archive(curr_p))
                new_batch["pooled_prompt_embedding"].append(prepare_for_archive(curr_pp))
            
            else:
                # Embedding Mutations (No inversion needed)
                if self.mutation_key == "prompt_embedding":
                    curr_p = curr_p + alpha * (direction if self.mutation_type == "svd" else self.sigma * direction)
                elif self.mutation_key == "pooled_prompt_embedding":
                    curr_pp = curr_pp + alpha * (direction if self.mutation_type == "svd" else self.sigma * direction)
                
                new_batch["latent"].append(prepare_for_archive(p_latent))
                new_batch["prompt_embedding"].append(prepare_for_archive(curr_p))
                new_batch["pooled_prompt_embedding"].append(prepare_for_archive(curr_pp))

            # Cleanup VRAM after each batch member to keep peak usage low
            torch.cuda.empty_cache()

        # Returns {key: np.array([Batch, Seq, Dim])}
        return {k: np.stack(v) for k, v in new_batch.items()}

    def tell(self, solution, objective, measures, add_info, **fields):
        pass