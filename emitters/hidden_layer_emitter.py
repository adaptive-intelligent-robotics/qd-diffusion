import numpy as np
import torch
from diffusers import DDIMScheduler
from diffusers.models import UNet2DConditionModel
from ribs.emitters._emitter_base import EmitterBase
from typing import Tuple, Dict, Optional

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def _get_target_layer(unet, layer_name):
    """Helper to fetch the specific submodule from the UNet."""
    if layer_name == "mid_block":
        return unet.mid_block
    
    # Handle nested layers e.g. "up_blocks.0"
    target = unet
    parts = layer_name.split(".")
    for part in parts:
        if part.isdigit():
            target = target[int(part)]
        else:
            target = getattr(target, part)
    return target

@torch.no_grad()
def roll_to_hook(x_T, prompt_embedding, *, unet, scheduler, guidance_scale, num_steps, hook_idx, device="cuda"):
    """Rolls the diffusion process forward from T to the hook timestep."""
    x  = x_T.to(device).clone()
    prompt_embedding = prompt_embedding.to(device)
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.to(device)

    for i, t in enumerate(timesteps):
        if i == hook_idx:
            # Return scalar t, index i, and full timesteps array
            return x.clone(), t.item(), i, timesteps  
        
        latent_model_input = torch.cat([x] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embedding).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        x = scheduler.step(noise_pred, t, x).prev_sample
        
    return x.clone(), timesteps[-1].item(), len(timesteps)-1, timesteps

@torch.no_grad()
def invert_to_start(
        start_latents,
        prompt_embedding,
        unet,
        scheduler,
        guidance_scale,
        timesteps,
        hook_idx,
        device="cuda"
    ):
    """Inverts the diffusion process from the hook point back to t=0 (Noise)."""
    scheduler.set_timesteps(len(timesteps), device=device)
    latents  = start_latents.clone()
    
    timesteps_to_reverse = reversed(timesteps[:hook_idx + 1])
    num_reverse_steps = len(timesteps_to_reverse)
    
    for i, t in enumerate(timesteps_to_reverse):
        if i >= num_reverse_steps - 1: continue
        
        latent_model_input = torch.cat([latents] * 2) 
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embedding).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        current_t = t
        next_t = timesteps_to_reverse[i + 1] if (i + 1) < num_reverse_steps else torch.tensor(-1)
        if next_t.item() == -1: break

        alpha_t = scheduler.alphas_cumprod[current_t]
        alpha_t_next = scheduler.alphas_cumprod[next_t]
        
        latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred

    return latents

# --- SVD Computations ---

def compute_latents_svd(
    x_t: torch.Tensor,
    t_scalar: int,
    *,
    unet,
    scheduler,
    prompt_embedding,
    guidance_scale=7.5,
    k=10,
    iters=20,
    fd_eps=1e-3,
    device="cuda",
    seed=None,
    layer="mid_block",
):
    """Computes d(Hidden)/d(Latents)."""
    xT = x_t.detach().to(device)
    t  = torch.tensor(t_scalar, device=device).expand(1)
    prompt_embedding = prompt_embedding.to(device)
    
    target_module = _get_target_layer(unet, layer)

    # 1. Forward Helper
    def forward_hidden(curr_x):
        # Prepare inputs (CFG duplication)
        uncond, cond = prompt_embedding[0:1], prompt_embedding[1:2]
        h_in = torch.cat([uncond, cond], dim=0)
        
        # Duplicate x for CFG
        x_in = torch.cat([curr_x, curr_x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        x_in = scheduler.scale_model_input(x_in, t[0])

        features = []
        def hook(m, i, o): features.append(o)
        h = target_module.register_forward_hook(hook)
        try:
            unet(x_in, t_in, encoder_hidden_states=h_in)
            feat = features[0]
            # Take Conditional half
            if feat.shape[0] >= 2:
                feat = feat[feat.shape[0]//2:]
            return feat.reshape(-1)
        finally:
            h.remove()

    # 2. Power Iteration
    # [FIX] Ensure seed is an integer
    safe_seed = int(seed) if seed is not None else 42
    rng = torch.Generator(device=device).manual_seed(safe_seed)
    vecs = []
    
    with torch.no_grad():
        y0 = forward_hidden(xT)
        
    for _ in range(k):
        v = torch.randn(xT.shape, device=device, dtype=xT.dtype, generator=rng)
        for p in vecs: v -= (v.flatten() @ p.flatten()) * p
        v = v / (v.norm() + 1e-8)

        for __ in range(iters):
            # Jacobian-Vector Product (Forward Diff)
            with torch.no_grad():
                y_pert = forward_hidden(xT + fd_eps * v)
            Jv = (y_pert - y0) / fd_eps
            
            # Vector-Jacobian Product (Autograd)
            x_req = xT.clone().requires_grad_(True)
            y_req = forward_hidden(x_req)
            loss = (y_req * Jv.detach()).sum()
            grad = torch.autograd.grad(loss, x_req, retain_graph=False)[0]
            
            v_new = grad
            for p in vecs: v_new -= (v_new.flatten() @ p.flatten()) * p
            v = (v_new / (v_new.norm() + 1e-8)).detach()
            
        vecs.append(v)
    return vecs

def compute_embedding_svd(
    x_t: torch.Tensor,
    t_scalar: int,
    *,
    unet,
    scheduler,
    prompt_embedding,
    guidance_scale=7.5,
    k=10,
    iters=20,
    fd_eps=1e-3,
    device="cuda",
    seed=None,
    layer="mid_block",
):
    """Computes d(Hidden)/d(PromptEmbedding)."""
    xT = x_t.detach().to(device)
    t  = torch.tensor(t_scalar, device=device).expand(1)
    
    # prompt_embedding is shape (2, 77, 768) -> [Uncond, Cond]
    # We only want to mutate Cond (index 1).
    full_embeds = prompt_embedding.detach().to(device)
    c_uncond = full_embeds[0:1]
    c_cond_init = full_embeds[1:2] # Target for SVD
    
    target_module = _get_target_layer(unet, layer)

    # 1. Forward Helper (varying cond embedding)
    def forward_hidden(curr_c_cond):
        h_in = torch.cat([c_uncond, curr_c_cond], dim=0)
        
        x_in = torch.cat([xT, xT], dim=0)
        t_in = torch.cat([t, t], dim=0)
        x_in = scheduler.scale_model_input(x_in, t[0])

        features = []
        def hook(m, i, o): features.append(o)
        h = target_module.register_forward_hook(hook)
        try:
            unet(x_in, t_in, encoder_hidden_states=h_in)
            feat = features[0]
            if feat.shape[0] >= 2:
                feat = feat[feat.shape[0]//2:]
            return feat.reshape(-1)
        finally:
            h.remove()

    # 2. Power Iteration
    # [FIX] Ensure seed is an integer
    safe_seed = int(seed) if seed is not None else 42
    rng = torch.Generator(device=device).manual_seed(safe_seed)
    vecs = []
    
    with torch.no_grad():
        y0 = forward_hidden(c_cond_init)
        
    for _ in range(k):
        v = torch.randn(c_cond_init.shape, device=device, dtype=c_cond_init.dtype, generator=rng)
        for p in vecs: v -= (v.flatten() @ p.flatten()) * p
        v = v / (v.norm() + 1e-8)

        for __ in range(iters):
            with torch.no_grad():
                y_pert = forward_hidden(c_cond_init + fd_eps * v)
            Jv = (y_pert - y0) / fd_eps
            
            c_req = c_cond_init.clone().requires_grad_(True)
            y_req = forward_hidden(c_req)
            loss = (y_req * Jv.detach()).sum()
            grad = torch.autograd.grad(loss, c_req, retain_graph=False)[0]
            
            v_new = grad
            for p in vecs: v_new -= (v_new.flatten() @ p.flatten()) * p
            v = (v_new / (v_new.norm() + 1e-8)).detach()
            
        vecs.append(v)
    return vecs


# =============================================================================
# 2. THE EMITTER CLASS
# =============================================================================

class SDHiddenLayerEmitter(EmitterBase):
    """
    Emits solutions for SD pipelines.
    
    Features:
    - Mutates either 'latent' or 'prompt_embedding'.
    - Supports SVD (Hidden Layer Sensitivity) and Gaussian noise mutations.
    - Gen 0 preserves the seed solution.
    """
    def __init__(
        self,
        archive,
        *,
        initial_solution: Dict[str, np.ndarray],
        mutation_key: str = "latent", # "latent" or "prompt_embedding"
        mutation_type: str = "svd",   # "svd" or "gaussian"
        batch_size: int,
        seed: int = None,

        # Model Components
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        
        # Hyperparams
        sigma: float = 0.1,         # For Gaussian
        alpha_min: float = -10.0,   # For SVD
        alpha_max: float = 10.0,    # For SVD
        num_inference_steps: int = 50,
        hook_proportion: float = 0.6,
        
        # SVD Params
        svd_k: int = 10,
        svd_iters: int = 20,
        svd_fd_eps: float = 1e-3,
        svd_layer: str = "mid_block",
        
        # Technical
        guidance_scale: float = 7.5,
        device: str = "cuda",
        bounds = None,
    ):
        self._solution_schema = archive.solution_dim 
        super().__init__(archive=archive, solution_dim=(), bounds=bounds, lower_bounds=None, upper_bounds=None)
        
        if initial_solution is None:
            raise ValueError("initial_solution must be provided.")
        self._seed_sol = initial_solution

        self.mutation_key = mutation_key
        self.mutation_type = mutation_type
        self.batch_size = batch_size
        self._device = device
        
        self._unet = unet
        self._scheduler = scheduler
        
        self.sigma = sigma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._num_inference_steps = num_inference_steps
        self._hook_proportion = hook_proportion
        self._guidance_scale = guidance_scale
        
        self._svd_k = svd_k
        self._svd_iters = svd_iters
        self._svd_fd_eps = svd_fd_eps
        self._svd_layer = svd_layer
        
        self._rng = np.random.default_rng(seed)
        self._torch_rng = torch.Generator(device=device).manual_seed(seed if seed is not None else 0)

        # Validation
        if self.mutation_key not in self._seed_sol:
             raise ValueError(f"mutation_key '{self.mutation_key}' not found in initial_solution")

    def _apply_svd_edit(self, x_t, basis, alpha, svd_index):
        svd_index = max(0, min(svd_index, len(basis) - 1))
        vec = basis[svd_index]
        normed_vec = vec / (vec.norm() + 1e-8)
        return x_t + alpha * normed_vec

    def _apply_noise_edit(self, x_t, alpha):
        # Use torch.randn instead of torch.randn_like to support generator
        noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=self._torch_rng)
        return x_t + alpha * noise

    def ask(self):
        # 1. Select Parent
        is_first_generation = False
        if self.archive.empty:
            is_first_generation = True
            # Use Seed
            parent_sol = {}
            for k, v in self._seed_sol.items():
                if v.ndim == len(self._solution_schema[k]) + 1:
                    parent_sol[k] = v[0]
                else:
                    parent_sol[k] = v
        else:
            # Sample Elite
            sampled = self.archive.sample_elites(1)["solution"]
            parent_sol = {k: v[0] for k, v in sampled.items()}

        # 2. Prepare Tensors
        parent_latents_np = parent_sol["latent"] 
        parent_prompt_np = parent_sol["prompt_embedding"]
        
        # Cast to UNet dtype
        current_prompt_embed = torch.tensor(
            parent_prompt_np, device=self._device, dtype=self._unet.dtype
        )
        parent_xT = torch.tensor(
            parent_latents_np, device=self._device, dtype=self._unet.dtype
        )
        
        if parent_xT.dim() == 3: parent_xT = parent_xT.unsqueeze(0) 
        if current_prompt_embed.dim() == 2: current_prompt_embed = current_prompt_embed.unsqueeze(0)

        # 3. Roll to Hook Point (We always need x_t for sensitivity)
        hook_idx = int(self._hook_proportion * self._num_inference_steps)
        
        x_t, t_scalar, idx, timesteps = roll_to_hook(
            x_T=parent_xT,
            prompt_embedding=current_prompt_embed, 
            unet=self._unet,
            scheduler=self._scheduler,
            guidance_scale=self._guidance_scale,
            num_steps=self._num_inference_steps,
            hook_idx=hook_idx,
            device=self._device,
        )

        # 4. Compute SVD Basis (if needed)
        basis_vecs = None
        if self.mutation_type == "svd":
            if self.mutation_key == "latent":
                basis_vecs = compute_latents_svd(
                    x_t=x_t, t_scalar=t_scalar, unet=self._unet,
                    scheduler=self._scheduler, prompt_embedding=current_prompt_embed,
                    k=self._svd_k, iters=self._svd_iters, fd_eps=self._svd_fd_eps,
                    layer=self._svd_layer, device=self._device,
                    # [FIX] Cast generated numpy int to python int
                    seed=int(self._rng.integers(1e6))
                )
            elif self.mutation_key == "prompt_embedding":
                basis_vecs = compute_embedding_svd(
                    x_t=x_t, t_scalar=t_scalar, unet=self._unet,
                    scheduler=self._scheduler, prompt_embedding=current_prompt_embed,
                    k=self._svd_k, iters=self._svd_iters, fd_eps=self._svd_fd_eps,
                    layer=self._svd_layer, device=self._device,
                    # [FIX] Cast generated numpy int to python int
                    seed=int(self._rng.integers(1e6))
                )

        # 5. Generate Batch
        new_sols = []
        for i in range(self.batch_size):
            # Preserve original in Gen 0
            if is_first_generation and i == 0:
                new_sols.append({
                    "latent": parent_latents_np,
                    "prompt_embedding": parent_prompt_np
                })
                continue
            
            # --- MUTATE ---
            alpha = self._rng.uniform(self.alpha_min, self.alpha_max)
            
            # A) Mutate LATENTS
            if self.mutation_key == "latent":
                if self.mutation_type == "svd":
                    idx_v = self._rng.integers(0, len(basis_vecs))
                    x_t_star = self._apply_svd_edit(x_t, basis_vecs, alpha, idx_v)
                else: # Gaussian
                    x_t_star = self._apply_noise_edit(x_t, self.sigma)
                
                # Invert x_t' -> x_0'
                x_T_star = invert_to_start(
                    start_latents=x_t_star,
                    prompt_embedding=current_prompt_embed,
                    unet=self._unet, scheduler=self._scheduler,
                    guidance_scale=self._guidance_scale,
                    timesteps=timesteps, hook_idx=idx, device=self._device
                )
                
                new_sols.append({
                    "latent": x_T_star.detach().cpu().numpy().squeeze(0),
                    "prompt_embedding": parent_prompt_np # Prompt Unchanged
                })
            
            # B) Mutate PROMPTS
            elif self.mutation_key == "prompt_embedding":
                # Separate Uncond (0) and Cond (1)
                full_embeds = current_prompt_embed.clone()
                c_cond = full_embeds[1:2]
                
                if self.mutation_type == "svd":
                    idx_v = self._rng.integers(0, len(basis_vecs))
                    c_cond_star = self._apply_svd_edit(c_cond, basis_vecs, alpha, idx_v)
                else: # Gaussian
                    c_cond_star = self._apply_noise_edit(c_cond, self.sigma)
                
                # Reconstruct full embedding (Uncond + Cond')
                full_embeds[1:2] = c_cond_star
                
                new_sols.append({
                    "latent": parent_latents_np, # Latent Unchanged
                    "prompt_embedding": full_embeds.detach().cpu().numpy()
                })

        # 6. Collate Batch
        final_batch = {}
        
        # Stack Latents
        latents_list = [s["latent"] for s in new_sols]
        final_batch["latent"] = np.stack(latents_list)
        
        # Stack Prompts
        prompts_list = [s["prompt_embedding"] for s in new_sols]
        final_batch["prompt_embedding"] = np.stack(prompts_list)

        # Copy extra keys from seed (skipping what we generated and 'image')
        for key in self._seed_sol.keys():
            if key in ["latent", "prompt_embedding", "image"]: 
                continue
            if key in parent_sol:
                val = parent_sol[key]
                final_batch[key] = np.repeat(val[np.newaxis, ...], self.batch_size, axis=0)

        return final_batch

    def tell(self, solution, objective, measures, add_info, **fields):
        pass