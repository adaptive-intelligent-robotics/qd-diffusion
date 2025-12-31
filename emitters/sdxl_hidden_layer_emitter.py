import numpy as np
import torch
from diffusers import EulerDiscreteScheduler
from diffusers.models import UNet2DConditionModel
from ribs.emitters._emitter_base import EmitterBase
from typing import Dict, Optional, List, Tuple

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

def compute_latents_svd(
    latents, 
    prompt_embeds, 
    pooled_embeds, 
    time_ids, 
    unet, 
    scheduler, 
    k=1, 
    iters=10,
    fd_eps=1e-2,        # <--- Added
    layer="mid_block"   # <--- Added
):
    xT = latents.detach().clone()
    scheduler.set_timesteps(1)
    t = scheduler.timesteps[0].expand(1).to(xT.device)
    
    # Resolve target layer once
    target_module = _get_target_layer(unet, layer)

    def forward_to_hidden(current_latents):
        latents_in = scheduler.scale_model_input(current_latents, t)
        added_cond = {"text_embeds": pooled_embeds, "time_ids": time_ids}
        features = []
        def hook(module, inp, out): features.append(out)
        
        # Register on dynamic target
        handle = target_module.register_forward_hook(hook)
        try:
            unet(latents_in, t, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond)
            return features[0] 
        finally:
            handle.remove()

    vecs = []
    rng = torch.Generator(device=xT.device).manual_seed(42)
    
    with torch.no_grad():
        y0 = forward_to_hidden(xT)

    for _ in range(k):
        v = torch.randn(xT.shape, device=xT.device, dtype=xT.dtype, generator=rng)
        for p in vecs: v -= (v.flatten() @ p.flatten()) * p
        v = v / (v.norm() + 1e-8)
        
        for _ in range(iters):
            with torch.no_grad():
                y_perturbed = forward_to_hidden(xT + fd_eps * v)
            Jv = (y_perturbed - y0) / fd_eps
            
            x_req = xT.clone().requires_grad_(True)
            y_req = forward_to_hidden(x_req)
            loss = (y_req * Jv.detach()).sum()
            grad = torch.autograd.grad(loss, x_req, retain_graph=False)[0]
            
            v_new = grad
            for p in vecs: v_new -= (v_new.flatten() @ p.flatten()) * p
            v = (v_new / (v_new.norm() + 1e-8)).detach()
            
        vecs.append(v)
    return vecs

def compute_embedding_svd(
    target_tensor, 
    fixed_tensor, 
    latents, 
    time_ids, 
    unet, 
    scheduler, 
    target_type="pooled",
    k=1,
    iters=10,
    fd_eps=1e-2,      # <--- Added
    layer="mid_block" # <--- Added
):
    device = target_tensor.device
    dtype = target_tensor.dtype
    c0 = target_tensor.detach().clone()
    
    scheduler.set_timesteps(1)
    t = scheduler.timesteps[0].expand(1).to(device)
    latents_in = scheduler.scale_model_input(latents, t)

    target_module = _get_target_layer(unet, layer)

    def forward_pass(embedding_input):
        if target_type == "pooled":
            p_embeds = fixed_tensor
            pooled = embedding_input
        else: 
            p_embeds = embedding_input
            pooled = fixed_tensor
            
        added_cond = {"text_embeds": pooled, "time_ids": time_ids}
        
        features = []
        def hook(module, inp, out): features.append(out)
        handle = target_module.register_forward_hook(hook)
        try:
            unet(latents_in, t, encoder_hidden_states=p_embeds, added_cond_kwargs=added_cond)
            return features[0]
        finally:
            handle.remove()

    vecs = []
    rng = torch.Generator(device=device).manual_seed(42)
    
    with torch.no_grad():
        y0 = forward_pass(c0)
        
    for _ in range(k):
        v = torch.randn(c0.shape, device=device, dtype=dtype, generator=rng)
        for p in vecs: v -= (v.flatten() @ p.flatten()) * p
        v = v / (v.norm() + 1e-8)
        
        for _ in range(iters):
            with torch.no_grad():
                y_perturbed = forward_pass(c0 + fd_eps * v)
            Jv = (y_perturbed - y0) / fd_eps
            
            c_req = c0.clone().requires_grad_(True)
            y_req = forward_pass(c_req)
            loss = (y_req * Jv.detach()).sum()
            grad = torch.autograd.grad(loss, c_req, retain_graph=False)[0]
            
            v_new = grad
            for p in vecs: v_new -= (v_new.flatten() @ p.flatten()) * p
            v = (v_new / (v_new.norm() + 1e-8)).detach()
            
        vecs.append(v)
    return vecs


# =============================================================================
# 2. THE UPDATED EMITTER
# =============================================================================

class SDXLHiddenLayerEmitter(EmitterBase):
    def __init__(
        self,
        archive,
        *,
        initial_solution: Dict[str, np.ndarray],
        mutation_key: str = "latent",
        mutation_type: str = "svd",
        batch_size: int,
        seed: int = None,

        # Model Components
        unet: UNet2DConditionModel,
        scheduler: EulerDiscreteScheduler,
        
        # Hyperparams
        sigma: float = 0.1,
        alpha_min: float = -5.0,
        alpha_max: float = 5.0,
        svd_k: int = 5,
        svd_iters: int = 5,
        
        # [ADDED] The missing params
        svd_fd_eps: float = 1e-2, 
        svd_layer: str = "mid_block",
        
        # Technical
        device: str = "cuda",
        bounds = None,
    ):
        self._solution_schema = archive.solution_dim
        super().__init__(archive=archive, solution_dim=(), bounds=bounds, lower_bounds=None, upper_bounds=None)
        
        if initial_solution is None:
            raise ValueError("initial_solution must be provided.")
            
        self.mutation_key = mutation_key
        self.mutation_type = mutation_type
        self.batch_size = batch_size
        self._device = device
        
        self._unet = unet
        self._scheduler = scheduler
        
        self.sigma = sigma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.svd_k = svd_k
        self.svd_iters = svd_iters
        
        # [ADDED] Store them
        self.svd_fd_eps = svd_fd_eps
        self.svd_layer = svd_layer
        
        self._rng = np.random.default_rng(seed)
        
        # Sanitize Seed
        self._seed_sol = {}
        for k, v in initial_solution.items():
            if isinstance(v, torch.Tensor):
                self._seed_sol[k] = v.detach().cpu().numpy()
            else:
                self._seed_sol[k] = np.array(v)

        if self.mutation_key not in self._seed_sol:
            raise ValueError(f"mutation_key '{self.mutation_key}' not in initial_solution")

    def ask(self):
        # 1. Select Parent
        is_first_gen = False
        if self.archive.empty:
            is_first_gen = True
            parent_sol = {}
            for k, v in self._seed_sol.items():
                if v.ndim == len(self._solution_schema[k]) + 1:
                    parent_sol[k] = v[0]
                else:
                    parent_sol[k] = v
        else:
            sampled = self.archive.sample_elites(1)["solution"]
            parent_sol = {k: v[0] for k, v in sampled.items()}

        # 2. Prepare Tensors
        latents = torch.tensor(parent_sol["latent"], device=self._device, dtype=self._unet.dtype)
        prompt_embeds = torch.tensor(parent_sol["prompt_embedding"], device=self._device, dtype=self._unet.dtype)
        pooled_embeds = torch.tensor(parent_sol["pooled_prompt_embedding"], device=self._device, dtype=self._unet.dtype)
        
        if latents.dim() == 3: latents = latents.unsqueeze(0)
        if prompt_embeds.dim() == 2: prompt_embeds = prompt_embeds.unsqueeze(0)
        if pooled_embeds.dim() == 1: pooled_embeds = pooled_embeds.unsqueeze(0)
        
        time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], device=self._device, dtype=self._unet.dtype)

        if self.mutation_key == "latent":
            target_tensor = latents
        elif self.mutation_key == "prompt_embedding":
            target_tensor = prompt_embeds
        elif self.mutation_key == "pooled_prompt_embedding":
            target_tensor = pooled_embeds
        else:
            raise ValueError(f"Unknown mutation key: {self.mutation_key}")

        # 3. Compute Basis
        basis_vecs = None
        
        if self.mutation_type == "svd":
            if self.mutation_key == "latent":
                basis_vecs = compute_latents_svd(
                    latents, prompt_embeds, pooled_embeds, time_ids,
                    self._unet, self._scheduler, 
                    k=self.svd_k, 
                    iters=self.svd_iters,
                    fd_eps=self.svd_fd_eps,   # <--- Pass it
                    layer=self.svd_layer      # <--- Pass it
                )
            else:
                t_type = "pooled" if self.mutation_key == "pooled_prompt_embedding" else "prompt"
                other = prompt_embeds if t_type == "pooled" else pooled_embeds
                
                basis_vecs = compute_embedding_svd(
                    target_tensor, other, latents, time_ids,
                    self._unet, self._scheduler, 
                    target_type=t_type,
                    k=self.svd_k, 
                    iters=self.svd_iters,
                    fd_eps=self.svd_fd_eps,   # <--- Pass it
                    layer=self.svd_layer      # <--- Pass it
                )

        # 4. Generate Batch
        mutated_targets = []
        
        for i in range(self.batch_size):
            if is_first_gen and i == 0:
                mutated_targets.append(target_tensor.clone())
                continue

            if self.mutation_type == "gaussian":
                noise = torch.randn_like(target_tensor)
                child = target_tensor + (noise * self.sigma)
                mutated_targets.append(child)
                
            elif self.mutation_type == "svd":
                alpha = self._rng.uniform(self.alpha_min, self.alpha_max)
                idx = self._rng.integers(0, len(basis_vecs))
                direction = basis_vecs[idx]
                direction = direction / (direction.norm() + 1e-8)
                child = target_tensor + (alpha * direction)
                mutated_targets.append(child)

        batch_tensor = torch.stack(mutated_targets)
        batch_np = batch_tensor.detach().cpu().numpy()
        if batch_np.ndim == target_tensor.ndim + 1:
             batch_np = batch_np.squeeze(1)

        final_batch = {}
        final_batch[self.mutation_key] = batch_np
        
        for k in self._seed_sol.keys():
            if k == self.mutation_key: continue
            val = parent_sol[k] 
            final_batch[k] = np.repeat(val[np.newaxis, ...], self.batch_size, axis=0)
            
        return final_batch

    def tell(self, solution, objective, measures, add_info, **fields):
        pass