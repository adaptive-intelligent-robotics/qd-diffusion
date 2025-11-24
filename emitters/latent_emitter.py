import numpy as np
import torch
from diffusers import DDIMScheduler
from diffusers.models import UNet2DConditionModel
from ribs.emitters._emitter_base import EmitterBase
from typing import Tuple, Dict, Optional

# =============================================================================
# 1. HELPER FUNCTIONS (Gradient/Hook Logic)
# =============================================================================

def _phi_hidden(unet: UNet2DConditionModel, x_t, t, prompt_embedding, guidance_scale, *, 
                layer, scheduler):
    """
    Capture a hidden activation from diffusers UNet via a forward hook.
    Returns the CONDITIONAL half (for CFG) flattened to 1D.
    """
    assert scheduler is not None
    
    # CFG duplication
    uncond, cond = prompt_embedding[0:1], prompt_embedding[1:2]
    x_in = torch.cat([x_t, x_t], dim=0)
    t_in = torch.cat([t, t], dim=0)
    h_in = torch.cat([uncond, cond], dim=0)

    x_in = scheduler.scale_model_input(x_in, t[0])

    # Pick target module
    if isinstance(layer, str) and layer == "mid_block":
        target = unet.mid_block
    else:
        mod = unet
        for key in layer:
            mod = mod[key] if isinstance(key, int) else getattr(mod, key)
        target = mod

    captured = {}
    def _hook(_m, _inp, out):
        captured["feat"] = out

    handle = target.register_forward_hook(_hook)
    try:
        _ = unet(x_in, t_in, encoder_hidden_states=h_in)
        feats = captured.get("feat", None)
        if feats is None:
            raise RuntimeError("Hidden feature was not captured; check `layer` choice.")
        
        if feats.dim() >= 1 and feats.size(0) >= 2:
            feats = feats[feats.size(0)//2:]

        return feats.reshape(-1)
    finally:
        handle.remove()

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

def compute_svd_basis_at_hook(
    x_t: torch.Tensor,
    t_scalar: int,
    *,
    unet,
    scheduler,
    prompt_embedding,
    guidance_scale: float = 7.5,
    k: int = 10,
    iters: int = 20,
    fd_eps: float = 1e-3,
    device: str = "cuda",
    seed: int | None = None,
    layer="mid_block",
):
    """Computes the top-k Singular Vectors of the Jacobian at the hook point."""
    xT = x_t.detach().to(device)
    t  = torch.tensor(t_scalar, device=device).expand(1)
    prompt_embedding = prompt_embedding.to(device)

    with torch.no_grad():
        y0 = _phi_hidden(
            unet, xT, t, prompt_embedding, guidance_scale,
            layer=layer, scheduler=scheduler
        )

    def Jx_v(vx: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y1 = _phi_hidden(
                unet, xT + fd_eps * vx, t, prompt_embedding, guidance_scale,
                layer=layer, scheduler=scheduler
            )
        return (y1 - y0) / fd_eps

    def JxT_u(u_flat: torch.Tensor) -> torch.Tensor:
        x_req = xT.clone().requires_grad_(True)
        y = _phi_hidden(
            unet, x_req, t, prompt_embedding, guidance_scale,
            layer=layer, scheduler=scheduler
        )
        dot = (y * u_flat.detach()).sum()
        (g,) = torch.autograd.grad(
            dot, x_req, retain_graph=False, allow_unused=True
        )
        return torch.zeros_like(x_req) if g is None else g

    rng = None
    if seed is not None:
        rng = torch.Generator(device=device).manual_seed(int(seed))

    vecs, sigmas = [], []
    for _ in range(k):
        v = torch.randn(xT.shape, device=xT.device, generator=rng, dtype=xT.dtype)
        for p in vecs:
            v -= (v.flatten() @ p.flatten()) * p
        v = v / (v.norm() + 1e-8)

        for __ in range(iters):
            u  = Jx_v(v)
            Av = JxT_u(u)
            for p in vecs:
                Av -= (Av.flatten() @ p.flatten()) * p
            v = (Av / (Av.norm() + 1e-8)).detach()

        u  = Jx_v(v)
        Av = JxT_u(u)
        rq = (v.flatten() @ Av.flatten())
        sigmas.append(float(torch.sqrt(torch.clamp(rq, min=0.0))))
        vecs.append(v.clone())

    return vecs, sigmas

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


# =============================================================================
# 2. THE EMITTER CLASS
# =============================================================================

class SDLatentEmitter(EmitterBase):
    """
    Emits solutions for SD 1.5 pipelines.
    Expects keys: 'latent' and 'prompt_embedding'.
    
    Features:
    - Uses `initial_solution` as the seed for Generation 0.
    - In Gen 0, the first offspring is exact clone (alpha=0) to preserve original.
    - Casts inputs to UNet precision (Float32/16) for speed and safety.
    """
    def __init__(
        self,
        archive,
        *,
        initial_solution: Dict[str, np.ndarray], # Mandatory Seed (Size 1)
        batch_size: int,
        seed: int = None,

        # --- Diffusion Model Components ---
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        
        # --- Algorithm Hyperparams ---
        mutation_type: str = "svd",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        hook_proportion: float = 0.6,
        alpha_min: float = -10.0,
        alpha_max: float = 10.0,
        
        # --- SVD-Specific Params ---
        svd_k: int = 10,
        svd_iters: int = 20,
        svd_fd_eps: float = 1e-3,
        svd_layer: str | Tuple = "mid_block",
        
        # --- Technical ---
        device: str = "cuda",
        bounds = None,
    ):
        
        self._solution_schema = archive.solution_dim 
        # Pass empty tuple for solution_dim to bypass EmitterBase checks
        super().__init__(
            archive=archive,
            solution_dim=(), 
            bounds=bounds,
        )
        
        # --- Handle Seed ---
        if initial_solution is None:
            raise ValueError("initial_solution must be provided as the seed.")

        # Store seed. We expect it to be size 1, but we handle normalization later.
        self._seed_sol = initial_solution

        # --- Standard Setup ---
        self._rng = np.random.default_rng(seed)
        self._torch_rng = torch.Generator(device=device).manual_seed(seed if seed is not None else 0)
        self.batch_size = batch_size
        self._device = device
        
        self._unet = unet
        self._scheduler = scheduler
        
        self._mutation_type = mutation_type
        self._guidance_scale = guidance_scale
        self._num_inference_steps = num_inference_steps
        self._hook_proportion = hook_proportion
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        
        self._svd_k = svd_k
        self._svd_iters = svd_iters
        self._svd_fd_eps = svd_fd_eps
        self._svd_layer = svd_layer

    def _apply_svd_edit(self, x_t, basis, alpha, svd_index):
        svd_index = max(0, min(svd_index, len(basis) - 1))
        vec = basis[svd_index]
        normed_vec = vec / (vec.norm() + 1e-8)
        return x_t + alpha * normed_vec

    def _apply_noise_edit(self, x_t, alpha):
        noise = torch.randn_like(x_t, generator=self._torch_rng)
        return x_t + alpha * noise

    def ask(self):
        """
        Generates `batch_size` solutions.
        - If Iteration 0: Mutates the Seed. First child is exact copy.
        - If Iteration 1+: Mutates an Elite from Archive.
        """
        
        # 1. Select Parent
        is_first_generation = False

        if self.archive.empty:
            is_first_generation = True
            # Use Seed. Collapse batch dim (1, ...) -> (...)
            parent_sol = {
                k: (v[0] if v.shape[0] == 1 else v) 
                for k, v in self._seed_sol.items()
            }
        else:
            # Sample from Archive. Returns dict of batches (1, ...).
            sampled = self.archive.sample_elites(1)["solution"]
            parent_sol = {k: v[0] for k, v in sampled.items()}

        # 2. Extract Data & Cast to UNet Precision
        parent_latents_np = parent_sol["latent"] 
        parent_prompt_np = parent_sol["prompt_embedding"]
        
        # [SPEED FIX] Cast to self._unet.dtype (e.g. float32/16)
        current_prompt_embed = torch.tensor(
            parent_prompt_np, 
            device=self._device, 
            dtype=self._unet.dtype
        )
       
        parent_xT = torch.tensor(
            parent_latents_np, 
            device=self._device, 
            dtype=self._unet.dtype
        )
        
        # Ensure 4D (1, C, H, W)
        if parent_xT.dim() == 3: 
            parent_xT = parent_xT.unsqueeze(0) 

        # 3. Roll to Hook Point
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
        if self._mutation_type == "svd":
            basis_vecs, _ = compute_svd_basis_at_hook(
                x_t=x_t,
                t_scalar=t_scalar,
                unet=self._unet,
                scheduler=self._scheduler,
                prompt_embedding=current_prompt_embed,
                guidance_scale=self._guidance_scale,
                k=self._svd_k,
                iters=self._svd_iters,
                fd_eps=self._svd_fd_eps,
                device=self._device,
                seed=self._rng.integers(1e6),
                layer=self._svd_layer,
            )

        # 5. Generate Batch of Offspring
        new_latents_list = []
        for i in range(self.batch_size):
            
            # [LOGIC] If Gen 0, keep index 0 as exact copy (Original Image)
            if is_first_generation and i == 0:
                alpha = 0.0
            else:
                alpha = self._rng.uniform(self._alpha_min, self._alpha_max)
            
            if self._mutation_type == "svd":
                svd_index = self._rng.integers(0, len(basis_vecs))
                x_t_star = self._apply_svd_edit(x_t, basis_vecs, alpha, svd_index)
            else:
                x_t_star = self._apply_noise_edit(x_t, alpha)
            
            # Invert back to t=0 (Latent Space)
            x_T_star = invert_to_start(
                start_latents=x_t_star,
                prompt_embedding=current_prompt_embed, 
                unet=self._unet,
                scheduler=self._scheduler,
                guidance_scale=self._guidance_scale,
                timesteps=timesteps,
                hook_idx=idx,
                device=self._device,
            )
            new_latents_list.append(x_T_star)
            
        # Stack -> (Batch, 1, C, H, W)
        mutated_latents_batch = torch.stack(new_latents_list).cpu().numpy()
        
        # Squeeze batch dim 1 -> (Batch, C, H, W)
        if mutated_latents_batch.ndim == 5 and mutated_latents_batch.shape[1] == 1:
             mutated_latents_batch = mutated_latents_batch.squeeze(1)

        # 6. Construct Final Dictionary
        final_batch = {}
        
        # New Latents
        final_batch["latent"] = mutated_latents_batch
        
        # Repeat Prompts (Identical for whole batch)
        parent_prompt = parent_sol["prompt_embedding"]
        final_batch["prompt_embedding"] = np.repeat(
            parent_prompt[np.newaxis, ...], 
            self.batch_size, 
            axis=0
        )
        
        # [FIX] Iterate over seed keys to prevent accessing 'image'
        # This mirrors the logic in your SDXL emitter.
        for key in self._seed_sol.keys():
            if key in ["latent", "prompt_embedding"]: 
                continue # Already handled above
            
            if key in parent_sol:
                parent_item = parent_sol[key]
                final_batch[key] = np.repeat(
                    parent_item[np.newaxis, ...], 
                    self.batch_size, 
                    axis=0
                )
                
        return final_batch

    def tell(self, solution, objective, measures, add_info, **fields):
        pass