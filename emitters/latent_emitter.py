import numpy as np
import torch
from diffusers import DDIMScheduler
from diffusers.models import UNet2DConditionModel
from ribs.emitters._emitter_base import EmitterBase
from typing import Tuple

def _phi_hidden(unet: UNet2DConditionModel, x_t, t, prompt_embedding, guidance_scale, *, 
                layer, scheduler):
    """
    Capture a hidden activation from diffusers UNet via a forward hook.
    Returns the CONDITIONAL half (for CFG) flattened to 1D.
    """
    assert scheduler is not None, "Pass the scheduler so we can call scale_model_input once."

    # CFG duplication
    uncond, cond = prompt_embedding[0:1], prompt_embedding[1:2]
    x_in = torch.cat([x_t, x_t], dim=0)
    t_in = torch.cat([t, t], dim=0)
    h_in = torch.cat([uncond, cond], dim=0)

    x_in = scheduler.scale_model_input(x_in, t[0]) # Use scalar t

    # pick target module
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
    xT = x_t.detach().to(device)
    t  = torch.tensor(t_scalar, device=device).expand(1) # _phi_hidden needs a tensor
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
    
    # We must reset timesteps to the full range for the scheduler
    scheduler.set_timesteps(len(timesteps), device=device)
    
    latents  = start_latents.clone()
    
    # Get the timesteps to reverse over
    timesteps_to_reverse = reversed(timesteps[:hook_idx + 1])
    num_reverse_steps = len(timesteps_to_reverse)
    
    for i, t in enumerate(timesteps_to_reverse):
    
        if i >= num_reverse_steps - 1: continue
        
        latent_model_input = torch.cat([latents] * 2) 
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embedding).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # This is the tricky part of DDIM inversion
        current_t = t
        next_t = timesteps_to_reverse[i + 1] if (i + 1) < num_reverse_steps else torch.tensor(-1)
        if next_t.item() == -1: # End of loop
             break

        alpha_t = scheduler.alphas_cumprod[current_t]
        alpha_t_next = scheduler.alphas_cumprod[next_t]
        
        latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred

    return latents




class LatentMutationEmitter(EmitterBase):
    """
    Emits solutions by mutating elites from the archive at an intermediate
    diffusion timestep (t).
    
    This emitter uses a "1-Parent-Many-Offspring" strategy to
    amortize the high cost of computing the SVD basis. When `ask()` is
    called, it samples ONE parent from the archive, computes its SVD
    basis, and then generates the *entire batch* of offspring from it.
    
    If the archive is empty, it uses `x0` as the parent. If 
    `initial_solutions` are provided, the first call to `ask()` will
    return them to seed the archive.
    
    Solutions are stored in the archive in their native 4D/5D shape,
    e.g., (1, 4, 64, 64).

    Args:
        archive (ribs.archives.ArchiveBase): Archive of solutions.
        x0 (torch.Tensor or np.ndarray): The initial solution (e.g.,
            shape (1, 4, 64, 64)) to use as a parent when the
            archive is empty. Must be provided if `initial_solutions` is None.
        initial_solutions (np.ndarray): An array of solutions
            (e.g., shape (batch_size, 1, 4, 64, 64)) to be returned
            the first time `ask()` is called when the archive is empty.
            If `None`, `x0` will be used to generate solutions.
        bounds (None or array-like): Bounds of the solution space.
            (Not typically used for latents, but part of the API).
        batch_size (int): Number of solutions to return in `ask()`.
        seed (int): Value to seed the random number generator.
        
        prompt_embedding (torch.Tensor): The prompt embedding for the model.
        unet (UNet2DConditionModel): The diffusion model's UNet.
        scheduler (DDIMScheduler): The diffusion scheduler.
        
        mutation_type (str): Type of mutation ("svd" or "noise").
        guidance_scale (float): Classifier-free guidance scale.
        num_inference_steps (int): Total diffusion steps.
        hook_proportion (float): Pct of steps to roll forward (0.0-1.0).
        alpha_min (float): Minimum mutation strength.
        alpha_max (float): Maximum mutation strength.
        svd_k (int): Number of singular vectors to compute.
        svd_iters (int): Power iterations for SVD.
        svd_fd_eps (float): Epsilon for finite-difference in JVP.
        svd_layer (str or Tuple): The UNet layer to hook.
        device (str): PyTorch device.
    """
    def __init__(
        self,
        archive,
        *,
        # --- pyribs Standard ---
        x0: torch.Tensor | np.ndarray = None,
        initial_solutions: np.ndarray = None,
        bounds: np.ndarray = None,
        batch_size: int,
        seed: int = None,

        # --- Diffusion Model Components ---
        prompt_embedding: torch.Tensor,
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
    ):
        
        # 1. Initialize EmitterBase
        self._solution_shape = archive.solution_dim
        super().__init__(
            archive=archive,
            solution_dim=self._solution_shape,
            bounds=bounds,
        )
        
        # 2. Handle x0 and initial_solutions (like GaussianEmitter)
        if x0 is None and initial_solutions is None:
            raise ValueError("Either x0 or initial_solutions must be provided.")
        if x0 is not None and initial_solutions is not None:
            raise ValueError("x0 and initial_solutions cannot both be provided.")

        self._x0_numpy = None
        if x0 is not None:
            if isinstance(x0, torch.Tensor):
                self._x0_numpy = x0.cpu().numpy()
            else:
                self._x0_numpy = np.array(x0)
            
            if self._x0_numpy.shape != self._solution_shape:
                raise ValueError(
                    f"x0 shape ({self._x0_numpy.shape}) must match "
                    f"archive.solution_dim ({self._solution_shape})"
                )

        self._initial_solutions = initial_solutions
        if self._initial_solutions is not None:
            # Check batch shape
            if self._initial_solutions.shape[1:] != self._solution_shape:
                 raise ValueError(
                    f"initial_solutions shape ({self._initial_solutions.shape}) must "
                    f"match (batch_size, *archive.solution_dim) "
                    f"(i.e., (N, {self._solution_shape}))"
                )

        # 3. Store components and hyperparams
        self._rng = np.random.default_rng(seed)
        self._torch_rng = torch.Generator(device=device).manual_seed(seed if seed is not None else 0)
        
        self.batch_size = batch_size
        self._device = device
        
        self._unet = unet
        self._scheduler = scheduler
        self._prompt_embedding = prompt_embedding.to(device)
        
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
        Generates `batch_size` solutions by mutating ONE parent
        from the archive.
        
        Returns a (batch_size, *solution_shape) NumPy array.
        """
        
        # 1. Seeding: Return initial solutions on first call if provided
        if self.archive.empty and self._initial_solutions is not None:
            solutions = self._initial_solutions
            self._initial_solutions = None  # Only return them once
            return solutions

        # 2. Sample ONE parent from the archive
        if self.archive.empty:
            # Use x0 as the parent (e.g., shape (4, 64, 64))
            parent_xT_numpy = self._x0_numpy
        else:
            # Sample 1 elite, get its 'solution' field, select first item
            # Shape is (4, 64, 64)
            parent_xT_numpy = self.archive.sample_elites(1)["solution"][0]
        
        # 3. Convert 3D parent -> 3D tensor
        parent_xT = torch.tensor(
            parent_xT_numpy, 
            device=self._device, 
            dtype=torch.float32
        )
        
        
        if parent_xT.dim() == 3:
            parent_xT = parent_xT.unsqueeze(0)
        
        # 4. Roll this parent to the hook layer
        hook_idx = int(self._hook_proportion * self._num_inference_steps)
        x_t, t_scalar, idx, timesteps = roll_to_hook(
            x_T=parent_xT,
            prompt_embedding=self._prompt_embedding,
            unet=self._unet,
            scheduler=self._scheduler,
            guidance_scale=self._guidance_scale,
            num_steps=self._num_inference_steps,
            hook_idx=hook_idx,
            device=self._device,
        )

        # 5. Compute the mutation basis (SVD or Noise)
        # x_t is (1, 4, 64, 64), so basis_vecs are (1, 4, 64, 64)
        basis_vecs = None
        if self._mutation_type == "svd":
            basis_vecs, _ = compute_svd_basis_at_hook(
                x_t=x_t,
                t_scalar=t_scalar,
                unet=self._unet,
                scheduler=self._scheduler,
                prompt_embedding=self._prompt_embedding,
                guidance_scale=self._guidance_scale,
                k=self._svd_k,
                iters=self._svd_iters,
                fd_eps=self._svd_fd_eps,
                device=self._device,
                seed=self._rng.integers(1e6),
                layer=self._svd_layer,
            )

        # 6. Generate the batch of offspring
        new_latents_list = []
        for _ in range(self.batch_size):
            alpha = self._rng.uniform(self._alpha_min, self._alpha_max)
            
            if self._mutation_type == "svd":
                svd_index = self._rng.integers(0, len(basis_vecs))
                # x_t_star is (1, 4, 64, 64)
                x_t_star = self._apply_svd_edit(x_t, basis_vecs, alpha, svd_index)
            
            elif self._mutation_type == "noise":
                # x_t_star is (1, 4, 64, 64)
                x_t_star = self._apply_noise_edit(x_t, alpha)
            
            else:
                raise RuntimeError("Invalid mutation type.")
            
            # Invert the mutated latent
            # x_T_star is (1, 4, 64, 64)
            x_T_star = invert_to_start(
                start_latents=x_t_star,
                prompt_embedding=self._prompt_embedding,
                unet=self._unet,
                scheduler=self._scheduler,
                guidance_scale=self._guidance_scale,
                timesteps=timesteps,
                hook_idx=idx,
                device=self._device,
            )
            new_latents_list.append(x_T_star)
            
        # 7. Stack and convert to NumPy
        solutions_tensor = torch.stack(new_latents_list)

        #
        solutions_for_archive = solutions_tensor.reshape(self.batch_size, *self.solution_dim)

        return solutions_for_archive.cpu().numpy()
    
    def tell(self, solution, objective, measures, add_info, **fields):
        # No-op, as we sample parents uniformly.
        pass