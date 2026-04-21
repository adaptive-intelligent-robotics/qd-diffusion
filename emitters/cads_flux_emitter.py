import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from ribs.emitters._emitter_base import EmitterBase
from typing import Tuple, Dict, Optional, Union, List

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

# =============================================================================
# 2. CADS SCHEDULE AND RESCALING
# =============================================================================

def create_piecewise_gamma_schedule(num_steps: int, tau1: float = 0.2, tau2: float = 0.8) -> np.ndarray:
    """
    Creates the piecewise linear annealing schedule gamma(t).
    
    Args:
        num_steps: Number of diffusion steps.
        tau1: Threshold below which gamma = 1 (no corruption at the end).
        tau2: Threshold above which gamma = 0 (high corruption at the start).
        
    Returns:
        Array of gamma values [num_steps].
    """
    # Diffusion time t runs from 1.0 down to 0.0
    t = np.linspace(1.0, 0.0, num_steps, dtype=np.float32)
    gamma = np.zeros_like(t)
    
    for i, time_val in enumerate(t):
        if time_val <= tau1:
            gamma[i] = 1.0
        elif tau1 < time_val < tau2:
            gamma[i] = (tau2 - time_val) / (tau2 - tau1)
        else:
            # time_val >= tau2
            gamma[i] = 0.0
            
    return gamma

def apply_cads_conditioning(y: torch.Tensor, gamma_t: float, s: float, psi: float) -> torch.Tensor:
    """
    Applies noise to the condition and rescales it according to CADS Algorithm 1.
    
    Args:
        y: Clean conditioning vector.
        gamma_t: Annealing factor for the current step (0.0 to 1.0).
        s: Initial noise scale.
        psi: Mixing factor for rescaling (0.0 to 1.0).
        
    Returns:
        The final corrupted and rescaled conditioning vector.
    """
    if gamma_t >= 1.0:
        return y
        
    device = y.device
    dtype = y.dtype
    
    # Variance-preserving noise injection
    n = torch.randn_like(y, device=device, dtype=dtype)
    y_hat = (gamma_t ** 0.5) * y + s * ((1.0 - gamma_t) ** 0.5) * n
    
    # Statistical normalisation and rescaling
    mu_in = y.mean()
    sigma_in = y.std()
    
    mu_hat = y_hat.mean()
    sigma_hat = y_hat.std()
    
    # Rescale to match original distribution
    y_rescaled = ((y_hat - mu_hat) / (sigma_hat + 1e-7)) * sigma_in + mu_in
    
    # Final mixing
    y_final = psi * y_rescaled + (1.0 - psi) * y_hat
    
    return y_final

@torch.no_grad()
def flux_forward_cads(pipe, latents, prompt_embeds, pooled_prompt_embeds, 
                      text_ids, latent_img_ids, num_steps, guidance_scale, 
                      gamma_schedule, noise_scale_s: float = 1.0, mixing_factor_psi: float = 0.5):
    """
    Forward pass through diffusion using strict CADS noise application and rescaling.
    """
    device = latents.device
    curr = latents.clone()
    
    mu = get_flux_mu(pipe, curr)
    pipe.scheduler.set_timesteps(num_steps, device=device, mu=mu)
    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas
    
    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0]) if pipe.transformer.config.guidance_embeds else None
    
    for i in range(num_steps):
        t = timesteps[i]
        
        if gamma_schedule is not None:
            gamma_t = float(gamma_schedule[i])
            curr_prompt_embeds = apply_cads_conditioning(
                prompt_embeds, gamma_t, noise_scale_s, mixing_factor_psi
            )
            curr_pooled_embeds = apply_cads_conditioning(
                pooled_prompt_embeds, gamma_t, noise_scale_s, mixing_factor_psi
            )
        else:
            curr_prompt_embeds = prompt_embeds
            curr_pooled_embeds = pooled_prompt_embeds
            
        with pipe.transformer.cache_context("cond"):
            v = pipe.transformer(
                hidden_states=curr,
                timestep=t.expand(curr.shape[0]).to(curr.dtype) / 1000.0,
                guidance=guidance,
                pooled_projections=curr_pooled_embeds,
                encoder_hidden_states=curr_prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_img_ids,
                return_dict=False,
            )[0]
        
        curr += (sigmas[i+1] - sigmas[i]) * v
    
    return curr

# =============================================================================
# 3. CADS FLUX EMITTER
# =============================================================================

class CADSFluxEmitter(EmitterBase):
    """
    Condition-Annealed Diffusion Sampling emitter for Flux-1 pipelines.
    Implements the piecewise linear schedule and statistical rescaling 
    from the original CADS paper.
    """
    
    def __init__(
        self,
        archive,
        initial_solution: dict,
        pipe,
        text_ids,
        latent_img_ids,
        batch_size: int = 4,
        seed: int = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        tau1: float = 0.2,
        tau2: float = 0.8,
        noise_scale_s: float = 1.0,
        mixing_factor_psi: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize CADS emitter with static piecewise scheduling.
        
        Args:
            archive: QD archive
            initial_solution: Template solution dict
            pipe: Flux diffusion pipeline
            text_ids: Text token IDs tensor
            latent_img_ids: Image latent IDs tensor
            batch_size: Number of samples per batch
            seed: Random seed for reproducibility
            num_inference_steps: Number of diffusion denoising steps
            guidance_scale: Unconditional guidance scale
            tau1: Threshold below which gamma = 1 (end of decay)
            tau2: Threshold above which gamma = 0 (start of decay)
            noise_scale_s: Initial noise scale multiplier (s in the paper)
            mixing_factor_psi: Rescaling mixing factor (psi in the paper)
            device: Device to run on ("cuda" or "cpu")
        """
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
        
        self.batch_size = batch_size
        self.num_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        # New CADS parameters hardcoded as floats
        self.tau1 = tau1
        self.tau2 = tau2
        self.noise_scale_s = noise_scale_s
        self.mixing_factor_psi = mixing_factor_psi
        
        self.device = device
        self._rng = np.random.default_rng(seed)
    
    def _create_schedule(self, num_steps: int) -> np.ndarray:
        """Create a piecewise gamma schedule for this batch element."""
        return create_piecewise_gamma_schedule(num_steps, self.tau1, self.tau2)
    
    def ask(self):
        """
        Generate a batch of solutions using CADS with static schedule thresholds.
        """
        if self.archive.empty:
            parent_sol = {k: v.to(self.device) for k, v in self._seed_sol.items()}
        else:
            sampled = self.archive.sample_elites(1)["solution"]
            parent_sol = {
                k: torch.from_numpy(v[0]).to(device=self.device, dtype=self.pipe.transformer.dtype)
                for k, v in sampled.items()
            }
        
        # Ensure correct shapes [1, Seq, Dim]
        p_latent = parent_sol["latent"]
        if p_latent.ndim == 2:
            p_latent = p_latent.unsqueeze(0)
        
        p_embeds = parent_sol["prompt_embedding"]
        if p_embeds.ndim == 2:
            p_embeds = p_embeds.unsqueeze(0)
        
        p_pooled = parent_sol["pooled_prompt_embedding"]
        if p_pooled.ndim == 1:
            p_pooled = p_pooled.unsqueeze(0)
        
        new_batch = {"latent": [], "prompt_embedding": [], "pooled_prompt_embedding": []}
        
        def prepare_for_archive(t):
            arr = t.detach().cpu().float().numpy()
            if arr.ndim > 0 and arr.shape[0] == 1:
                return np.squeeze(arr, axis=0)
            return arr
        
        for i in range(self.batch_size):
            if self.archive.empty and i == 0:
                for k in ["latent", "prompt_embedding", "pooled_prompt_embedding"]:
                    new_batch[k].append(prepare_for_archive(parent_sol[k]))
                continue
            
            # Create identical gamma schedule for each batch member
            gamma_schedule = self._create_schedule(self.num_steps)
            
            with torch.no_grad():
                generated_latent = flux_forward_cads(
                    self.pipe,
                    p_latent,
                    p_embeds,
                    p_pooled,
                    self.text_ids,
                    self.latent_img_ids,
                    self.num_steps,
                    self.guidance_scale,
                    gamma_schedule=gamma_schedule,
                    noise_scale_s=self.noise_scale_s,
                    mixing_factor_psi=self.mixing_factor_psi
                )
            
            new_batch["latent"].append(prepare_for_archive(generated_latent))
            new_batch["prompt_embedding"].append(prepare_for_archive(p_embeds))
            new_batch["pooled_prompt_embedding"].append(prepare_for_archive(p_pooled))
            
            torch.cuda.empty_cache()
        
        return {k: np.stack(v) for k, v in new_batch.items()}
    
    def tell(self, solution, objective, measures, add_info, **fields):
        pass