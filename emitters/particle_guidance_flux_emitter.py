"""
Particle Guidance Emitter for Flux Models in a QD Loop.

Implements the Particle Guidance mechanism for latent space exploration
with Flux-1 diffusion models, optimized for quality-diversity search.

Supports particle repulsion in both latent space and feature space (DINO).
"""

import numpy as np
import torch
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from ribs.emitters._emitter_base import EmitterBase
from typing import Dict, Optional, Union
import copy

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
# 2. PARTICLE GUIDANCE HELPERS
# =============================================================================

def compute_particle_gradients_flux(
    latents: torch.Tensor,
    scores: torch.Tensor,
    sigma: float,
    coeff: float = 0.5,
    power: float = 2.0,
    sigma_threshold: float = 0.5,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Computes particle guidance gradients using latent-space score divergence.
    
    Args:
        latents: Current latent samples [batch_size, seq_len, dim]
        scores: Noise prediction scores [batch_size, seq_len, dim]
        sigma: Current noise level
        coeff: Particle guidance coefficient
        power: Power for distance weighting
        device: Device to compute on
    
    Returns:
        Particle guidance gradient for latents
    """
    batch_size = latents.shape[0]
    if batch_size <= 1:
        return torch.zeros_like(latents)

    # Flatten sequence dimension for distance computation
    latents_flat = latents.reshape(batch_size, -1)
    scores_flat = -scores.reshape(batch_size, -1)
    
    # Compute pairwise differences
    diff = latents_flat.unsqueeze(1) - latents_flat.unsqueeze(0)  # [N, N, d]
    
    # Remove diagonal (self-differences) and reshape
    diff = diff[~torch.eye(diff.shape[0], dtype=bool)].view(diff.shape[0], -1, diff.shape[-1])
    
    # Compute distances
    distance = torch.norm(diff, p=2, dim=-1, keepdim=True)  # [N, N-1, 1]
    
    # Compute bandwidth using median heuristic
    num_images = latents_flat.shape[0]
    h_t = (distance.median(dim=1, keepdim=True)[0]) ** 2 / np.log(num_images - 1)
    
    # Compute kernel weights
    weights = torch.exp(-(distance ** power / h_t))
    
    coeff_applied = coeff if sigma >= sigma_threshold else 0.0
    
    # Compute gradient
    grad_phi = 2 * weights * diff / h_t * sigma * coeff_applied
    grad_phi = grad_phi.sum(dim=1)
    grad_phi = grad_phi.view_as(latents)
    
    return grad_phi


def compute_dino_gradients_flux(
    latents: torch.Tensor,
    pipe,
    dino_model,
    text_ids: torch.Tensor,
    latent_img_ids: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    coeff: float = 0.5,
    power: float = 2.0,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Computes particle guidance gradients using DINO feature space.
    
    Requires backpropagation through VAE decoder and DINO model.
    """
    latents.requires_grad_(True)
    pipe.vae.decoder.requires_grad_(True)
    dino_model.eval()
    pipe.vae.eval()
    
    # Decode latents to images
    x_pred = decode_latents_flux(latents, pipe)
    
    # Extract DINO features
    dino_out = dino_model(x_pred)
    
    # Compute pairwise differences in feature space
    batch_size = dino_out.shape[0]
    latents_vec = dino_out.view(batch_size, -1)
    diff = latents_vec.unsqueeze(1) - latents_vec.unsqueeze(0)  # [N, N, d]
    
    # Remove diagonal
    diff = diff[~torch.eye(diff.shape[0], dtype=bool)].view(diff.shape[0], -1, diff.shape[-1])
    
    # Compute distances
    distance = torch.norm(diff, p=2, dim=-1, keepdim=True)  # [N, N-1, 1]
    
    # Compute bandwidth
    num_images = latents_vec.shape[0]
    h_t = (distance.median(dim=1, keepdim=True)[0]) ** 2 / np.log(num_images - 1)
    
    # Compute kernel weights
    weights = torch.exp(-(distance ** power / h_t))
    
    # Compute gradient in feature space
    grad_phi = 2 * weights * diff / h_t * coeff
    grad_phi = grad_phi.sum(dim=1)
    
    # Backprop to latents
    eval_sum = torch.sum(dino_out * grad_phi.detach())
    deps_dx_backprop = torch.autograd.grad(eval_sum, latents)[0]
    grad_phi_latent = deps_dx_backprop.view_as(latents)
    
    return grad_phi_latent


def decode_latents_flux(latents: torch.Tensor, pipe, stay_on_device: bool = False) -> torch.Tensor:
    """
    Decode Flux latents to images using VAE.
    
    Args:
        latents: Latent samples
        pipe: Flux pipeline
        stay_on_device: Whether to keep on device or move to CPU
    
    Returns:
        Decoded images [batch, 3, height, width]
    """
    latents_decoded = latents / pipe.vae_scale_factor
    pipe.vae.eval()
    for param in pipe.vae.parameters():
        param.requires_grad = True
    
    image = pipe.vae.decode(latents_decoded).sample
    
    if not stay_on_device:
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    
    return image


# =============================================================================
# 3. PARTICLE GUIDANCE EMITTER FOR FLUX
# =============================================================================

class ParticleGuidanceFluxEmitter(EmitterBase):
    """
    Emitter for Flux-1 pipelines guided by particle repulsion in the diffusion loop.
    
    Implements particle guidance to encourage diversity while maintaining quality.
    Supports both latent-space and DINO feature-space particle repulsion.
    """
    
    def __init__(
        self,
        archive,
        pipe,
        text_ids: torch.Tensor,
        latent_img_ids: torch.Tensor,
        initial_solution: Optional[Dict[str, np.ndarray]] = None,
        use_dino: bool = False,
        dino_model: Optional[torch.nn.Module] = None,
        batch_size: int = 4,
        seed: Optional[int] = None,
        # Diffusion parameters
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        particle_coeff: float = 0.5,
        sigma_threshold: float = 0.5,
        dino_sigma_threshold: float = 3.0,
        # Sampling parameters
        generator_seed: Optional[int] = None,
        device: str = "cuda"
    ):
        """
        Args:
            archive: QD archive to store solutions
            pipe: Flux pipeline (must have scheduler, transformer, vae, etc.)
            text_ids: Text token IDs tensor for Flux
            latent_img_ids: Image latent IDs tensor for Flux
            initial_solution: Initial solution for seeding
            use_dino: Whether to use DINO-based guidance
            dino_model: DINO model (required if use_dino=True)
            batch_size: Number of particles to generate per ask() (all jointly denoised)
            seed: RNG seed
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            particle_coeff: Particle guidance coefficient
            dino_sigma_threshold: Sigma threshold for DINO guidance
            generator_seed: Seed for torch.Generator
            device: Device to run on
        """
        super().__init__(
            archive=archive,
            solution_dim=(),
            bounds=None,
            lower_bounds=None,
            upper_bounds=None
        )
        
        self.pipe = pipe
        self.text_ids = text_ids[0] if text_ids.ndim == 3 else text_ids
        self.latent_img_ids = latent_img_ids[0] if latent_img_ids.ndim == 3 else latent_img_ids
        self.use_dino = use_dino
        self.dino_model = dino_model
        self.batch_size = batch_size
        self.num_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.particle_coeff = particle_coeff
        self.sigma_threshold = sigma_threshold
        self.dino_sigma_threshold = dino_sigma_threshold
        self.device = device
        
        self._rng = np.random.default_rng(seed)
        self._torch_rng = torch.Generator(device=device)
        if generator_seed is not None:
            self._torch_rng.manual_seed(generator_seed)
        
        self._initial_solution = initial_solution
        self._scheduler_setup = False
        
        if use_dino and dino_model is None:
            raise ValueError("dino_model must be provided when use_dino=True")
    
    def ask(self):
        """
        Generate a batch of new solutions using particle guidance diffusion.

        Returns:
            Dictionary with keys "latent", "prompt_embedding", and "pooled_prompt_embedding"
        """
        # 1. Parent Selection
        if self.archive.empty:
            if self._initial_solution is None:
                raise ValueError("Archive is empty and no initial_solution provided")
            parent_solution = self._initial_solution
        else:
            sampled = self.archive.sample_elites(1)["solution"]
            parent_solution = {k: v[0] for k, v in sampled.items()}

        # 2. Convert parent to torch tensors
        latent_data = parent_solution["latent"]
        if isinstance(latent_data, torch.Tensor):
            parent_latent = latent_data.float().to(device=self.device, dtype=self.pipe.transformer.dtype)
        else:
            parent_latent = torch.from_numpy(latent_data).float().to(
                device=self.device, dtype=self.pipe.transformer.dtype
            )

        # T5 text embeddings: [seq_len, 4096] -> [1, seq_len, 4096]
        embedding_data = parent_solution["prompt_embedding"]
        if isinstance(embedding_data, torch.Tensor):
            parent_embedding = embedding_data.float().to(device=self.device, dtype=self.pipe.transformer.dtype)
        else:
            parent_embedding = torch.from_numpy(embedding_data).float().to(
                device=self.device, dtype=self.pipe.transformer.dtype
            )

        # CLIP pooled embeddings: [768] -> [1, 768]
        pooled_data = parent_solution["pooled_prompt_embedding"]
        if isinstance(pooled_data, torch.Tensor):
            parent_pooled = pooled_data.float().to(device=self.device, dtype=self.pipe.transformer.dtype)
        else:
            parent_pooled = torch.from_numpy(pooled_data).float().to(
                device=self.device, dtype=self.pipe.transformer.dtype
            )

        # Ensure correct batch shapes for Flux
        if parent_latent.ndim == 2:      # [seq_len, dim] -> [1, seq_len, dim]
            parent_latent = parent_latent.unsqueeze(0)
        if parent_embedding.ndim == 2:   # [seq_len, 4096] -> [1, seq_len, 4096]
            parent_embedding = parent_embedding.unsqueeze(0)
        if parent_pooled.ndim == 1:      # [768] -> [1, 768]
            parent_pooled = parent_pooled.unsqueeze(0)

        # 3. Setup scheduler once
        if not self._scheduler_setup:
            mu = get_flux_mu(self.pipe, parent_latent)
            self.pipe.scheduler.set_timesteps(self.num_steps, device=self.device, mu=mu)
            self._scheduler_setup = True

        # 4. Prepare timesteps and sigmas
        timesteps = self.pipe.scheduler.timesteps
        all_sigmas = self.pipe.scheduler.sigmas

        # 5. Generate all new particles jointly so they can repel each other
        new_latents = torch.randn(
            (self.batch_size,) + parent_latent.shape[1:],
            generator=self._torch_rng,
            device=self.device,
            dtype=self.pipe.transformer.dtype
        )
        if hasattr(self.pipe.scheduler, "init_noise_sigma"):
            new_latents = new_latents * self.pipe.scheduler.init_noise_sigma

        new_latents = self._denoise_with_particle_guidance(
            new_latents, parent_embedding, parent_pooled, timesteps, all_sigmas
        )
        torch.cuda.empty_cache()

        embedding_np = parent_embedding.detach().cpu().float().numpy().squeeze(0)
        pooled_np = parent_pooled.detach().cpu().float().numpy().squeeze(0)

        return {
            "latent": new_latents.detach().cpu().float().numpy(),
            "prompt_embedding": np.stack([embedding_np] * self.batch_size),
            "pooled_prompt_embedding": np.stack([pooled_np] * self.batch_size),
        }
    
    def _denoise_with_particle_guidance(
        self,
        latents: torch.Tensor,
        prompt_embedding: torch.Tensor,
        pooled_embedding: torch.Tensor,
        timesteps: torch.Tensor,
        all_sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the Flux denoising loop with particle guidance.

        All particles are denoised jointly so pairwise repulsion can be computed.

        Args:
            latents: Noisy latents [N, seq_len, 64]
            prompt_embedding: T5 text embeddings [1, seq_len, 4096]
            pooled_embedding: CLIP pooled embeddings [1, 768]
            timesteps: Scheduler timesteps
            all_sigmas: Sigma values from scheduler

        Returns:
            Denoised latents [N, seq_len, 64]
        """
        device = latents.device
        N = latents.shape[0]

        # Expand embeddings to match particle batch size
        prompt_emb_batch = prompt_embedding.expand(N, -1, -1).contiguous()  # [N, seq_len, 4096]
        pooled_emb_batch = pooled_embedding.expand(N, -1).contiguous()      # [N, 768]

        for i, t in enumerate(timesteps):
            guidance = torch.full(
                [1], self.guidance_scale,
                device=device,
                dtype=torch.float32
            ).expand(N)

            with torch.no_grad():
                v = self.pipe.transformer(
                    hidden_states=latents,
                    timestep=t.expand(N).to(latents.dtype) / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_emb_batch,
                    encoder_hidden_states=prompt_emb_batch,
                    txt_ids=self.text_ids,
                    img_ids=self.latent_img_ids,
                    return_dict=False,
                )[0]

            sigma = all_sigmas[i]

            if self.use_dino:
                if sigma > self.dino_sigma_threshold:
                    grad_phi = compute_dino_gradients_flux(
                        latents,
                        self.pipe,
                        self.dino_model,
                        self.text_ids,
                        self.latent_img_ids,
                        prompt_emb_batch,
                        pooled_emb_batch,
                        coeff=self.particle_coeff,
                        device=device,
                    )
                else:
                    grad_phi = torch.zeros_like(latents)
            else:
                grad_phi = compute_particle_gradients_flux(
                    latents,
                    v,
                    sigma.item(),
                    coeff=self.particle_coeff,
                    sigma_threshold=self.sigma_threshold,
                    device=device,
                )

            latents = latents + (all_sigmas[i+1] - all_sigmas[i]) * (v - grad_phi)

        return latents
    
    def tell(self, solution, objective, measures, add_info, **fields):
        """Tell the emitter about objective and measures (minimal)."""
        pass
