import numpy as np
import torch
from functools import partial
from scoring_function import fitness_function
from model_utils import (
    generate_images_from_latents,
    generate_images_from_latents_sdxl_turbo,
    expand_prompt_embedding,
)
from ribs.emitters._emitter_base import EmitterBase
from typing import Callable, Dict, Optional


class AdamDictEmitter(EmitterBase):
    """
    Gradient-based emitter that optimizes a selected genotype key with Adam.

    - Works with dictionary solutions (same schema as DNS/CVT archives).
    - Uses a provided PyTorch loss_fn that returns a scalar given a batch tensor.
    - Other genotype keys are broadcast/copied from the parent batch.
    """

    def __init__(
        self,
        archive,
        *,
        initial_solution: Dict[str, np.ndarray],
        mutation_key: str,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int,
        lr: float = 1e-3,
        steps: int = 1,
        micro_batch_size: Optional[int] = None,
        device: str = "cuda",
        seed: Optional[int] = None,
        bounds=None,
        emitter_id: int = 0,
    ):
        self._solution_schema = archive.solution_dim
        super().__init__(archive=archive, solution_dim=(), bounds=bounds, lower_bounds=None, upper_bounds=None)

        if initial_solution is None:
            raise ValueError("initial_solution must be provided.")
        if mutation_key not in initial_solution:
            raise ValueError(f"mutation_key '{mutation_key}' not in initial_solution")

        # Store seed solution as numpy for deterministic replay
        self._seed_sol = {}
        for k, v in initial_solution.items():
            if isinstance(v, torch.Tensor):
                self._seed_sol[k] = v.detach().cpu().numpy()
            else:
                self._seed_sol[k] = np.array(v)

        self.mutation_key = mutation_key
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.lr = lr
        self.steps = steps
        self.micro_batch_size = micro_batch_size if micro_batch_size is not None else min(batch_size, 4)
        self._device = device
        self._emitter_id = emitter_id

        self._rng = np.random.default_rng(seed)
        self._torch_dtype = torch.float32

        # Initialize persistent parameter and optimizer with seed mutation tensor
        seed_mut = self._seed_sol[self.mutation_key]
        if seed_mut.ndim == len(self._solution_schema[self.mutation_key]):
            seed_mut = seed_mut[np.newaxis, ...]
        if seed_mut.shape[0] < self.batch_size:
            reps = [self.batch_size] + [1] * (seed_mut.ndim - 1)
            seed_mut = np.repeat(seed_mut, reps[0], axis=0)
        self._param = torch.nn.Parameter(
            torch.tensor(seed_mut.astype(np.float32), device=self._device, dtype=self._torch_dtype, requires_grad=True)
        )
        self._optim = torch.optim.Adam([self._param], lr=self.lr)

    # ------------------------------------------------------------------
    def _broadcast_parent(self, parent_batch: np.ndarray, target_shape: tuple):
        """Ensure parent_batch has leading batch dim == batch_size."""
        arr = parent_batch
        if arr.ndim == len(target_shape):
            arr = arr[np.newaxis, ...]
        if arr.shape[0] < self.batch_size:
            reps = [self.batch_size] + [1] * (arr.ndim - 1)
            arr = np.repeat(arr, reps[0], axis=0)
        return arr.astype(np.float32)

    def ask(self):

        # 1) Select parents
        if self.archive.empty:
            parent_sol = {}
            for k, v in self._seed_sol.items():
                target_shape = (self.batch_size, *self._solution_schema[k])
                parent_sol[k] = self._broadcast_parent(v, self._solution_schema[k])
        else:
            sampled = self.archive.sample_elites(self._emitter_id)["solution"]
            parent_sol = {}
            for k, v in sampled.items():
                target_shape = (self.batch_size, *self._solution_schema[k])
                parent_sol[k] = self._broadcast_parent(v, self._solution_schema[k])


        # 2) Gradient steps
        for _ in range(self.steps):
            self._optim.zero_grad()
            loss = self.loss_fn(self._param)
            if loss.dim() != 0:
                loss = loss.mean()
            loss.backward()
            self._optim.step()

        updated_np = self._param.detach().cpu().numpy()

        # 3) Build children dict
        children = {}
        for k, v in parent_sol.items():
            if k == self.mutation_key:
                children[k] = updated_np
            else:
                children[k] = v
        
        # Free memory
        torch.cuda.empty_cache()

        return children

    def tell(self, solution, objective, measures, add_info, **fields):
        # Adam emitter does not need feedback to update internal state
        return None



def prepare_adam_loss_fn(
    config,
    genotype,
    prompt,
    initial_solution,
    pipe,
    scheduler,
    clip_model,
    clip_tokenizer,
    device,
):
    if config.model.name == "sd-1-5":
        # Prepare fixed tensors (batch size always 1 for Adam)
        latents_init = torch.tensor(np.array(initial_solution["latent"]), device=device, dtype=pipe.unet.dtype)
        prompt_embeds_init = torch.tensor(np.array(initial_solution["prompt_embedding"]), device=device, dtype=pipe.unet.dtype)
        
        # Define loss function based on mutation key
        if genotype == "latent":
            def adam_loss_fn(genotypes: torch.Tensor) -> torch.Tensor:
                # genotypes shape: (1, 4, 64, 64) -> squeeze to (1, 4, 64, 64) (keep batch dim for latents)
                # Actually latents need the batch dimension, so no squeeze needed
                generated_images = generate_images_from_latents(
                    latents=genotypes,
                    prompt_embedding=prompt_embeds_init,
                    num_inference_steps=config.model.num_inference_steps,
                    guidance_scale=config.model.guidance_scale,
                    scheduler=scheduler,
                    unet=pipe.unet,
                    vae=pipe.vae,
                    with_grad=True,
                )
                fitnesses = fitness_function(
                    generated_images=generated_images,
                    prompt=prompt,
                    clip_model=clip_model,
                    clip_tokenizer=clip_tokenizer,
                    with_grad=True,
                )
                return -fitnesses.mean()
        
        elif genotype == "prompt_embedding":
            def adam_loss_fn(genotypes: torch.Tensor) -> torch.Tensor:
                # genotypes shape: (1, 2, 77, 768) -> squeeze to (2, 77, 768)
                # For SD-1.5, prompt_embedding should be (2, 77, 768) for guidance
                if genotypes.shape[0] == 1:
                    genotypes = genotypes.squeeze(0)
                
                generated_images = generate_images_from_latents(
                    latents=latents_init,
                    prompt_embedding=genotypes,
                    num_inference_steps=config.model.num_inference_steps,
                    guidance_scale=config.model.guidance_scale,
                    scheduler=scheduler,
                    unet=pipe.unet,
                    vae=pipe.vae,
                    with_grad=True,
                )
                fitnesses = fitness_function(
                    generated_images=generated_images,
                    prompt=prompt,
                    clip_model=clip_model,
                    clip_tokenizer=clip_tokenizer,
                    with_grad=True,
                )
                return -fitnesses.mean()
        
        else:
            raise ValueError(f"Unsupported genotype '{genotype}' for SD-1.5 Adam emitter.")

    elif config.model.name == "sdxl-turbo":
        
        # Prepare fixed embeddings/latents and time_ids
        latents_init = torch.tensor(np.array(initial_solution["latent"]), device=device, dtype=pipe.unet.dtype)
        prompt_embeds_init = torch.tensor(np.array(initial_solution["prompt_embedding"]), device=device, dtype=pipe.unet.dtype)
        pooled_embeds_init = torch.tensor(np.array(initial_solution["pooled_prompt_embedding"]), device=device, dtype=pipe.unet.dtype)

        time_ids = torch.tensor([[1024, 1024, 0, 0, 512, 512]], device=device, dtype=pipe.unet.dtype)

        # Define loss function based on mutation key
        if genotype == "latent":
            
            def adam_loss_fn(genotypes: torch.Tensor) -> torch.Tensor:
                generated_images = generate_images_from_latents_sdxl_turbo(
                    latents=genotypes,
                    prompt_embedding=prompt_embeds_init,
                    pooled_prompt_embedding=pooled_embeds_init,
                    time_ids=time_ids,
                    num_inference_steps=config.model.num_inference_steps,
                    scheduler=pipe.scheduler,
                    unet=pipe.unet,
                    vae=pipe.vae,
                    with_grad=True,
                    device=device,
                )
                fitnesses = fitness_function(
                    generated_images=generated_images,
                    prompt=prompt,
                    clip_model=clip_model,
                    clip_tokenizer=clip_tokenizer,
                    with_grad=True,
                )
                return -fitnesses.mean()

        elif genotype == "prompt_embedding":
            def adam_loss_fn(genotypes: torch.Tensor) -> torch.Tensor:
                generated_images = generate_images_from_latents_sdxl_turbo(
                    latents=latents_init,
                    prompt_embedding=genotypes,
                    pooled_prompt_embedding=pooled_embeds_init,
                    time_ids=time_ids,
                    num_inference_steps=config.model.num_inference_steps,
                    scheduler=pipe.scheduler,
                    unet=pipe.unet,
                    vae=pipe.vae,
                    with_grad=True,
                    device=device,
                )
                fitnesses = fitness_function(
                    generated_images=generated_images,
                    prompt=prompt,
                    clip_model=clip_model,
                    clip_tokenizer=clip_tokenizer,
                    with_grad=True,
                )
                return -fitnesses.mean()

        elif genotype == "pooled_prompt_embedding":
            def adam_loss_fn(genotypes: torch.Tensor) -> torch.Tensor:
                generated_images = generate_images_from_latents_sdxl_turbo(
                    latents=latents_init,
                    prompt_embedding=prompt_embeds_init,
                    pooled_prompt_embedding=genotypes,
                    time_ids=time_ids,
                    num_inference_steps=config.model.num_inference_steps,
                    scheduler=pipe.scheduler,
                    unet=pipe.unet,
                    vae=pipe.vae,
                    with_grad=True,
                    device=device,
                )
                fitnesses = fitness_function(
                    generated_images=generated_images,
                    prompt=prompt,
                    clip_model=clip_model,
                    clip_tokenizer=clip_tokenizer,
                    with_grad=True,
                )
                return -fitnesses.mean()
        else:
            raise ValueError(f"Unsupported genotype '{genotype}' for SDXL-Turbo Adam emitter.")

    else:
        raise NotImplementedError(f"Adam emitter not supported for model {config.model.name}.")
    
    return adam_loss_fn