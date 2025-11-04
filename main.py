import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import gc
import hydra
import logging
import numpy as np
import torch
import wandb
torch.cuda.empty_cache()

from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline
)
from emitters.custom_gaussian import CustomGaussianEmitter
from functools import partial
from image_utils import (
    save_image,
)
from metrics import CSVLogger, me_metrics_function
from model_utils import (
    toggle_prompt_embedding_shape,
    get_text_embeddings,
    generate_images_from_latents,
    save_offspring_data,
)

from omegaconf import DictConfig, OmegaConf
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler
from scoring_function import get_scoring_function
from transformers import (
    CLIPModel,
    CLIPTokenizerFast,
)

os.environ["WANDB_MODE"] = "disabled"

@hydra.main(config_path="config/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig) -> None:
    
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().handlers[0].setLevel(logging.INFO)
    logger = logging.getLogger(f"{__name__}")

        
    # Debug logging for directories
    output_dir = hydra.core.hydra_config.HydraConfig.get().run.dir
    logger.warning(f"Hydra output directory: {output_dir}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialise wandb
    wandb.init(
        project=f"Latent Exploration",
        name=f"{config.algorithm.name}",
        config=OmegaConf.to_container(config, resolve=True),
    )

    # Make directories for saving data
    _initial_values_dir = os.path.join(output_dir, "initial_values/")
    _images_dir = os.path.join(output_dir, "images/",)
    _metrics_dir = os.path.join(output_dir, "metrics/")
    _repertoire_dir = os.path.join(output_dir, "repertoire/")
    _offspring_dir = os.path.join(output_dir, "offspring/")

    os.makedirs(_initial_values_dir, exist_ok=True)
    os.makedirs(_images_dir, exist_ok=True)
    os.makedirs(_metrics_dir, exist_ok=True)
    os.makedirs(_repertoire_dir, exist_ok=True)
    os.makedirs(_offspring_dir, exist_ok=True)
    os.makedirs(_images_dir + f"init/", exist_ok=True)
    os.makedirs(_images_dir + f"final/", exist_ok=True)
    
    metrics = dict.fromkeys(["iteration", "qd_score", "coverage", "max_fitness", "time", "evaluations"], np.array([]))
    csv_logger = CSVLogger(
        filename=_metrics_dir + f"mapelites-logs.csv",
        header=list(metrics.keys())
    )
    # Load Diffusion Model Pipelines
    logger.warning("--- Loading Models ---")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    clip_tokenizer = CLIPTokenizerFast.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer", torch_dtype=torch.float32, cache_dir="/var/tmp/hf")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True, cache_dir="/var/tmp/hf").to("cuda") # type: ignore
    
    # Set random seed for reproducibility
    generator = generator=torch.Generator(device="cuda").manual_seed(config.algorithm.seed)
    torch.manual_seed(config.algorithm.seed)
    torch.cuda.manual_seed_all(config.algorithm.seed)
    np.random.seed(config.algorithm.seed)
    
    # Set up scoring function
    generate_images_from_latents_fn = partial(generate_images_from_latents,
        num_inference_steps=config.generation.num_inference_steps,
        guidance_scale=config.generation.guidance_scale,
        scheduler=scheduler,
        unet=pipe.unet,
        vae=pipe.vae,
    )
    
    scoring_fn = get_scoring_function(
        descriptor_name=config.descriptor.name,
        project_desc=config.descriptor.project_desc,
        project_dim=config.descriptor.projection_dim,
        prompt=config.generation.prompt,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        generate_images_from_latents_fn=generate_images_from_latents_fn,
        generator=generator,
        save_dir=_initial_values_dir,
    )
    
    # Create initial images 
    init_prompt_embeddings = get_text_embeddings(
        prompt=config.generation.prompt,
        batch_size=config.emitter.batch_size,
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        with_grad=False,
    ) # [batch_size * 2, 77, 768]
    torch.save(init_prompt_embeddings.cpu(), os.path.join(_initial_values_dir, "initial_prompt_embeddings.pt"))
    
    init_latents = torch.randn(1, 4, 64, 64, device="cuda", generator=generator)
    init_latents = init_latents.repeat(config.emitter.batch_size, 1, 1, 1)
    init_latents *= scheduler.init_noise_sigma
    torch.save(init_latents.cpu(), os.path.join(_initial_values_dir, "initial_latents.pt"))
    
    init_fitnesses, init_descriptors, init_images = scoring_fn(init_latents, init_prompt_embeddings)

    torch.save(init_fitnesses, os.path.join(_initial_values_dir, "initial_fitnesses.pt"))
    torch.save(init_descriptors, os.path.join(_initial_values_dir, "initial_descriptors.pt"))
    for i, img in enumerate(init_images):
        save_image(img, os.path.join(_images_dir, f"init/initial_image_{i}.png"))
        
    # Set up MAP-Elites
    if config.algorithm.name == "map_elites":

        if config.algorithm.genotype == "embedding":
            init_prompt_embeddings = toggle_prompt_embedding_shape(init_prompt_embeddings) # Convert to [batch_size, 2, 77, 768]
            archive = GridArchive(
                solution_dim=tuple(init_prompt_embeddings.shape[1:]), # ignore batch dimension
                dims=config.algorithm.map_elites.grid_dims,
                ranges=config.algorithm.map_elites.grid_ranges,
                seed=config.algorithm.seed,
            )
        elif config.algorithm.genotype == "latents":
            archive = GridArchive(
                solution_dim=tuple(init_latents.shape[1:]), # ignore batch dimension
                dims=config.algorithm.map_elites.grid_dims,
                ranges=config.algorithm.map_elites.grid_ranges,
                seed=config.algorithm.seed,
            )
      
    if config.emitter.name == "gaussian":
        if config.algorithm.genotype == "embedding":
            emitter = CustomGaussianEmitter(
                archive=archive,
                initial_solutions=init_prompt_embeddings.cpu().numpy(),
                sigma=config.emitter.gaussian.embedding_sigma,
                batch_size=config.emitter.batch_size,
                seed=config.algorithm.seed,
            )
        elif config.algorithm.genotype == "latents":
            emitter = CustomGaussianEmitter(
                archive=archive,
                initial_solutions=init_latents.cpu().numpy().squeeze(),
                sigma=config.emitter.gaussian.latents_sigma,
                batch_size=config.emitter.batch_size,
                seed=config.algorithm.seed,
            )
            
    scheduler = Scheduler(archive, [emitter])
            
    if config.algorithm.name == "map_elites":
        logger.warning("--- Algorithm initialisation ---")
        for itr in range(1, config.algorithm.map_elites.max_iterations + 1):
            solutions = scheduler.ask()
            
            if config.algorithm.genotype == "embedding":
                solutions = toggle_prompt_embedding_shape(solutions) # Convert to [batch_size * 2, 77, 768]
                fitnesses, descriptors, images = scoring_fn(init_latents, solutions)
                save_offspring_data(solutions, fitnesses, descriptors, images, itr, config, _offspring_dir, _images_dir)
                
            elif config.algorithm.genotype == "latents":
                fitnesses, descriptors, images = scoring_fn(solutions, init_prompt_embeddings.cpu().detach())
                save_offspring_data(solutions, fitnesses, descriptors, images, itr, config, _offspring_dir, _images_dir)
                        
            scheduler.tell(fitnesses, descriptors)
            
            current_metrics = me_metrics_function(scheduler.archive, itr, config)
            wandb.log(current_metrics)
            csv_logger.log(current_metrics)
            logger.warning(f"--- ITERATION: {itr} ---")
            logger.warning(f"QD Score: {current_metrics['qd_score']:.3f}")
            logger.warning(f"Coverage: {current_metrics['coverage']:.3f}%")
            logger.warning(f"Max Fitness: {current_metrics['max_fitness']:.3f}")

            
if __name__ == "__main__":
    main() 