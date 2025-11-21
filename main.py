import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import gc
import hydra
import logging
import numpy as np
import torch
import wandb
torch.cuda.empty_cache()

from diffusers import (
    AutoPipelineForText2Image,
    DDIMScheduler,
    StableDiffusionPipeline
)
from emitters.custom_gaussian import GaussianDictEmitter
from emitters.latent_emitter import SDLatentEmitter
from emitters.dict_scheduler import DictScheduler
from emitters.sdxl_hidden_layer_emitter import SDXLHiddenLayerEmitter
from functools import partial
from image_utils import save_image
from metrics import CSVLogger, me_metrics_function
from model_utils import (
    get_text_embeddings,
    get_sdxl_prompt_embeddings,
    generate_images_from_latents,
    generate_images_from_latents_sdxl_turbo,
    plot_repertoire_images,
    save_offspring_data,
    save_repertoire,
)

from omegaconf import DictConfig, OmegaConf
from repertoires.dns import DNSDictArchive
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
    
    metrics = dict.fromkeys(["iteration", "qd_score", "coverage", "max_fitness", "time", "evaluations"], np.array([]))
    csv_logger = CSVLogger(
        filename=_metrics_dir + f"mapelites-logs.csv",
        header=list(metrics.keys())
    )
    
    # Set random seed for reproducibility
    generator = generator=torch.Generator(device="cuda").manual_seed(config.algorithm.seed)
    torch.manual_seed(config.algorithm.seed)
    torch.cuda.manual_seed_all(config.algorithm.seed)
    np.random.seed(config.algorithm.seed)
    
    # Load Diffusion Model Pipelines
    logger.warning("--- Loading Models ---")
    
    init_latent = torch.randn(1, 4, 64, 64, device="cuda", generator=generator)
    
    if config.model.name == "sd-1-5":
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", version="fp16").to("cuda")
        scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        solution_schema = {
            "latent": (4, 64, 64),
            "prompt_embedding": (2, 77, 768),
        }
        generate_images_from_latents_fn = partial(generate_images_from_latents,
            num_inference_steps=config.generation.num_inference_steps,
            guidance_scale=config.generation.guidance_scale,
            scheduler=scheduler,
            unet=pipe.unet,
            vae=pipe.vae,
        )        
        
        init_prompt_embedding = get_text_embeddings(
            prompt=config.generation.prompt,
            batch_size=1,
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            with_grad=False,
        ) # [batch_size * 2, 77, 768]
        init_latent *= scheduler.init_noise_sigma
        init_image = generate_images_from_latents_fn(
            latents=init_latent,
            prompt_embedding=init_prompt_embedding
        )
        
        # Save seed image, latent and prompt embedding
        torch.save(init_prompt_embedding.cpu(), os.path.join(_initial_values_dir, "initial_prompt_embedding.pt"))
        torch.save(init_latent.cpu(), os.path.join(_initial_values_dir, "initial_latent.pt"))
        save_image(init_image[0], os.path.join(_initial_values_dir, "initial_image.png"))
        
        # Create initial solution dict
        initial_solution = {
            "latent": init_latent.cpu(),
            "prompt_embedding": init_prompt_embedding.cpu(),
        }
        
        del init_image, init_latent, init_prompt_embedding
        gc.collect()
            
    elif config.model.name == "sdxl-turbo":
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
        pipe.to("cuda")
        solution_schema = {
            "latent": (4, 64, 64),
            "prompt_embedding": (77, 2048),
            "pooled_prompt_embedding": (1280,),
        }
    
        generate_images_from_latents_fn = partial(generate_images_from_latents_sdxl_turbo,
            num_inference_steps=1,
            scheduler=pipe.scheduler,
            unet=pipe.unet,
            vae=pipe.vae,
        )
        
        init_prompt_embedding, init_pooled_prompt_embedding = get_sdxl_prompt_embeddings(
            prompt=config.generation.prompt,
            tokenizer_1=pipe.tokenizer,          
            tokenizer_2=pipe.tokenizer_2,
            text_encoder_1=pipe.text_encoder,    
            text_encoder_2=pipe.text_encoder_2,
            device="cuda"
        )

        init_image = generate_images_from_latents_fn(
            latents=init_latent,
            prompt_embedding=init_prompt_embedding,
            pooled_prompt_embedding=init_pooled_prompt_embedding,
            time_ids=[1024, 1024, 0, 0, 512, 512],
        )
        
        # Save seed image, latent and prompt embedding
        torch.save(init_prompt_embedding.cpu(), os.path.join(_initial_values_dir, "initial_prompt_embedding.pt"))
        torch.save(init_latent.cpu(), os.path.join(_initial_values_dir, "initial_latent.pt"))
        torch.save(init_pooled_prompt_embedding.cpu(), os.path.join(_initial_values_dir, "initial_pooled_prompt_embedding.pt"))
        save_image(init_image[0], os.path.join(_initial_values_dir, "initial_image.png"))
        
        # Create initial solution dict
        initial_solution = {
            "latent": init_latent.cpu(),
            "prompt_embedding": init_prompt_embedding.cpu(),
            "pooled_prompt_embedding": init_pooled_prompt_embedding.cpu(),
        }
        
        del init_image, init_latent, init_prompt_embedding, init_pooled_prompt_embedding    
        gc.collect()
        
    # Load CLIP model for fitness calculations
    clip_tokenizer = CLIPTokenizerFast.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer", dtype=torch.float32, cache_dir="/var/tmp/hf")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True, cache_dir="/var/tmp/hf").to("cuda") # type: ignore
    
    # Set up scoring function
    if config.algorithm.name == "map_elites":
        project_desc = True
    else:
        project_desc = config.descriptor.project_desc
    
    scoring_fn, descriptor_dim = get_scoring_function(
        descriptor_name=config.descriptor.name,
        project_desc=project_desc,
        project_dim=config.descriptor.projection_dim,
        prompt=config.generation.prompt,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        generate_images_from_latents_fn=generate_images_from_latents_fn,
        generator=generator,
        model_name=config.model.name,
        save_dir=_initial_values_dir,
    )
    
    logger.warning("--- Setting up Archive ---")
    # Set up MAP-Elites
    if config.algorithm.name == "map_elites":
        pass # TO DO

    #     if config.algorithm.genotype == "embedding":
    #         init_prompt_embeddings = toggle_prompt_embedding_shape(init_prompt_embeddings) # Convert to [batch_size, 2, 77, 768]
    #         archive = GridArchive(
    #             solution_dim=tuple(init_prompt_embeddings.shape[1:]), # ignore batch dimension
    #             dims=config.algorithm.map_elites.grid_dims,
    #             ranges=config.algorithm.map_elites.grid_ranges,
    #             seed=config.algorithm.seed,
    #         )
    #     elif config.algorithm.genotype == "latent":
    #         archive = GridArchive(
    #             solution_dim=tuple(init_latents.shape[1:]), # ignore batch dimension
    #             dims=config.algorithm.map_elites.grid_dims,
    #             ranges=config.algorithm.map_elites.grid_ranges,
    #             seed=config.algorithm.seed,
    #         )
    
    archive = DNSDictArchive(
        solution_dim=solution_schema, # ignore batch dimension
        measure_dim=descriptor_dim,
        max_size=config.algorithm.max_pop_size,
        fitness_threshold=config.algorithm.fitness_threshold,
        k=config.algorithm.dns.k,
        distance_metric=config.algorithm.dns.distance_metric,
        seed=config.algorithm.seed,
        dtype=np.float32,
    )
     
    logger.warning("--- Setting up Emitter ---")
    if config.emitter.name == "gaussian":
        emitter = GaussianDictEmitter(
            archive=archive,
            initial_solution=initial_solution,
            sigma=config.emitter.gaussian.latents_sigma,
            batch_size=config.emitter.batch_size,
            seed=config.algorithm.seed,
            genotype_key=config.algorithm.genotype
        )
        del initial_solution
        gc.collect()
            
    elif config.emitter.name == "hidden_layer" and config.model.name == "sd-1-5":
        emitter = SDLatentEmitter(
                archive=archive,
                initial_solution=initial_solution,
                unet=pipe.unet,
                scheduler=scheduler,
                batch_size=config.emitter.batch_size,
                seed=config.algorithm.seed,
                mutation_type=config.emitter.hidden_layer.mutation_type,
                guidance_scale=config.generation.guidance_scale,
                num_inference_steps=config.generation.num_inference_steps,
                hook_proportion = config.emitter.hidden_layer.hook_proportion,
                alpha_min=config.emitter.hidden_layer.alpha_min,
                alpha_max=config.emitter.hidden_layer.alpha_max,
                svd_k=config.emitter.hidden_layer.svd_k,
                svd_iters=config.emitter.hidden_layer.svd_iters,
                svd_fd_eps=config.emitter.hidden_layer.svd_fd_eps,
                svd_layer=config.emitter.hidden_layer.svd_layer,
                device=device
        )   
    
    elif config.emitter.name == "prompt_embedding" and config.model.name == "sd-1-5":
        pass # TO DO
    
    elif config.emitter.name == "hidden_layer" and config.model.name == "sdxl-turbo":
        emitter = SDXLHiddenLayerEmitter(
            archive=archive,
            initial_solution=initial_solution,
            mutation_key=config.algorithm.genotype,
            mutation_type=config.emitter.hidden_layer.mutation_type,
            batch_size=config.emitter.batch_size,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            alpha_min=config.emitter.hidden_layer.alpha_min,
            alpha_max=config.emitter.hidden_layer.alpha_max,
            sigma=config.emitter.hidden_layer.sigma,
            svd_k=config.emitter.hidden_layer.svd_k,
            svd_iters=config.emitter.hidden_layer.svd_iters,
            svd_fd_eps=config.emitter.hidden_layer.svd_fd_eps,
            svd_layer=config.emitter.hidden_layer.svd_layer,
            seed=config.algorithm.seed,  
            device=device,  
        )
    

    del initial_solution
    gc.collect()

    scheduler = DictScheduler(archive, [emitter])
            
    if config.algorithm.name == "map_elites" or config.algorithm.name == "dns":
        logger.warning("--- Algorithm initialisation ---")
        for itr in range(1, config.algorithm.max_iterations + 1):
            solutions = scheduler.ask()
            
            fitnesses, descriptors, images = scoring_fn(solutions)
            save_offspring_data(solutions, fitnesses, descriptors, images, itr, config, _offspring_dir, _images_dir, config.model.name)
            scheduler.tell(fitnesses, descriptors)
            
            current_metrics = me_metrics_function(scheduler.archive, itr, config)
            wandb.log(current_metrics)
            csv_logger.log(current_metrics)
            logger.warning(f"--- ITERATION: {itr} ---")
            logger.warning(f"QD Score: {current_metrics['qd_score']:.3f}")
            logger.warning(f"Coverage: {current_metrics['coverage']:.3f}%")
            logger.warning(f"Max Fitness: {current_metrics['max_fitness']:.3f}")
            
            if itr % config.save_repertoire_period == 0:
                save_repertoire(archive, itr, _repertoire_dir, config)
                plot_repertoire_images(
                    archive,
                    itr,
                    config,
                    generate_images_from_latents_fn,
                    _images_dir
                    )
        
        save_repertoire(archive, "final", _repertoire_dir)
        plot_repertoire_images(
            archive,
            "final",
            config,
            generate_images_from_latents_fn,
            _images_dir
            )

            
if __name__ == "__main__":
    main() 