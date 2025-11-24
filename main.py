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
    AutoPipelineForText2Image,
    DDIMScheduler,
    StableDiffusionPipeline
)
from emitters.custom_gaussian import GaussianDictEmitter
from emitters.hidden_layer_emitter import SDHiddenLayerEmitter
from emitters.dict_scheduler import DictScheduler
from emitters.random_seed import RandomSeedEmitter
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
        name=f"{config.algorithm}",
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
    
    metrics = dict.fromkeys(["iteration",
        "qd_score",
        "coverage",
        "max_fitness",
        "time",
        "evaluations",
        "avg_pairwise_dist",
        "avg_dissimilarity_ssim",
        "total_variance",
        "top_10_fitness_avg",
    ], np.array([]))
    csv_logger = CSVLogger(
        filename=_metrics_dir + f"mapelites-logs.csv",
        header=list(metrics.keys())
    )
    
    # Set random seed for reproducibility
    generator = generator=torch.Generator(device=device).manual_seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    
    # Load Diffusion Model Pipelines
    logger.warning("--- Loading Models ---")
    
    init_latent = torch.randn(1, 4, 64, 64, device=device, generator=generator)
        
    if config.model.name == "sd-1-5":
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
        scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        solution_schema = {
            "latent": (4, 64, 64),
            "prompt_embedding": (2, 77, 768),
            "image": (3, 512, 512),
        }
        generate_images_from_latents_fn = partial(generate_images_from_latents,
            num_inference_steps=config.model.num_inference_steps,
            guidance_scale=config.model.guidance_scale,
            scheduler=scheduler,
            unet=pipe.unet,
            vae=pipe.vae,
        )        
        
        init_prompt_embedding = get_text_embeddings(
            prompt=config.prompt,
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
        pipe.to(device)
        solution_schema = {
            "latent": (4, 64, 64),
            "prompt_embedding": (77, 2048),
            "pooled_prompt_embedding": (1280,),
            "image": (3, 512, 512),
        }
    
        generate_images_from_latents_fn = partial(generate_images_from_latents_sdxl_turbo,
            num_inference_steps=config.model.num_inference_steps,
            scheduler=pipe.scheduler,
            unet=pipe.unet,
            vae=pipe.vae,
        )
        
        init_prompt_embedding, init_pooled_prompt_embedding = get_sdxl_prompt_embeddings(
            prompt=config.prompt,
            tokenizer_1=pipe.tokenizer,          
            tokenizer_2=pipe.tokenizer_2,
            text_encoder_1=pipe.text_encoder,    
            text_encoder_2=pipe.text_encoder_2,
            device=device
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
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True, cache_dir="/var/tmp/hf").to(device) # type: ignore
    
    # Set up scoring function
    scoring_fn, descriptor_dim = get_scoring_function(
        descriptor_name=config.descriptor,
        project_desc=config.algorithm.project_desc,
        project_dim=config.algorithm.projection_dim,
        prompt=config.prompt,
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
    
    elif config.algorithm.name == "dns":
        archive = DNSDictArchive(
            solution_dim=solution_schema, # ignore batch dimension
            measure_dim=descriptor_dim,
            max_size=config.max_pop_size,
            fitness_threshold=config.fitness_threshold,
            k=config.algorithm.k,
            distance_metric=config.algorithm.distance_metric,
            seed=config.seed,
            dtype=np.float32,
        )
     
    logger.warning("--- Setting up Emitter ---")
    
    if config.emitter.type == "gaussian":
        emitter = GaussianDictEmitter(
            archive=archive,
            initial_solution=initial_solution,
            sigma=config.emitter.sigma,
            batch_size=config.batch_size,
            seed=config.seed,
            genotype_key=config.emitter.genotype
        )
            
    elif config.emitter.type == "hidden_layer" and config.model.name == "sd-1-5":
        if config.emitter.model_type != "sd-1-5":
            assert False, "Mismatch between emitter.model_type and model used"
            
        emitter = SDHiddenLayerEmitter(
                archive=archive,
                initial_solution=initial_solution,
                mutation_key=config.emitter.genotype,
                mutation_type=config.emitter.mutation_type,
                unet=pipe.unet,
                scheduler=scheduler,
                batch_size=config.batch_size,
                seed=config.seed,
                guidance_scale=config.model.guidance_scale,
                num_inference_steps=config.model.num_inference_steps,
                hook_proportion = config.emitter.hook_proportion,
                alpha_min=config.emitter.alpha_min,
                alpha_max=config.emitter.alpha_max,
                sigma=config.emitter.sigma,
                svd_k=config.emitter.svd_k,
                svd_iters=config.emitter.svd_iters,
                svd_fd_eps=config.emitter.svd_fd_eps,
                svd_layer=config.emitter.svd_layer,
                device=device
        )   
    
    elif config.emitter.type == "hidden_layer" and config.model.name == "sdxl-turbo":
        if config.emitter.model_type != "sdxl-turbo":
            assert False, "Mismatch between emitter.model_type and model used"
            
        emitter = SDXLHiddenLayerEmitter(
            archive=archive,
            initial_solution=initial_solution,
            mutation_key=config.emitter.genotype,
            mutation_type=config.emitter.mutation_type,
            batch_size=config.batch_size,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            alpha_min=config.emitter.alpha_min,
            alpha_max=config.emitter.alpha_max,
            sigma=config.emitter.sigma,
            svd_k=config.emitter.svd_k,
            svd_iters=config.emitter.svd_iters,
            svd_fd_eps=config.emitter.svd_fd_eps,
            svd_layer=config.emitter.svd_layer,
            seed=config.seed,  
            device=device,  
        )
    elif config.emitter.type == "prompt_embedding" and config.model.name == "sd-1-5":
        pass # TO DO

    elif config.emitter.type == "random_seeds":
        emitter = RandomSeedEmitter(
            archive=archive,
            initial_solution=initial_solution,
            batch_size=config.batch_size,
            seed=config.seed,
        )

    del initial_solution
    gc.collect()

    scheduler = DictScheduler(archive, [emitter])
            
    if config.algorithm.name == "map_elites" or config.algorithm.name == "dns":
        logger.warning("--- Algorithm initialisation ---")
        for itr in range(1, config.max_iterations + 1):
            solutions = scheduler.ask()
            
            fitnesses, descriptors, images = scoring_fn(solutions)
            save_offspring_data(solutions, fitnesses, descriptors, images, itr, config, _offspring_dir, _images_dir, config.model)
            scheduler.tell(fitnesses, descriptors, image=images)
            
            current_metrics = me_metrics_function(scheduler.archive, itr, config, config.ssim_sample_size)
            wandb.log(current_metrics)
            csv_logger.log(current_metrics)
            logger.warning(f"--- ITERATION: {itr} ---")
            logger.warning(f"QD Score: {current_metrics['qd_score']:.3f}")
            logger.warning(f"Max Fitness: {current_metrics['max_fitness']:.3f}")
            logger.warning(f"Top 10 Avg Fitness: {current_metrics['top_10_fitness_avg']:.3f}")
            logger.warning(f"SSIM Diversity: {current_metrics['avg_dissimilarity_ssim']:.3f}")
            logger.warning(f"Avg Pairwise Descriptor Distance: {current_metrics['avg_pairwise_dist']:.3f}")
            
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