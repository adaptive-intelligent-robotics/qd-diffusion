import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
import gc
import hydra
import logging
import numpy as np
import random
import torch
import wandb
torch.cuda.empty_cache()

# Make CUDA/cuDNN deterministic and disable TF32 for reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
# Enforce deterministic algorithms where supported; warn instead of raising if an op is not
torch.use_deterministic_algorithms(True, warn_only=True)
from emitters.adam_emitter import AdamDictEmitter, prepare_adam_loss_fn
from emitters.custom_sep_cma_es import CMAESDictEmitter
from emitters.custom_gaussian import GaussianDictEmitter
from emitters.custom_isoline import IsoLineDictEmitter
from emitters.flux_hidden_layer_emitter import FluxHiddenLayerEmitter
from emitters.hidden_layer_emitter import SDHiddenLayerEmitter
from emitters.imagebreeder import ImageBreederEmitter
from emitters.dict_scheduler import DictScheduler
from emitters.qdhf_models import get_qdhf_archive, update_qdhf_diversity_model
from emitters.random_seed import RandomSeedEmitter
from emitters.sdxl_hidden_layer_emitter import SDXLHiddenLayerEmitter
from metrics import CSVLogger, me_metrics_function
from model_utils import (
    plot_repertoire_images,
    save_offspring_data,
    save_repertoire,
    setup_models,
)

from omegaconf import DictConfig, OmegaConf
from prompts import prompts_dict
from repertoires.cvtme import CVTDictArchive
from repertoires.dns import DNSDictArchive
from repertoires.passive_population import PassivePopulationArchive
from scoring_function import get_scoring_function
from transformers import (
    CLIPModel,
    CLIPTokenizerFast,
)


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
    
    # Load in prompt
    if config.prompt=="":
        prompt = prompts_dict[config.prompt_key]
    else:
        prompt = config.prompt
        
    def make_safe_tag(text, max_bytes=64):
        # Normalize unicode characters to standard ASCII where possible
        text = text.replace("’", "'").replace("“", '"').replace("”", '"')
        
        # Encode to bytes, truncate, and decode back, ignoring incomplete chars
        encoded = text.encode('utf-8')[:max_bytes]
        return encoded.decode('utf-8', 'ignore').strip()

    # Usage
    prompt_tag = make_safe_tag(prompt)   
     
    # Initialise wandb
    wandb.init(
        project=f"Latent Exploration",
        name=f"{config.algorithm.name}",
        tags=[prompt_tag],
        config=OmegaConf.to_container(config, resolve=True),
        mode=config.wandb_mode,
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
        "avg_fitness",
        "coverage",
        "max_fitness",
        "time",
        "evaluations",
        "avg_pairwise_dist",
        "total_variance",
        f"top_{config.top_k}_fitness_avg",
        f"top_{config.top_k}_avg_pairwise_dist",
        f"top_{config.top_k}_total_variance",
        f"top_{config.top_k}_avg_dissimilarity_ssim",
    ], np.array([]))
    
    csv_logger = CSVLogger(
        filename=_metrics_dir + f"mapelites-logs.csv",
        header=list(metrics.keys())
    )
    
    logger.warning(f"----- PROMPT: {prompt} -----")
    
    # Set random seed for reproducibility
    generator = generator=torch.Generator(device=device).manual_seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Load Diffusion Model Pipelines
    logger.warning("--- Loading Models ---")
            
    pipe, scheduler, generate_images_from_latents_fn, solution_schema, initial_solution, text_ids, latent_img_ids = setup_models(
        config,
        prompt,
        generator,
        _initial_values_dir,
        device,
    )   
    
    # Load CLIP model for fitness calculations
    clip_tokenizer = CLIPTokenizerFast.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer", dtype=torch.float32, cache_dir="/var/tmp/hf")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True, cache_dir="/var/tmp/hf").to(device) # type: ignore
    clip_model.eval()
    
    # Set up scoring function
    scoring_fn, descriptor_dim = get_scoring_function(
        descriptor_name=config.descriptor,
        prompt=prompt,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        generate_images_from_latents_fn=generate_images_from_latents_fn,
        generator=generator,
        model_name=config.model.name,
        save_dir=_initial_values_dir,
        config=config,
    )
    # Calculate batch sizes for multi-emitters
    if config.emitter.genotype == "multi":
        if config.model.name == "sd-1-5":
            batch_sizes = [config.batch_size // 2, config.batch_size - (config.batch_size // 2)]
        elif config.model.name == "sdxl-turbo" or config.model.name == "flux":
            batch_sizes = [config.batch_size // 3, config.batch_size // 3, config.batch_size - 2 * (config.batch_size // 3)]
    else:
        batch_sizes = [config.batch_size]
        
    logger.warning("--- Setting up Archive ---")
    
    # Set up MAP-Elites
    if config.algorithm.name == "cvt-map-elites":
        ranges = [[config.algorithm.min_descriptor, config.algorithm.max_descriptor] for _ in range(descriptor_dim)]
        archive = CVTDictArchive(
            solution_dim=solution_schema, # ignore batch dimension
            cells=config.max_pop_size,
            ranges=ranges,
            samples=100*config.max_pop_size,
            seed=config.seed,
            dtype=np.float32,
        )
    
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
    
    elif config.algorithm.name == "qdhf":
        qdhf_update_schedule = [# 1 by default below, define additional updates according to qdhf paper
            int(0.1 * config.max_iterations),
            int(0.25 * config.max_iterations),
            int(0.5 * config.max_iterations),
        ]
        logger.warning ("--- Setting up QDHF Model ---")
        
        diversity_model, warmup_clip_features = update_qdhf_diversity_model(
            None,
            initial_solution,
            scoring_fn,
            config,
            return_clip_features=True
        )
        logger.warning("--- Setting up QDHF Archive ---")
        archive = get_qdhf_archive(
            initial_solution,
            scoring_fn,
            diversity_model,
            solution_schema,
            config, 
            warmup_clip_features
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
    elif config.emitter.type == "multi-gaussian":
        
        emitter_1 = GaussianDictEmitter(
            archive=archive,
            initial_solution=initial_solution,
            sigma=config.emitter.latent_sigma,
            batch_size=batch_sizes[0],
            seed=config.seed,
            genotype_key="latent"
        )
        
        emitter_2 = GaussianDictEmitter(
            archive=archive,
            initial_solution=initial_solution,
            sigma=config.emitter.prompt_embedding_sigma,
            batch_size=batch_sizes[1],
            seed=config.seed,
            genotype_key="prompt_embedding"
        )
        emitters = [emitter_1, emitter_2]
        
        if config.model.name == "sdxl-turbo" or config.model.name == "flux":
            emitter_3 = GaussianDictEmitter(
                archive=archive,
                initial_solution=initial_solution,
                sigma=config.emitter.pooled_prompt_embedding_sigma,
                batch_size=batch_sizes[2],
                seed=config.seed,
                genotype_key="pooled_prompt_embedding"
            )
            emitters.append(emitter_3)
    
    elif config.emitter.type == "isoline":
        emitter = IsoLineDictEmitter(
            archive=archive,
            initial_solution=initial_solution,
            iso_sigma=config.emitter.iso_sigma,
            line_sigma=config.emitter.line_sigma,
            batch_size=config.batch_size,
            seed=config.seed,
            genotype_key=config.emitter.genotype
        )
    
    elif config.emitter.type == "multi-isoline":
        emitter_1 = IsoLineDictEmitter(
            archive=archive,
            initial_solution=initial_solution,
            iso_sigma=config.emitter.latent_iso_sigma,
            line_sigma=config.emitter.latent_line_sigma,
            batch_size=batch_sizes[0],
            seed=config.seed,
            genotype_key="latent"
        )
        
        emitter_2 = IsoLineDictEmitter(
            archive=archive,
            initial_solution=initial_solution,
            iso_sigma=config.emitter.prompt_embedding_iso_sigma,
            line_sigma=config.emitter.prompt_embedding_line_sigma,
            batch_size=batch_sizes[1],
            seed=config.seed,
            genotype_key="prompt_embedding"
        )
        emitters = [emitter_1, emitter_2]
        
        if config.model.name == "sdxl-turbo" or config.model.name == "flux":
            emitter_3 = IsoLineDictEmitter(
                archive=archive,
                initial_solution=initial_solution,
                iso_sigma=config.emitter.pooled_prompt_embedding_iso_sigma,
                line_sigma=config.emitter.pooled_prompt_embedding_line_sigma,
                batch_size=batch_sizes[2],
                seed=config.seed,
                genotype_key="pooled_prompt_embedding"
            )
            emitters.append(emitter_3)
            
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
    elif config.emitter.type == "hidden_layer" and config.model.name == "flux":
        if config.emitter.model_type != "flux":
            assert False, "Mismatch between emitter.model_type and model used"
            
        emitter = FluxHiddenLayerEmitter(
            pipe=pipe,
            archive=archive,
            initial_solution=initial_solution,
            text_ids=text_ids,
            latent_img_ids=latent_img_ids,
            num_inference_steps=config.model.num_inference_steps,
            guidance_scale=config.model.guidance_scale,
            mutation_key=config.emitter.genotype,
            mutation_type=config.emitter.mutation_type,
            batch_size=config.batch_size,
            hook_proportion=(config.emitter.hook_min, config.emitter.hook_max),
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

    elif config.emitter.type == "hidden_layer_multi":
        if config.model.name == "sd-1-5":
            batch_size_1 = int(config.batch_size//2)
            batch_size_2 = config.batch_size - batch_size_1
                        
            emitter_1 = SDHiddenLayerEmitter(
                archive=archive,
                initial_solution=initial_solution,
                mutation_key="latent",
                mutation_type=config.emitter.mutation_type,
                unet=pipe.unet,
                scheduler=scheduler,
                batch_size=batch_size_1,
                seed=config.seed,
                guidance_scale=config.model.guidance_scale,
                num_inference_steps=config.model.num_inference_steps,
                hook_proportion = config.emitter.hook_proportion,
                alpha_min=config.emitter.latent_alpha_min,
                alpha_max=config.emitter.latent_alpha_max,
                sigma=config.emitter.latent_sigma,
                svd_k=config.emitter.svd_k,
                svd_iters=config.emitter.svd_iters,
                svd_fd_eps=config.emitter.svd_fd_eps,
                svd_layer=config.emitter.svd_layer,
                device=device
            ) 
                        
            emitter_2 = SDHiddenLayerEmitter(
                archive=archive,
                initial_solution=initial_solution,
                mutation_key="prompt_embedding",
                mutation_type=config.emitter.mutation_type,
                unet=pipe.unet,
                scheduler=scheduler,
                batch_size=batch_size_2,
                seed=config.seed,
                guidance_scale=config.model.guidance_scale,
                num_inference_steps=config.model.num_inference_steps,
                hook_proportion = config.emitter.hook_proportion,
                alpha_min=config.emitter.prompt_embedding_alpha_min,
                alpha_max=config.emitter.prompt_embedding_alpha_max,
                sigma=config.emitter.prompt_embedding_sigma,
                svd_k=config.emitter.svd_k,
                svd_iters=config.emitter.svd_iters,
                svd_fd_eps=config.emitter.svd_fd_eps,
                svd_layer=config.emitter.svd_layer,
                device=device
            ) 
            
            emitters = [emitter_1, emitter_2]
            
        elif config.model.name == "sdxl-turbo":
            batch_size_1 = int(config.batch_size // 3)
            batch_size_2 = int(config.batch_size // 3)
            batch_size_3 = config.batch_size - batch_size_1 - batch_size_2
            
            emitter_1 = SDXLHiddenLayerEmitter(
                archive=archive,
                initial_solution=initial_solution,
                mutation_key="latent",
                mutation_type=config.emitter.mutation_type,
                batch_size=batch_size_1,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                alpha_min=config.emitter.latent_alpha_min,
                alpha_max=config.emitter.latent_alpha_max,
                sigma=config.emitter.latent_sigma,
                svd_k=config.emitter.svd_k,
                svd_iters=config.emitter.svd_iters,
                svd_fd_eps=config.emitter.svd_fd_eps,
                svd_layer=config.emitter.svd_layer,
                seed=config.seed,  
                device=device,  
            )
            emitter_2 = SDXLHiddenLayerEmitter(
                archive=archive,
                initial_solution=initial_solution,
                mutation_key="prompt_embedding",
                mutation_type=config.emitter.mutation_type,
                batch_size=batch_size_2,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                alpha_min=config.emitter.prompt_embedding_alpha_min,
                alpha_max=config.emitter.prompt_embedding_alpha_max,
                sigma=config.emitter.prompt_embedding_sigma,
                svd_k=config.emitter.svd_k,
                svd_iters=config.emitter.svd_iters,
                svd_fd_eps=config.emitter.svd_fd_eps,
                svd_layer=config.emitter.svd_layer,
                seed=config.seed,  
                device=device,  
            )
            emitter_3 = SDXLHiddenLayerEmitter(
                archive=archive,
                initial_solution=initial_solution,
                mutation_key="pooled_prompt_embedding",
                mutation_type=config.emitter.mutation_type,
                batch_size=batch_size_3,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                alpha_min=config.emitter.pooled_prompt_embedding_alpha_min,
                alpha_max=config.emitter.pooled_prompt_embedding_alpha_max,
                sigma=config.emitter.pooled_prompt_embedding_sigma,
                svd_k=config.emitter.svd_k,
                svd_iters=config.emitter.svd_iters,
                svd_fd_eps=config.emitter.svd_fd_eps,
                svd_layer=config.emitter.svd_layer,
                seed=config.seed,  
                device=device,  
            )
            emitters = [emitter_1, emitter_2, emitter_3]
        
        elif config.model.name == "flux":
            batch_size_1 = int(config.batch_size // 3)
            batch_size_2 = int(config.batch_size // 3)
            batch_size_3 = config.batch_size - batch_size_1 - batch_size_2
            
            emitter_1 = FluxHiddenLayerEmitter(
                pipe=pipe,
                archive=archive,
                initial_solution=initial_solution,
                text_ids=text_ids,
                latent_img_ids=latent_img_ids,
                num_inference_steps=config.model.num_inference_steps,
                guidance_scale=config.model.guidance_scale,
                mutation_key="latent",
                mutation_type=config.emitter.mutation_type,
                batch_size=batch_size_1,
                hook_proportion=(config.emitter.hook_min, config.emitter.hook_max),
                alpha_min=config.emitter.latent_alpha_min,
                alpha_max=config.emitter.latent_alpha_max,
                sigma=config.emitter.latent_sigma,
                svd_k=config.emitter.svd_k,
                svd_iters=config.emitter.svd_iters,
                svd_fd_eps=config.emitter.svd_fd_eps,
                svd_layer=config.emitter.svd_layer,
                seed=config.seed,  
                device=device,  
            )
            emitter_2 = FluxHiddenLayerEmitter(
                pipe=pipe,
                archive=archive,
                initial_solution=initial_solution,
                text_ids=text_ids,
                latent_img_ids=latent_img_ids,
                num_inference_steps=config.model.num_inference_steps,
                guidance_scale=config.model.guidance_scale,
                mutation_key="prompt_embedding",
                mutation_type=config.emitter.mutation_type,
                batch_size=batch_size_2,
                hook_proportion=(config.emitter.hook_min, config.emitter.hook_max),
                alpha_min=config.emitter.prompt_embedding_alpha_min,
                alpha_max=config.emitter.prompt_embedding_alpha_max,
                sigma=config.emitter.prompt_embedding_sigma,
                svd_k=config.emitter.svd_k,
                svd_iters=config.emitter.svd_iters,
                svd_fd_eps=config.emitter.svd_fd_eps,
                svd_layer=config.emitter.svd_layer,
                seed=config.seed,  
                device=device,  
            )
            emitter_3 = FluxHiddenLayerEmitter(
                pipe=pipe,
                archive=archive,
                initial_solution=initial_solution,
                text_ids=text_ids,
                latent_img_ids=latent_img_ids,
                num_inference_steps=config.model.num_inference_steps,
                guidance_scale=config.model.guidance_scale,
                mutation_key="pooled_prompt_embedding",
                mutation_type=config.emitter.mutation_type,
                batch_size=batch_size_3,
                hook_proportion=(config.emitter.hook_min, config.emitter.hook_max),
                alpha_min=config.emitter.pooled_prompt_embedding_alpha_min,
                alpha_max=config.emitter.pooled_prompt_embedding_alpha_max,
                sigma=config.emitter.pooled_prompt_embedding_sigma,
                svd_k=config.emitter.svd_k,
                svd_iters=config.emitter.svd_iters,
                svd_fd_eps=config.emitter.svd_fd_eps,
                svd_layer=config.emitter.svd_layer,
                seed=config.seed,  
                device=device,  
            )
            emitters = [emitter_1, emitter_2, emitter_3]
            
    elif config.emitter.type == "random_seeds":
        emitter = RandomSeedEmitter(
            archive=archive,
            initial_solution=initial_solution,
            batch_size=config.batch_size,
            seed=config.seed,
        )
    
    elif config.emitter.type == "imagebreeder":
        # Need to seed imagebreeder with 2 images to introduce variation
        for k in initial_solution:
            initial_solution[k] = initial_solution[k].repeat(2, *([1]*(initial_solution[k].ndim - 1)))
            new_noise = torch.randn(1, 4, 64, 64, device=device, generator=generator)
            if config.model.name == "sd-1-5":
                new_noise = new_noise * scheduler.init_noise_sigma
            initial_solution["latent"][1] = new_noise.cpu()
        
        emitter = ImageBreederEmitter(
            archive=archive,
            initial_solution=initial_solution,
            mutation_type=config.emitter.name, 
            batch_size=config.batch_size,
            seed=config.seed
        )
        
    elif config.emitter.type == "adam":
        if config.emitter.genotype != "multi":
            max_pop_size = 1
            batch_sizes = [1]
        else:
            if config.model.name == "sd-1-5":
                max_pop_size = 2
                batch_sizes = [1, 1]
            elif config.model.name == "sdxl-turbo" or config.model.name == "flux":
                max_pop_size = 3
                batch_sizes = [1, 1, 1]
        
        archive = PassivePopulationArchive( # overwrite DNS archive
            solution_dim=solution_schema,
            measure_dim=descriptor_dim,
            population_size=max_pop_size,
            seed=config.seed,
            dtype=np.float32,
            batch_sizes=batch_sizes,
        )
        
        # Second archive to track what would be added to DNS if we kept all solutions
        passive_dns_archive = DNSDictArchive(
            solution_dim=solution_schema,
            measure_dim=descriptor_dim,
            max_size=config.max_pop_size,
            fitness_threshold=config.fitness_threshold,
            k=config.algorithm.k,
            distance_metric=config.algorithm.distance_metric,
            seed=config.seed,
            dtype=np.float32,
        )
        
        # Enable gradient checkpointing to reduce memory usage
        if hasattr(pipe.vae, 'enable_gradient_checkpointing'):
            pipe.vae.enable_gradient_checkpointing()
        
        if hasattr(pipe, 'unet'):
            if hasattr(pipe.unet, 'enable_gradient_checkpointing'):
                pipe.unet.enable_gradient_checkpointing()
        
        if hasattr(pipe, 'transformer'):
            if hasattr(pipe.transformer, 'enable_gradient_checkpointing'):
                pipe.transformer.enable_gradient_checkpointing()

        if config.emitter.genotype != "multi":
            adam_loss_fn = prepare_adam_loss_fn(
                config,
                config.emitter.genotype,
                prompt,
                initial_solution,
                pipe,
                scheduler,
                clip_model,
                clip_tokenizer,
                device
            )
            emitter = AdamDictEmitter( # TO DO: add setup for this
                archive=archive,
                initial_solution=initial_solution,
                mutation_key=config.emitter.genotype,
                loss_fn=adam_loss_fn,
                batch_size=1,
                lr=config.emitter.learning_rate,
                steps=config.emitter.steps,
                micro_batch_size=1,
                device=device,
                seed=config.seed,
                emitter_id=0,
            )
            
        else:
            adam_loss_fn_1 = prepare_adam_loss_fn(
                config,
                "latent",
                prompt,
                initial_solution,
                pipe,
                scheduler,
                clip_model,
                clip_tokenizer,
                device
            )
            adam_loss_fn_2 = prepare_adam_loss_fn(
                config,
                "prompt_embedding",
                prompt,
                initial_solution,
                pipe,
                scheduler,
                clip_model,
                clip_tokenizer,
                device,
            )
            emitter_1 = AdamDictEmitter(
                archive=archive,
                initial_solution=initial_solution,
                mutation_key="latent",
                loss_fn=adam_loss_fn_1,
                batch_size=1,
                lr=config.emitter.learning_rate,
                steps=config.emitter.steps,
                micro_batch_size=1,
                device=device,
                seed=config.seed,
                emitter_id=0,
            )
            emitter_2 = AdamDictEmitter(
                archive=archive,
                initial_solution=initial_solution,
                mutation_key="prompt_embedding",
                loss_fn=adam_loss_fn_2,
                batch_size=1,
                lr=config.emitter.learning_rate,
                steps=config.emitter.steps,
                micro_batch_size=1,
                device=device,
                seed=config.seed,
                emitter_id=1,
            )
            emitters = [emitter_1, emitter_2]
            
            if config.model.name == "sdxl-turbo" or config.model.name == "flux":
                adam_loss_fn_3 = prepare_adam_loss_fn(
                    config,
                    "pooled_prompt_embedding",
                    prompt,
                    initial_solution,
                    pipe,
                    scheduler,
                    clip_model,
                    clip_tokenizer,
                    device
                )
                emitter_3 = AdamDictEmitter(
                    archive=archive,
                    initial_solution=initial_solution,
                    mutation_key="pooled_prompt_embedding",
                    loss_fn=adam_loss_fn_3,
                    batch_size=1,
                    lr=config.emitter.learning_rate,
                    steps=config.emitter.steps,
                    micro_batch_size=1,
                    device=device,
                    seed=config.seed,
                    emitter_id=2,
                )
                emitters.append(emitter_3)
    
    elif config.emitter.type == "sep-cma-es":
        # Doesn't interact with the main archive, so create a passive population archive to store solutions
        archive = PassivePopulationArchive(
            solution_dim=solution_schema,
            measure_dim=descriptor_dim,
            population_size=config.batch_size, 
            seed=config.seed,
            dtype=np.float32,
            batch_sizes=batch_sizes, 
        )
        
        passive_dns_archive = DNSDictArchive(
            solution_dim=solution_schema,
            measure_dim=descriptor_dim,
            max_size=config.max_pop_size,
            fitness_threshold=config.fitness_threshold,
            k=config.algorithm.k,
            distance_metric=config.algorithm.distance_metric,
            seed=config.seed,
            dtype=np.float32,
        )
        if config.emitter.genotype != "multi":
            # Single Emitter Case
            emitter = CMAESDictEmitter(
                archive=archive,
                initial_solution=initial_solution,
                mutation_key=config.emitter.genotype, # e.g. "latent"
                sigma0=config.emitter.sigma,         # e.g. 0.1 or 1.0
                batch_size=config.batch_size,
                num_parents=config.batch_size // 2,
                seed=config.seed,
                bounds=None # Add [lower, upper] list if needed
            )
        else:
            emitter_1 = CMAESDictEmitter(
                archive=archive,
                initial_solution=initial_solution,
                mutation_key="latent",
                sigma0=config.emitter.latent_sigma,
                batch_size=batch_sizes[0],
                num_parents=batch_sizes[0] // 2,
                seed=config.seed,
                bounds=None
            )
            emitter_2 = CMAESDictEmitter(
                archive=archive,
                initial_solution=initial_solution,
                mutation_key="prompt_embedding",
                sigma0=config.emitter.prompt_embedding_sigma,
                batch_size=batch_sizes[1],
                num_parents=batch_sizes[1] // 2,
                seed=config.seed,
                bounds=None
            )
            emitters = [emitter_1, emitter_2]
            if config.model.name == "sdxl-turbo" or config.model.name == "flux":
                emitter_3 = CMAESDictEmitter(
                    archive=archive,
                    initial_solution=initial_solution,
                    mutation_key="pooled_prompt_embedding",
                    sigma0=config.emitter.pooled_prompt_embedding_sigma,
                    batch_size=batch_sizes[2],
                    num_parents=batch_sizes[2] // 2,
                    seed=config.seed,
                    bounds=None
                )
                emitters.append(emitter_3)
                
    else:
        assert False, "Emitter type not recognised"
                    
    
    if config.emitter.genotype != "multi":
        scheduler = DictScheduler(archive, [emitter])
    else:
        scheduler = DictScheduler(archive, emitters)


    logger.warning("--- Algorithm initialisation ---")
    if config.emitter.type == "adam":
        if config.emitter.genotype != "multi":
            max_iters = config.max_iterations * config.batch_size
            save_repertoire_period = config.save_repertoire_period * config.batch_size
        else:
            max_iters = np.ceil(config.max_iterations * config.batch_size / len(emitters))
            save_repertoire_period = np.ceil(config.save_repertoire_period * config.batch_size / len(emitters))
    else:
        max_iters = config.max_iterations
        save_repertoire_period = config.save_repertoire_period
           
    for itr in range(1, int(max_iters) + 1):
        
        if config.algorithm.name == "qdhf":
            if itr in qdhf_update_schedule:
                logger.warning(f"--- Updating QDHF Diversity Model at iteration {itr} ---")
                diversity_model = update_qdhf_diversity_model(
                    archive, 
                    initial_solution,
                    scoring_fn,
                    config,
                )
                new_archive = get_qdhf_archive(
                    archive.data()["solution"],
                    scoring_fn,
                    diversity_model,
                    solution_schema,
                    config,
                )

                if config.emitter.genotype != "multi":
                    scheduler = DictScheduler(new_archive, [emitter])
                else:
                    scheduler = DictScheduler(new_archive, emitters)  
                          
        solutions = scheduler.ask()
        
        if config.algorithm.name != "qdhf":
            fitnesses, descriptors, images = scoring_fn(solutions)
            save_offspring_data(solutions, fitnesses, descriptors, images, itr, config, _offspring_dir, _images_dir, config.model)
            scheduler.tell(fitnesses, descriptors, image=images)

        elif config.algorithm.name == "qdhf":
            dns_fitnesses, fitnesses, dns_descriptors, clip_descriptors, images = scoring_fn(solutions)
            with torch.no_grad():
                descriptors = diversity_model(clip_descriptors)
            save_offspring_data(solutions, dns_fitnesses, dns_descriptors, images, itr, config, _offspring_dir, _images_dir, config.model)
            scheduler.tell(
                fitnesses,
                descriptors,
                image=images,
                dns_fitness=dns_fitnesses,
                dns_descriptor=dns_descriptors
            )
        
        if config.emitter.type in ["adam", "sep-cma-es"]:
            solutions_to_add = solutions.copy()
            solutions_to_add["image"] = images.detach().cpu().numpy()  # Add images to solutions dict
            passive_dns_archive.add(
                    solutions_to_add, 
                    fitnesses, 
                    descriptors
            )
            current_metrics = me_metrics_function(passive_dns_archive, itr, config)
            
        else:
            current_metrics = me_metrics_function(scheduler.archive, itr, config)
            
        wandb.log(current_metrics)
        csv_logger.log(current_metrics)
        logger.warning(f"--- ITERATION: {itr} ---")
        logger.warning(f"---- Evaluations: {current_metrics['evaluations']}")
        logger.warning(f"Max Fitness: {current_metrics['max_fitness']:.3f}")
        logger.warning(f"QD Score: {current_metrics['qd_score']:.3f}")
        logger.warning(f"Avg Descriptor Distance: {current_metrics['avg_pairwise_dist']:.3f}")
        logger.warning(f"Top {config.top_k} Avg Fitness: {current_metrics[f'top_{config.top_k}_fitness_avg']:.3f}")
        logger.warning(f"Top {config.top_k} Avg Descriptor Distance: {current_metrics[f'top_{config.top_k}_avg_pairwise_dist']:.3f}")
        
        if itr % save_repertoire_period == 0:
            if config.emitter.type in ["adam", "sep-cma-es"]:
                save_repertoire(passive_dns_archive, itr, _repertoire_dir, config)
                plot_repertoire_images(
                    passive_dns_archive,
                    itr,
                    config,
                    generate_images_from_latents_fn,
                    _images_dir
                    )
            else:
                save_repertoire(scheduler.archive, itr, _repertoire_dir, config)
                plot_repertoire_images(
                    scheduler.archive,
                    itr,
                    config,
                    generate_images_from_latents_fn,
                    _images_dir
                    )
    
    if config.emitter.type in ["adam", "sep-cma-es"]:
        save_repertoire(passive_dns_archive, "final", _repertoire_dir, config)
        plot_repertoire_images(
            passive_dns_archive,
            "final",
            config,
            generate_images_from_latents_fn,
            _images_dir
            )
    else:
        save_repertoire(scheduler.archive, "final", _repertoire_dir, config)
        plot_repertoire_images(
            scheduler.archive,
            "final",
            config,
            generate_images_from_latents_fn,
            _images_dir
            )

            
if __name__ == "__main__":
    main() 