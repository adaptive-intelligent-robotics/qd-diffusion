import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import gc
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
torch.cuda.empty_cache()

from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    LMSDiscreteScheduler
)
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    CLIPTextModel,
    CLIPTokenizerFast,
)
from feature_functions import (
    brightness,
    average_hue,
    get_artclip_score_model,
    dino_feature_extractor
)

from model_utils import (
    get_text_embeddings,
    generate_images_from_latents,
    fitness_function,
    save_images,
    save_final_images
)

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from qdax.baselines.cmaes import CMAES
from qdax.core.emitters.mutation_operators import isoline_variation, slerp_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, MapElitesRepertoire
from qdax.core.map_elites import MAPElites
from qdax.utils.metrics import default_qd_metrics, CSVLogger
from qdax.utils.plotting import plot_map_elites_results
from typing import Callable, List
import time
from tqdm import tqdm


@hydra.main(config_path="config/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig) -> None:

    # Dynamically set environment variable based on the algorithm
    if config.algorithm.name == "adam":
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
    else:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().handlers[0].setLevel(logging.INFO)
    logger = logging.getLogger(f"{__name__}")

    # Debug logging for directories
    logger.warning(f"Current working directory: {os.getcwd()}")
    logger.warning(f"Original working directory: {hydra.utils.get_original_cwd()}")
    
    # Get Hydra's output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().run.dir
    logger.warning(f"Hydra output directory: {output_dir}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialise wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"Latent Exploration",
        name=f"{config.algorithm.name}",
        # track hyperparameters and run metadata
        config=OmegaConf.to_container(config, resolve=True),
    )

    _images_dir = os.path.join(output_dir, "images/",)
    _metrics_dir = os.path.join(output_dir, "metrics/")
    _repertoire_dir = os.path.join(output_dir, "repertoire/")


    os.makedirs(_images_dir, exist_ok=True)
    os.makedirs(_metrics_dir, exist_ok=True)
    os.makedirs(_repertoire_dir, exist_ok=True)
    os.makedirs(_images_dir + f"init/", exist_ok=True)
    os.makedirs(_images_dir + f"final/", exist_ok=True)

    # Load Diffusion Model Pipelines
    logger.warning("--- Loading Models ---")
    vae = AutoencoderKL.from_pretrained(config.model.name, subfolder="vae", torch_dtype=torch.float32).cuda()
    tokenizer = CLIPTokenizerFast.from_pretrained(config.model.name, subfolder="tokenizer", torch_dtype=torch.float32)
    text_encoder = CLIPTextModel.from_pretrained(config.model.name, subfolder="text_encoder", torch_dtype=torch.float32).cuda().eval()
    unet = UNet2DConditionModel.from_pretrained(config.model.name, subfolder="unet", torch_dtype=torch.float32).cuda()
    unet.enable_gradient_checkpointing()
    scheduler = LMSDiscreteScheduler.from_pretrained(config.model.name, subfolder="scheduler", torch_dtype=torch.float32)
    
    # Initialize random key
    key = jax.random.PRNGKey(config.algorithm.seed)
    
    # Initialise scoring function
    generate_images_from_latents_fn = partial(generate_images_from_latents,
        num_inference_steps=config.generation.num_inference_steps,
        guidance_scale=config.generation.guidance_scale,
        scheduler=scheduler,
        unet=unet,
        vae=vae,
    )
    
    if config.algorithm.name == "adam":
        batch_size = config.algorithm.adam.batch_size
        with_grad = config.algorithm.adam.with_grad
    elif config.algorithm.name == "cma_es":
        batch_size = config.algorithm.cma_es.population_size
        with_grad = config.algorithm.cma_es.with_grad
    elif config.algorithm.name == "map_elites":
        batch_size = config.algorithm.map_elites.batch_size
        with_grad = config.algorithm.map_elites.with_grad
    
    init_prompt_embedding = get_text_embeddings(
        prompt=config.generation.prompt,
        batch_size=1,  # type: ignore
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        with_grad=False,  # No gradients needed for initial image generation
    )
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda") # type: ignore
    
    descriptor_functions = []
    
    if config.algorithm.descriptor_type == "dino":
        dino_model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(device)
        dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        dino_fun =  partial(
            dino_feature_extractor,
            processor=dino_processor,
            model=dino_model,
            projection_matrix=torch.randn(2, 768, device=device),
            device=device
        )
        descriptor_functions.append(dino_fun)
    
    else:
        for d in config.algorithm.descriptor_names:
            if d == "brightness":
                descriptor_functions.append(brightness)
            elif d == "average_hue":
                descriptor_functions.append(average_hue)
            elif d == "artclip_score":
                descriptor_functions.append(get_artclip_score_model(d))
            elif d == "theme_and_logic":
                descriptor_functions.append(get_artclip_score_model(d))
            elif d == "creativity":
                descriptor_functions.append(get_artclip_score_model(d))
            elif d == "layout_and_composition":
                descriptor_functions.append(get_artclip_score_model(d))
            elif d == "space_and_perspective":
                descriptor_functions.append(get_artclip_score_model(d))
            elif d == "sense_of_order":
                descriptor_functions.append(get_artclip_score_model(d))
            elif d == "light_and_shadow":
                descriptor_functions.append(get_artclip_score_model(d))
            elif d == "color":
                descriptor_functions.append(get_artclip_score_model(d))
            elif d == "details_and_texture":
                descriptor_functions.append(get_artclip_score_model(d))
            elif d == "overall":
                descriptor_functions.append(get_artclip_score_model(d))
            elif d == "mood":
                descriptor_functions.append(get_artclip_score_model(d))
            else:
                raise ValueError(f"Descriptor function {d} not found")
            

    def scoring_fn(latents,
        prompt_embedding,
        key,
        prompt,
        clip_model,
        clip_tokenizer,
        generate_images_from_latents_fun,
        descriptor_functions: List[Callable],
        descriptor_type: str,
    ):
        
        # Generate images
        generated_images = generate_images_from_latents_fun(latents, prompt_embedding)
              
        fitnesses = fitness_function(generated_images=generated_images,
            prompt=prompt,
            clip_model=clip_model,
            clip_tokenizer=clip_tokenizer,
        )
        
        # Convert generated images to numpy
        generated_images_np = np.array(generated_images.cpu())
                
        descriptors = []
        
        for d_fun in descriptor_functions:
            descriptor = jnp.array([d_fun(image) for image in generated_images_np])           
            descriptors.append(descriptor)
        
        if descriptor_type == "dino":
            descriptors = jnp.atleast_2d(jnp.array(descriptors).squeeze()) # Shape: (batch_size, num_descriptors)     
        else:
            descriptors = jnp.stack(descriptors, axis=-1)
           
        return fitnesses, descriptors, {}

    scoring_function = partial(scoring_fn,
        prompt=config.generation.prompt,
        clip_model=clip_model,
        clip_tokenizer=tokenizer,
        generate_images_from_latents_fun=generate_images_from_latents_fn,
        descriptor_functions=descriptor_functions,
        descriptor_type=config.algorithm.descriptor_type,
    )

    # Compute the MAP-Elites centroids
    key, subkey = jax.random.split(key)
    logger.warning("--- Generating Centroids ---")

    centroids = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=config.algorithm.map_elites.num_init_cvt_samples,
        num_centroids=config.algorithm.map_elites.num_centroids,
        minval=config.algorithm.map_elites.min_descriptor,
        maxval=config.algorithm.map_elites.max_descriptor,
        key=subkey,
    )
        
    map_elites_metrics_fn = partial(default_qd_metrics, qd_offset=0.0)
        
    if config.algorithm.name == "cma_es":

        logger.warning("--- Algorithm initialisation ---")

        
        def cma_es_scoring_fn(
            genotypes, 
            key,
            batch_size,
            prompt_embedding,
        ):
            image_genotypes = jnp.reshape(genotypes, (batch_size, 4, 64, 64))
        
            fitnesses, descriptors, _ = scoring_function(image_genotypes, prompt_embedding, key)
                                    
            return fitnesses.squeeze(), descriptors.squeeze()
        
        init_prompt_embeddings = init_prompt_embedding.repeat(batch_size, 1, 1)
        
        partial_fitness_fn = partial(cma_es_scoring_fn,
            batch_size=config.algorithm.cma_es.population_size,
            prompt_embedding=init_prompt_embeddings,
        )

        # Create empty repertoire with default values
        passive_repertoire = MapElitesRepertoire.init_default(
            genotype=jnp.zeros((4, 64, 64)), 
            centroids=centroids
        )
            
        # Initialize CMA-ES
        key, subkey = jax.random.split(key)
        cmaes = CMAES(
            population_size=config.algorithm.cma_es.population_size,
            num_best=config.algorithm.cma_es.num_best,
            search_dim=4*64*64,
            fitness_function=partial_fitness_fn,
            mean_init=jax.random.normal(subkey, shape=(4*64*64,), dtype=jnp.float32),
            init_sigma=config.algorithm.cma_es.sigma,
            delay_eigen_decomposition=config.algorithm.cma_es.delay_eigen_decomposition,
        )
        
        state = cmaes.init()
        
        # Save initial mean image
        init_samples_latents = jnp.reshape(state.mean, (1, 4, 64, 64))
        init_images = generate_images_from_latents_fn(init_samples_latents, init_prompt_embedding)
        save_images(init_images, iteration="init", folder=_images_dir + f"init/")
        
        metrics = dict.fromkeys(["iteration", "qd_score", "coverage", "max_fitness", "time", "evaluations"], jnp.array([]))
        csv_logger = CSVLogger(
            filename=_metrics_dir + f"cmaes-logs.csv",
            header=list(metrics.keys())
        )

        # Log initial metrics
        current_metrics = map_elites_metrics_fn(passive_repertoire)
        current_metrics["iteration"] = 0
        current_metrics["time"] = 0.0
        current_metrics["evaluations"] = 0.0
        wandb.log(current_metrics)
        csv_logger.log(current_metrics)
        logger.warning(f"--- ITERATION: {0} ---")
        logger.warning(f"QD Score: {current_metrics['qd_score']:.3f}")
        logger.warning(f"Coverage: {current_metrics['coverage']:.3f}%")
        logger.warning(f"Max Fitness: {current_metrics['max_fitness']:.3f}")       
            
        # Run optimization
        for i in tqdm(range(config.algorithm.cma_es.max_iterations)):
            start_time = time.time()
                      
            # Sample, sample_size = population_size
            key, subkey = jax.random.split(key)
            samples = cmaes.sample(state, subkey)
                                    
            # udpate
            key, subkey = jax.random.split(key)
            state, fitnesses, descriptors = cmaes.update(state, samples, subkey)
                        
            # Add initial samples
            passive_repertoire = passive_repertoire.add(
                jnp.reshape(samples, (config.algorithm.cma_es.population_size, 4, 64, 64)),
                descriptors,
                fitnesses
            )
            current_metrics = map_elites_metrics_fn(passive_repertoire)
                        
            timelapse = time.time() - start_time

            # Metrics
            current_metrics["iteration"] = i+1
            current_metrics["time"] = timelapse
            current_metrics["evaluations"] = (i+1) * config.algorithm.cma_es.population_size
            
            # Log
            wandb.log(current_metrics)
            csv_logger.log(current_metrics)
            logger.warning(f"--- ITERATION: {i+1} ---")
            logger.warning(f"CMA-ES Population best fitness: {jnp.max(fitnesses):.3f}")
            logger.warning(f"QD Score: {current_metrics['qd_score']:.3f}")
            logger.warning(f"Coverage: {current_metrics['coverage']:.3f}%")
            logger.warning(f"Max Fitness: {current_metrics['max_fitness']:.3f}")       
            
            # Concatenate
            current_metrics = jax.tree.map(lambda x: jnp.array([x]), current_metrics)
            metrics = jax.tree.map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

            # check stop condition
            stop_condition = cmaes.stop_condition(state)

            if stop_condition:
                break


        logger.warning(f"Num iterations before stop condition: {i}")  

        # Plot results
        fig, axes = plot_map_elites_results(env_steps=range(config.algorithm.cma_es.max_iterations),
            metrics=metrics,
            repertoire=passive_repertoire,
            min_descriptor=config.algorithm.map_elites.min_descriptor,
            max_descriptor=config.algorithm.map_elites.max_descriptor
        )
        fig.savefig(_images_dir + f"passive_repertoire-results.png")
        
        # Save final repertoire
        np.savez(_repertoire_dir + f"final_passive_repertoire.npz", 
            genotypes=passive_repertoire.genotypes,
            descriptors=passive_repertoire.descriptors,
            fitnesses=passive_repertoire.fitnesses,
            centroids=centroids
        )
        
        # Save final images
        if config.algorithm.variation.genotype == "embedding":
            save_embedding = None
            save_latents = init_latent
            
        elif config.algorithm.variation.genotype == "latents":
            save_embedding = init_prompt_embedding
            save_latents = None
            
        save_final_images(repertoire=passive_repertoire,
            num_inference_steps=config.generation.num_inference_steps,
            guidance_scale=config.generation.guidance_scale,
            scheduler=scheduler,
            unet=unet,
            vae=vae,
            n_sample=5,
            folder=_images_dir + f"final/",
            init_embedding=save_embedding,
            init_latent=save_latents,
        )
        
    elif config.algorithm.name == "adam":
        logger.warning("--- Algorithm initialisation ---")
        
        init_prompt_embeddings = init_prompt_embedding.repeat(batch_size, 1, 1)
        
        generate_images_from_latents_with_grad_fn = partial(generate_images_from_latents,
            prompt_embedding=init_prompt_embeddings,
            num_inference_steps=config.generation.num_inference_steps,
            guidance_scale=config.generation.guidance_scale,
            scheduler=scheduler,
            unet=unet,
            vae=vae,
            with_grad=True  # Enable gradients for image generation
        )
                    
        def adam_loss_fn(
            genotypes,
            generate_images_from_latents_fun,
            prompt,
        ):
            
            # Generate images
            generated_images = generate_images_from_latents_fun(genotypes)
            fitnesses = fitness_function(generated_images=generated_images,
                prompt=prompt,
                clip_model=clip_model,
                clip_tokenizer=tokenizer,
                with_grad=True  # Enable gradients for fitness calculation
            )
            
            return -fitnesses.squeeze().mean()
        
        loss_fn = partial(adam_loss_fn,
            generate_images_from_latents_fun=generate_images_from_latents_with_grad_fn,
            prompt=config.generation.prompt,
        )
        
        # Create empty repertoire with default values
        passive_repertoire = MapElitesRepertoire.init_default(
            genotype=jnp.zeros((4, 64, 64)), 
            centroids=centroids
        )
        
        # Initialise genotype as torch tensor
        key, subkey = jax.random.split(key)  # JAX PRNG handling

        init_genotypes = jax.random.normal(subkey, shape=(config.algorithm.adam.batch_size, 4, 64, 64), dtype=jnp.float32)
        init_genotypes_np = np.array(init_genotypes)  # ← assuming JAX to NumPy here
        init_genotypes_torch = torch.nn.Parameter(
            torch.tensor(init_genotypes_np, dtype=torch.float32, device='cuda', requires_grad=True)
        )

        # Save initial images
        init_images = generate_images_from_latents_fn(init_genotypes, init_prompt_embeddings)
        save_images(init_images, iteration="init", folder=_images_dir + f"init/")
        
        # Evaluate initial population and add to passive repertoire
        key, subkey = jax.random.split(key)
        fitnesses, descriptors, _ = scoring_function(init_genotypes, init_prompt_embeddings, subkey)
        passive_repertoire = passive_repertoire.add(
            jnp.reshape(init_genotypes, (config.algorithm.adam.batch_size, 4, 64, 64)),
            descriptors,
            fitnesses
        )                    
        del init_genotypes, init_genotypes_np, init_images, fitnesses, descriptors, centroids
            
        # Initialize Adam optimizer
        adam_optimiser = torch.optim.Adam([init_genotypes_torch], lr=config.algorithm.adam.learning_rate)

        # Initialize metrics
        metrics = dict.fromkeys(["iteration", "loss", "qd_score", "coverage", "max_fitness", "time", "evaluations"], jnp.array([]))

        csv_logger = CSVLogger(
            filename=_metrics_dir + f"adam-logs.csv",
            header=list(metrics.keys())
        )
        
        # Log initial metrics
        current_metrics = map_elites_metrics_fn(passive_repertoire)
        current_metrics["iteration"] = 0
        current_metrics["time"] = 0.0
        current_metrics["evaluations"] = 0.0
        wandb.log(current_metrics)
        csv_logger.log(current_metrics)
        logger.warning(f"--- ITERATION: {0} ---")
        logger.warning(f"QD Score: {current_metrics['qd_score']:.3f}")
        logger.warning(f"Coverage: {current_metrics['coverage']:.3f}%")
        logger.warning(f"Max Fitness: {current_metrics['max_fitness']:.3f}")       
        
        # Run optimization
        for i in range(config.algorithm.adam.max_iterations):
            start_time = time.time()

            # Detach and re-wrap the tensor to prevent backward reuse error
            init_genotypes_torch = torch.nn.Parameter(init_genotypes_torch.detach().clone().requires_grad_())

            # Apply Adam
            adam_optimiser.zero_grad()
            loss = loss_fn(init_genotypes_torch)  
            loss.backward()
            adam_optimiser.step()
            torch.cuda.empty_cache()
            gc.collect()
            
            timelapse = time.time() - start_time

            # Convert torch to jnp
            genotypes_jnp = jnp.array(init_genotypes_torch.detach().cpu().numpy(), dtype=jnp.float32)
            fitnesses, descriptors, _ = scoring_function(genotypes_jnp, init_prompt_embeddings, key)
            passive_repertoire = passive_repertoire.add(
                jnp.reshape(genotypes_jnp, (config.algorithm.adam.batch_size, 4, 64, 64)),
                descriptors,
                fitnesses
            )
            del genotypes_jnp, fitnesses, descriptors
            
            current_metrics = map_elites_metrics_fn(passive_repertoire)
                        
            # Metrics
            current_metrics["iteration"] = i+1
            current_metrics["time"] = timelapse
            current_metrics["loss"] = loss.item()
            current_metrics["evaluations"] = (i+1) * config.algorithm.adam.batch_size
            
            # Log
            wandb.log(current_metrics)
            csv_logger.log(current_metrics)
            logger.warning(f"--- ITERATION: {i+1} ---")
            logger.warning(f"Loss: {loss.item():.4f}")
            logger.warning(f"QD Score: {current_metrics['qd_score']:.3f}")
            logger.warning(f"Coverage: {current_metrics['coverage']:.3f}%")
            logger.warning(f"Max Fitness: {current_metrics['max_fitness']:.3f}")       
                 
            # Concatenate
            current_metrics = jax.tree.map(lambda x: jnp.array([x]), current_metrics)
            metrics = jax.tree.map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

        # Plot results
        fig, axes = plot_map_elites_results(env_steps=range(config.algorithm.adam.max_iterations),
            metrics=metrics,
            repertoire=passive_repertoire,
            min_descriptor=config.algorithm.map_elites.min_descriptor,
            max_descriptor=config.algorithm.map_elites.max_descriptor
        )
        fig.savefig(_images_dir + f"passive_repertoire-results.png")
        
        # Save final repertoire
        np.savez(_repertoire_dir + f"final_passive_repertoire.npz", 
            genotypes=passive_repertoire.genotypes,
            descriptors=passive_repertoire.descriptors,
            fitnesses=passive_repertoire.fitnesses,
            centroids=passive_repertoire.centroids
        )
        
        # Save final images
        if config.algorithm.variation.genotype == "embedding":
            save_embedding = None
            save_latents = init_latent
            
        elif config.algorithm.variation.genotype == "latents":
            save_embedding = init_prompt_embedding
            save_latents = None
            
        save_final_images(repertoire=passive_repertoire,
            num_inference_steps=config.generation.num_inference_steps,
            guidance_scale=config.generation.guidance_scale,
            scheduler=scheduler,
            unet=unet,
            vae=vae,
            n_sample=5,
            folder=_images_dir + f"final/",
            init_embedding=save_embedding,
            init_latent=save_latents,
        )
        
    elif config.algorithm.name == "map_elites":

        logger.warning("--- Algorithm initialisation ---")
    
        key, subkey = jax.random.split(key)  # JAX PRNG handling
        
        # Define a metrics function
        if config.algorithm.variation.variation_type == "isoline":
            variation_fn = partial(
                isoline_variation,
                iso_sigma=config.algorithm.variation.iso_sigma,
                line_sigma=config.algorithm.variation.line_sigma
            )
        
        elif config.algorithm.variation.variation_type == "slerp":
            variation_fn = slerp_variation

        mixing_emitter = MixingEmitter(
            mutation_fn=None,
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=config.algorithm.map_elites.batch_size
        )
        
        init_latent = jax.random.normal(subkey, shape=(1, 4, 64, 64), dtype=jnp.float32)
        
        if config.algorithm.variation.genotype == "embedding":
            init_latents = init_latent.repeat(batch_size, axis=0)  # Repeat for batch size
            init_prompt_embeddings = init_prompt_embedding.repeat(batch_size, 1, 1)
            init_prompt_embeddings = init_prompt_embedding.repeat(batch_size, 1, 1) + \
                torch.randn_like(init_prompt_embedding.repeat(batch_size, 1, 1)) * config.algorithm.variation.iso_sigma
                            
        elif config.algorithm.variation.genotype == "latents":
            init_latents = init_latent.repeat(batch_size, axis=0)
            init_prompt_embeddings = init_prompt_embedding.repeat(batch_size, 1, 1)  
            init_latents += jax.random.normal(
                subkey, shape=init_latents.shape, dtype=jnp.float32
            ) * config.algorithm.variation.iso_sigma
            
            
        init_images = generate_images_from_latents_fn(init_latents, init_prompt_embeddings)
        save_images(init_images, iteration="init", folder=_images_dir + f"init/")

        # Instantiate MAP-Elites
        map_elites = MAPElites(
            scoring_function=None,
            emitter=mixing_emitter,
            metrics_function=map_elites_metrics_fn,
        )
        
        # Evaluate the initial population
        key, subkey = jax.random.split(key)
        
        fitnesses, descriptors, extra_scores = scoring_function(init_latents, init_prompt_embeddings, subkey)
        
        # Compute initial repertoire and emitter state
        if config.algorithm.variation.genotype == "embedding":
            
            embedding_genotypes = init_prompt_embeddings.reshape(batch_size, 2, 77, 1024)

            repertoire, emitter_state, metrics = map_elites.init_ask_tell(
                genotypes=jnp.array(embedding_genotypes.detach().cpu().numpy(), dtype=jnp.float32),
                fitnesses=fitnesses,
                descriptors=descriptors,
                centroids=centroids,
                key=key,
                extra_scores=extra_scores,
            )
            
        elif config.algorithm.variation.genotype == "latents":
            repertoire, emitter_state, metrics = map_elites.init_ask_tell(
                genotypes=init_latents,
                fitnesses=fitnesses,
                descriptors=descriptors,
                centroids=centroids,
                key=key,
                extra_scores=extra_scores,
            )
            
        
        metrics = dict.fromkeys(["iteration", "qd_score", "coverage", "max_fitness", "time", "evaluations"], jnp.array([]))
        csv_logger = CSVLogger(
            filename=_metrics_dir + f"mapelites-logs.csv",
            header=list(metrics.keys())
        )
        
        # Log initial metrics
        current_metrics = map_elites_metrics_fn(repertoire)
        current_metrics["iteration"] = 0.0
        current_metrics["time"] = 0.0
        current_metrics["evaluations"] = config.algorithm.map_elites.batch_size
        wandb.log(current_metrics)
        csv_logger.log(current_metrics)
        logger.warning(f"--- ITERATION: {0} ---")
        logger.warning(f"QD Score: {current_metrics['qd_score']:.3f}")
        logger.warning(f"Coverage: {current_metrics['coverage']:.3f}%")
        logger.warning(f"Max Fitness: {current_metrics['max_fitness']:.3f}")      
          
        # Main loop
        for i in tqdm(range(config.algorithm.map_elites.max_iterations)):
            start_time = time.time()
            key, subkey = jax.random.split(key)
            # Generate solutions
            genotypes, extra_info = map_elites.ask(repertoire, emitter_state, subkey)

            # Evaluate solutions: get fitness, descriptor and extra scores.
            # This is where custom evaluations on CPU or GPU can be added.
            key, subkey = jax.random.split(key)
            if config.algorithm.variation.genotype == "embedding":
                embeddings = genotypes.reshape(2 * batch_size, 77, 1024)
                fitnesses, descriptors, extra_scores = scoring_function(init_latents, embeddings, subkey)
            elif config.algorithm.variation.genotype == "latents":
                fitnesses, descriptors, extra_scores = scoring_function(genotypes, init_prompt_embeddings, subkey)

            # Update MAP-Elites
            repertoire, emitter_state, current_metrics = map_elites.tell(
                genotypes=genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors,
                repertoire=repertoire,
                emitter_state=emitter_state,
                extra_scores=extra_scores,
                extra_info=extra_info,
            )

            timelapse = time.time() - start_time

            # Metrics
            current_metrics["iteration"] = i+1
            current_metrics["time"] = timelapse
            current_metrics["evaluations"] = (i+2) * config.algorithm.map_elites.batch_size

            # Log
            wandb.log(current_metrics)
            csv_logger.log(current_metrics)
            logger.warning(f"--- ITERATION: {i + 1} ---")
            logger.warning(f"QD Score: {current_metrics['qd_score']:.3f}")
            logger.warning(f"Coverage: {current_metrics['coverage']:.3f}%")
            logger.warning(f"Max Fitness: {current_metrics['max_fitness']:.3f}")  
                      
            # Concatenate
            current_metrics = jax.tree.map(lambda x: jnp.array([x]), current_metrics)
            metrics = jax.tree.map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)


                   
        # Plot results
        fig, axes = plot_map_elites_results(env_steps=range(config.algorithm.map_elites.max_iterations),
            metrics=metrics,
            repertoire=repertoire,
            min_descriptor=config.algorithm.map_elites.min_descriptor,
            max_descriptor=config.algorithm.map_elites.max_descriptor
        )
        fig.savefig(_images_dir + f"mapelites-results.png")
        
        # Save final repertoire
        np.savez(_repertoire_dir + f"final_repertoire.npz",
            genotypes=repertoire.genotypes,
            descriptors=repertoire.descriptors,
            fitnesses=repertoire.fitnesses,
            centroids=centroids
        )
        
        # Save final images
        if config.algorithm.variation.genotype == "embedding":
            save_embedding = None
            save_latents = init_latent
            
        elif config.algorithm.variation.genotype == "latents":
            save_embedding = init_prompt_embedding
            save_latents = None
            
        save_final_images(repertoire=repertoire,
            num_inference_steps=config.generation.num_inference_steps,
            guidance_scale=config.generation.guidance_scale,
            scheduler=scheduler,
            unet=unet,
            vae=vae,
            n_sample=5,
            folder=_images_dir + f"final/",
            init_embedding=save_embedding,
            init_latent=save_latents,
        )
    

if __name__ == "__main__":
    main() 