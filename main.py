import os
from pathlib import Path
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

from APDDv2.demo import load_model, get_score
import APDDv2.models.clip as clip
from diffusers import (
    DiffusionPipeline, 
    StableDiffusionPipeline,
    UNet2DConditionModel,
    PNDMScheduler,
    AutoencoderKL,
    LMSDiscreteScheduler
)
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer
from feature_functions import (
    brightness,
    average_hue,
    get_artclip_score_model,
)

from model_utils import (get_text_embeddings, 
    generate_images_from_latents,
    generate_image_from_genotype,
    fitness_function,
    save_images,
    save_final_images
)
from PIL import Image

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from qdax.baselines.cmaes import CMAES
from qdax.core.emitters.mutation_operators import isoline_variation
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
    vae = AutoencoderKL.from_pretrained(config.model.name, subfolder="vae", torch_dtype=torch.float16).cuda()
    tokenizer = CLIPTokenizer.from_pretrained(config.model.name, subfolder="tokenizer", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(config.model.name, subfolder="text_encoder", torch_dtype=torch.float16).cuda()
    unet = UNet2DConditionModel.from_pretrained(config.model.name, subfolder="unet", torch_dtype=torch.float16).cuda()
    scheduler = LMSDiscreteScheduler.from_pretrained(config.model.name, subfolder="scheduler", torch_dtype=torch.float16)
    
    # Initialize random key
    key = jax.random.PRNGKey(config.algorithm.seed)
    
    # Initialise scoring function
    prompt_embeddings = get_text_embeddings(
        prompt=config.generation.prompt,
        batch_size=config.algorithm.batch_size,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
    )
    
    generate_images_from_latents_fn = partial(generate_images_from_latents,
        prompt=config.generation.prompt,
        num_inference_steps=config.generation.num_inference_steps,
        guidance_scale=config.generation.guidance_scale,
        scheduler=scheduler,
        unet=unet,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder
    )
    
    generate_image_from_genotype_fn = partial(generate_image_from_genotype,
        prompt=config.generation.prompt,
        num_inference_steps=config.generation.num_inference_steps,
        guidance_scale=config.generation.guidance_scale,
        scheduler=scheduler,
        unet=unet,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder
    )
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    descriptor_functions = []
    
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
        

    def scoring_fn(genotypes,
        key,
        prompt,
        clip_model,
        clip_processor,
        generate_images_from_latents_fun,
        descriptor_functions: List[Callable]
    ):
        
        # Convert jax genotypes to pytorch
        latents = torch.tensor(np.array(genotypes), dtype=torch.float16).cuda()    
    
        # Generate images
        generated_images = generate_images_from_latents_fun(latents)
        generated_images_np = np.array([np.array(img) for img in generated_images])  
        generated_images_jnp = jnp.array(generated_images_np, dtype=jnp.float16)  # Ensure JAX dtype
        
        fitnesses = fitness_function(generated_images=generated_images,
            prompt=prompt,
            clip_model=clip_model,
            clip_processor=clip_processor,
        )
                
        descriptors = []
        
        for d_fun in descriptor_functions:
            descriptor = jnp.array([d_fun(image) for image in generated_images_np])           
            descriptors.append(descriptor)
            
        descriptors = jnp.stack(descriptors, axis=-1)  # Shape: (, num_descriptors)
        
        return fitnesses, descriptors, {}

    scoring_function = partial(scoring_fn,
        prompt=config.generation.prompt,
        clip_model=clip_model,
        clip_processor=clip_processor,
        generate_images_from_latents_fun=generate_images_from_latents_fn,
        descriptor_functions=descriptor_functions,
    )

    # Compute the MAP-Elites centroids
    key, subkey = jax.random.split(key)
    logger.warning("--- Loading Centroids ---")

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
        ):
            image_genotypes = jnp.reshape(genotypes, (batch_size, 4, 64, 64))
        
            fitnesses, descriptors, _ = scoring_function(image_genotypes, key)
                                    
            return fitnesses.squeeze(), descriptors.squeeze()
        
        partial_fitness_fn = partial(cma_es_scoring_fn,
            batch_size=config.algorithm.cma_es.population_size,
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
            mean_init=jax.random.normal(subkey, shape=(4*64*64,), dtype=jnp.float16),
            init_sigma=config.algorithm.cma_es.sigma,
            delay_eigen_decomposition=config.algorithm.cma_es.delay_eigen_decomposition,
        )
        
        state = cmaes.init()
        
        # Save initial mean image
        init_samples_images = jnp.reshape(state.mean, (1, 4, 64, 64))
        init_images = generate_image_from_genotype_fn(init_samples_images)
        save_images(init_images, iteration="init", folder=_images_dir + f"init/")
        
        metrics = dict.fromkeys(["iteration", "qd_score", "coverage", "max_fitness", "time"], jnp.array([]))
        csv_logger = CSVLogger(
            filename=_metrics_dir + f"cmaes-logs.csv",
            header=list(metrics.keys())
        )

        # Run optimization
        for i in tqdm(range(config.algorithm.cma_es.max_iterations)):
            start_time = time.time()

            # Sample, sample_size = population_size
            samples, key = cmaes.sample(state, key)

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
            current_metrics["iteration"] = i
            current_metrics["time"] = timelapse
            
            # Log
            wandb.log(current_metrics)
            csv_logger.log(current_metrics)
            logger.warning(f"--- ITERATION: {i} ---")
            logger.warning(f"QD Score: {current_metrics['qd_score']:.2f}")
            logger.warning(f"Coverage: {current_metrics['coverage']:.2f}%")
            logger.warning("Max Fitness:" +  str(current_metrics['max_fitness']))
            
            # Concatenate
            current_metrics = jax.tree.map(lambda x: jnp.array([x]), current_metrics)
            metrics = jax.tree.map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

            # check stop condition
            stop_condition = cmaes.stop_condition(state)

            if stop_condition:
                break


        logger.warning("Num iterations before stop condition: ", i)
               
        # Plot results
        fig, axes = plot_map_elites_results(env_steps=range(config.algorithm.map_elites.max_iterations),
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
        save_final_images(passive_repertoire,
            config.generation.prompt,
            config.generation.num_inference_steps,
            config.generation.guidance_scale,
            scheduler,
            unet,
            vae,
            tokenizer,
            text_encoder,
            n_sample=5,
            folder=_images_dir + f"final/"
        )
                
   
    elif config.algorithm.name == "map_elites":

        logger.warning("--- Algorithm initialisation ---")
    
        key, subkey = jax.random.split(key)  # JAX PRNG handling
        init_genotypes = jax.random.normal(subkey, shape=(config.algorithm.batch_size, 4, 64, 64), dtype=jnp.float16)
        init_images = generate_image_from_genotype_fn(init_genotypes)
        save_images(init_images, iteration="init", folder=_images_dir + f"init/")
        
        # Define a metrics function
        variation_fn = partial(
            isoline_variation, iso_sigma=config.algorithm.iso_sigma, line_sigma=config.algorithm.line_sigma
        )

        mixing_emitter = MixingEmitter(
            mutation_fn=None,
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=config.algorithm.batch_size
        )
       # Instantiate MAP-Elites
        map_elites = MAPElites(
            scoring_function=None,
            emitter=mixing_emitter,
            metrics_function=map_elites_metrics_fn,
        )
        
        # Evaluate the initial population
        key, subkey = jax.random.split(key)
        fitnesses, descriptors, extra_scores = scoring_function(init_genotypes, subkey)
        
        
        # Compute initial repertoire and emitter state
        repertoire, emitter_state, metrics = map_elites.init_ask_tell(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            key=key,
            extra_scores=extra_scores,
        )
        
        
        metrics = dict.fromkeys(["iteration", "qd_score", "coverage", "max_fitness", "time"], jnp.array([]))
        csv_logger = CSVLogger(
            filename=_metrics_dir + f"mapelites-logs.csv",
            header=list(metrics.keys())
        )

        # Main loop
        for i in tqdm(range(config.algorithm.map_elites.max_iterations)):
            start_time = time.time()
            key, subkey = jax.random.split(key)
            # Generate solutions
            genotypes, extra_info = map_elites.ask(repertoire, emitter_state, subkey)

            # Evaluate solutions: get fitness, descriptor and extra scores.
            # This is where custom evaluations on CPU or GPU can be added.
            key, subkey = jax.random.split(key)
            fitnesses, descriptors, extra_scores = scoring_function(genotypes, subkey)

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
            current_metrics["iteration"] = i
            current_metrics["time"] = timelapse

            # Log
            wandb.log(current_metrics)
            csv_logger.log(current_metrics)
            logger.warning(f"--- ITERATION: {i} ---")
            logger.warning(f"QD Score: {current_metrics['qd_score']:.2f}")
            logger.warning(f"Coverage: {current_metrics['coverage']:.2f}%")
            logger.warning("Max Fitness:" +  str(current_metrics['max_fitness']))
            
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
        save_final_images(repertoire,
            config.generation.prompt,
            config.generation.num_inference_steps,
            config.generation.guidance_scale,
            scheduler,
            unet,
            vae,
            tokenizer,
            text_encoder,
            n_sample=5,
            folder=_images_dir + f"final/"
        )
        
    

if __name__ == "__main__":
    main() 