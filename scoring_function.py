import jax.numpy as jnp
import numpy as np

from typing import Callable, List
from functools import partial
from model_utils import generate_images_from_latents, fitness_function
from descriptors import get_descriptor_fun


def scoring_fn(
    latents,
    prompt_embeddings,
    generate_images_from_latents_fun,
    fitness_fun,
    descriptor_fun,
):
    # Generate images
    generated_images = generate_images_from_latents_fun(latents, prompt_embeddings)
    generated_images_np = np.array(generated_images.cpu())

    # Find fitnesses  
    fitnesses = fitness_fun(generated_images=generated_images)
    
    # Find descriptors
    descriptors = np.array([descriptor_fun(image) for image in generated_images_np])           
        
    return fitnesses, descriptors, generated_images


def get_scoring_function(
    descriptor_name,
    project_desc,
    project_dim,
    prompt,
    clip_model,
    clip_tokenizer,
    generate_images_from_latents_fn,
    generator,
    save_dir,
):
    
    descriptor_fn = get_descriptor_fun(
        descriptor_name=descriptor_name,
        project_desc=project_desc,
        projection_dim=project_dim,
        generator=generator,
        save_dir=save_dir
    )
    
    fitness_fn = partial(fitness_function,
        prompt=prompt,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
    )
    
    scoring_function = partial(scoring_fn,
        generate_images_from_latents_fun=generate_images_from_latents_fn,
        fitness_fun=fitness_fn,
        descriptor_fun=descriptor_fn,
    )
    
    return scoring_function