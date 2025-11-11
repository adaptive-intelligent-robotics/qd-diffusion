import numpy as np
import torch

import torch.nn.functional as F
from functools import partial
from descriptors import get_descriptor_fun

def fitness_function(generated_images,
    prompt,
    clip_model,
    clip_tokenizer,
    with_grad=False
):
    # Normalize images to match CLIP's expected range
    # CLIP expects images in [0, 1] → then normalized to mean/std
    normalized_images = F.interpolate(generated_images, size=224, mode='bilinear')  # resize to 224x224
    normalized_images = (normalized_images - 0.5) / 0.5  # normalize to [-1, 1] — assuming input is [0, 1]

    # Tokenize prompt (no gradients needed)
    text_inputs = clip_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(generated_images.device)

    # Forward through CLIP (no processor)
    if with_grad:
        image_features = clip_model.get_image_features(normalized_images)
        text_features = clip_model.get_text_features(**text_inputs)
        
    else:
        with torch.no_grad():
            image_features = clip_model.get_image_features(normalized_images)
            text_features = clip_model.get_text_features(**text_inputs)

    # Cosine similarity
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    # logit_scale = clip_model.logit_scale.exp()
    fitnesses_torch = (image_features @ text_features.T).squeeze() #* logit_scale
    
    if with_grad:
        # If gradients are needed, return the tensor directly
        return fitnesses_torch
    
    else:        
        return fitnesses_torch.detach().cpu().numpy()  # Convert to NumPy
    
def scoring_function(
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
    
    descriptor_fn, descriptor_dim = get_descriptor_fun(
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
    
    scoring_fn = partial(scoring_function,
        generate_images_from_latents_fun=generate_images_from_latents_fn,
        fitness_fun=fitness_fn,
        descriptor_fun=descriptor_fn,
    )
    
    return scoring_fn, descriptor_dim