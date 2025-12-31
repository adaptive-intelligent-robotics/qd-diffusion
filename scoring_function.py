import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from descriptors import get_descriptor_fun
import descriptors
from model_utils import toggle_prompt_embedding_shape

def fitness_function(generated_images,
    prompt,
    clip_model,
    clip_tokenizer,
    with_grad=False
):
    # Normalize images to match CLIP's expected range
    # CLIP expects images in [0, 1] → then normalized to mean/std
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(generated_images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(generated_images.device)
    
    normalized_images = F.interpolate(generated_images, size=224, mode='bilinear')  # resize to 224x224
    normalized_images = (normalized_images - mean[None, :, None, None]) / std[None, :, None, None]  # normalize to mean/std
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

def qdhf_fitness_function(
    generated_images,
    prompt,
    clip_model,
    clip_tokenizer
):
    
    """Computes CLIP scores for a batch of images and a given text prompt."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(generated_images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(generated_images.device)
    normalized_images = F.interpolate(generated_images, size=224, mode='bilinear')  # resize to 224x224
    normalized_images = (normalized_images - mean[None, :, None, None]) / std[None, :, None, None]  # normalize to mean/std

    # img_tensor = torch.stack([CLIP_PREPROCESS(img) for img in imgs]).to(DEVICE)
    tokenized_text = clip_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(generated_images.device)
    
    with torch.no_grad():
        outputs = clip_model(
            pixel_values=normalized_images, 
            **tokenized_text
        )
        
        img_logits = outputs.logits_per_image
        
        clip_features = clip_model.get_image_features(pixel_values=normalized_images)
        clip_features = clip_features / clip_features.norm(p=2, dim=-1, keepdim=True)       

    img_logits = img_logits.detach().cpu().numpy().astype(np.float32)[:, 0]
    img_logits = 1 / img_logits * 100
    # Remap the objective from minimizing [0, 10] to maximizing [0, 100]
    qdhf_fitnesses = (10.0 - img_logits) * 10.0
    
    return qdhf_fitnesses, clip_features.cpu()

    
    
def scoring_function(
    solutions,
    generate_images_from_latents_fun,
    fitness_fun,
    descriptor_fun,
    model_name,
):
    
    if model_name == "sd-1-5":
        toggled_prompt_embeddings = toggle_prompt_embedding_shape(solutions["prompt_embedding"])
        generated_images = generate_images_from_latents_fun(solutions["latent"], toggled_prompt_embeddings)
        
    elif model_name == "sdxl-turbo":
        batch_size = solutions["latent"].shape[0]
        time_ids = torch.tensor([[1024, 1024, 0, 0, 512, 512]] * batch_size).to(solutions["latent"].device)
        generated_images = generate_images_from_latents_fun(
            solutions["latent"],
            solutions["prompt_embedding"],
            solutions["pooled_prompt_embedding"],
            time_ids=time_ids,
        )
    
    elif model_name == "flux":
        generated_images = generate_images_from_latents_fun(
            solutions["latent"],
            solutions["prompt_embedding"],
            solutions["pooled_prompt_embedding"],
        )

    generated_images_np = np.array(generated_images.cpu())

    # Find fitnesses  
    fitnesses = fitness_fun(generated_images=generated_images)
    
    # Find descriptors
    descriptors = np.array([descriptor_fun(image) for image in generated_images_np])           
        
    return fitnesses, descriptors, generated_images

    
def qdhf_scoring_function(
    solutions,
    generate_images_from_latents_fun,
    fitness_fun,
    qdhf_fitness_fun,
    descriptor_fun,
    model_name,
):
    
    if model_name == "sd-1-5":
        toggled_prompt_embeddings = toggle_prompt_embedding_shape(solutions["prompt_embedding"])
        generated_images = generate_images_from_latents_fun(solutions["latent"], toggled_prompt_embeddings)
        
    elif model_name == "sdxl-turbo":
        batch_size = solutions["latent"].shape[0]
        time_ids = torch.tensor([[1024, 1024, 0, 0, 512, 512]] * batch_size).to(solutions["latent"].device)
        generated_images = generate_images_from_latents_fun(
            solutions["latent"],
            solutions["prompt_embedding"],
            solutions["pooled_prompt_embedding"],
            time_ids=time_ids,
        )
    
    elif model_name == "flux":
    
        generated_images = generate_images_from_latents_fun(
            solutions["latent"],
            solutions["prompt_embedding"],
            solutions["pooled_prompt_embedding"],
        )

    generated_images_np = np.array(generated_images.cpu())

    # Find fitnesses  
    fitnesses = fitness_fun(generated_images=generated_images)
    
    # Find descriptors
    dreamsim_features = np.array([descriptor_fun(image) for image in generated_images_np])
    
    qdhf_fitnesses, clip_features = qdhf_fitness_fun(generated_images=generated_images)  
            
    return fitnesses, qdhf_fitnesses, dreamsim_features, clip_features, generated_images


def get_scoring_function(
    descriptor_name,
    prompt,
    clip_model,
    clip_tokenizer,
    generate_images_from_latents_fn,
    generator,
    model_name, 
    save_dir,
    config,
):
    
    descriptor_fn, descriptor_dim = get_descriptor_fun(
        descriptor_name=descriptor_name,
        generator=generator,
        save_dir=save_dir
    )
    
    fitness_fn = partial(fitness_function,
        prompt=prompt,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
    )
    
    if config.algorithm.name == "qdhf":
        
        qdhf_fitness_fn = partial(qdhf_fitness_function,
            prompt=prompt,
            clip_model=clip_model,
            clip_tokenizer=clip_tokenizer
        )
                                       
        if config.descriptor != "dreamsim":
            raise ValueError("QDHF algorithm requires 'dreamsim' descriptor.")
        
        scoring_fn = partial(qdhf_scoring_function,
            generate_images_from_latents_fun=generate_images_from_latents_fn,
            fitness_fun=fitness_fn,
            qdhf_fitness_fun=qdhf_fitness_fn,
            descriptor_fun=descriptor_fn,
            model_name=model_name,
        )
        
    else:
        scoring_fn = partial(scoring_function,
            generate_images_from_latents_fun=generate_images_from_latents_fn,
            fitness_fun=fitness_fn,
            descriptor_fun=descriptor_fn,
            model_name=model_name,
        )
    
    return scoring_fn, descriptor_dim