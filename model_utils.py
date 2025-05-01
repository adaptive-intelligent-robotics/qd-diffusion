from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer
import torch
from PIL import Image
import jax.numpy as jnp
import numpy as np
import os
import random


# Function to encode text prompt into embeddings
def get_text_embeddings(prompt,
        batch_size,
        tokenizer,
        text_encoder,
    ):
    text_input = tokenizer([str(prompt)] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.cuda())[0]
    
    # Generate unconditional embeddings for classifier-free guidance
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.cuda())[0]
    
    # Concatenate conditional & unconditional embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings]) 
    return text_embeddings


def generate_images_from_latents(latents,
        prompt,
        num_inference_steps,
        guidance_scale,
        scheduler,
        unet,
        vae,
        tokenizer,
        text_encoder
    ):

    batch_size = latents.shape[0]  # Number of images in batch
    text_embeddings = get_text_embeddings(prompt, batch_size, tokenizer, text_encoder)

    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma  # Scale initial noise

    for t in scheduler.timesteps:
        # Expand latents for classifier-free guidance (double batch size)
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # Predict noise
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Apply classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Denoising step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents into images
    with torch.no_grad():
        images = vae.decode(latents / 0.18215).sample  # Scale down before decoding

    # Convert tensor to PIL images
    images = (images / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
    images = (images * 255).cpu().numpy().astype(np.uint8)  # Convert to uint8
    pil_images = [Image.fromarray(img.transpose(1, 2, 0)) for img in images]  # CHW → HWC

    return pil_images



def fitness_function(generated_images,
        prompt,
        clip_model,
        clip_processor,
):
    # Generate CLIP Scores
    inputs = clip_processor(text=prompt, 
                    images=generated_images, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    
    # Convert fitnesses to jax arrays
    fitnesses_torch = outputs.logits_per_image  # PyTorch tensor
    fitnesses_np = fitnesses_torch.detach().cpu().numpy()  # Convert to NumPy
    fitnesses = jnp.array(fitnesses_np, dtype=jnp.float16)  # Convert to JAX dtype
    
    return fitnesses


def generate_image_from_genotype(image_array,
        prompt,
        num_inference_steps,
        guidance_scale,
        scheduler,
        unet,
        vae,
        tokenizer,
        text_encoder
    ):
    """Displays an image from a NumPy array."""
    # Convert jax genotypes to pytorch
    latents = torch.tensor(np.array(image_array), dtype=torch.float16).cuda()    
    
    # Generate images
    generated_images = generate_images_from_latents(latents, prompt, num_inference_steps, guidance_scale, scheduler, unet, vae, tokenizer, text_encoder)

    return generated_images


def save_images(images, iteration="", filenames=None, folder="results"):
    """Saves a batch of images into a results folder."""
    
    # Ensure the directory exists
    os.makedirs(folder, exist_ok=True)
    
    if filenames is None:
        filenames = [f"iteration_{iteration}_sample_{i+1}.png" for i in range(len(images))]
      
    for i, image_array in enumerate(images):
        # Convert to NumPy if needed
        image_array = np.array(image_array)

        # Normalize to [0,255] if needed
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
        image_array = (image_array * 255).astype(np.uint8)

        # Handle shape issues
        if image_array.shape == (512, 512, 3):  # Already in correct format
            image = Image.fromarray(image_array, mode="RGB")
        elif image_array.shape[0] == 3:  # If (3, H, W) -> Convert to (H, W, 3)
            image_array = np.transpose(image_array, (1, 2, 0))
            image = Image.fromarray(image_array, mode="RGB")
        elif image_array.shape[0] == 1:  # If (1, H, W) -> Convert to (H, W)
            image_array = np.squeeze(image_array, axis=0)
            image = Image.fromarray(image_array, mode="L")
        elif image_array.ndim == 2:  # If already (H, W), use grayscale
            image = Image.fromarray(image_array, mode="L")
        else:
            raise ValueError(f"Unexpected shape {image_array.shape} for image {i}")

        # Save image
        image.save(os.path.join(folder, filenames[i]))
        


def save_final_images(repertoire,
        prompt,
        num_inference_steps,
        guidance_scale,
        scheduler,
        unet,
        vae,
        tokenizer,
        text_encoder,
        n_sample=5,
        folder="results"
    ):
    """
    Saves non-empty solutions from the repertoire using the provided save_images function.
    Samples the highest fitness genotype and n_sample other genotypes.
    
    Parameters:
    - repertoire: Dictionary containing 'genotypes', 'descriptors', and 'fitnesses'.
    - save_images_fn: Function to save images (expects (images, filenames, folder) as input).
    - prompt: The prompt used for generating the images from genotypes.
    - n_sample: The number of additional genotypes to sample.
    - folder: Directory to save the final images.
    """
    os.makedirs(folder, exist_ok=True)  # Ensure the directory exists

    valid_genotypes = []
    fitnesses = []
    descriptors = []
    filenames = []

    # Collect valid genotypes, descriptors, fitnesses, and filenames
    for i, (fitness, descriptor, genotype) in enumerate(zip(repertoire.fitnesses, repertoire.descriptors, repertoire.genotypes)):
        if fitness == -jnp.inf:  # Skip empty solutions
            continue
        valid_genotypes.append(genotype)
        fitnesses.append(fitness[0])
        descriptors.append(descriptor)
        filename = f"final_image_{fitness[0]}_{descriptor[0]:.4f}_{descriptor[1]:.4f}.png"
        filenames.append(filename)

    # Check if we have enough valid genotypes
    if len(valid_genotypes) <= n_sample + 1:  # +1 for the best genotype
        # If not enough, use all valid genotypes
        combined_genotypes = valid_genotypes
        combined_filenames = filenames
    else:
        # Step 1: Find the genotype with the highest fitness
        best_fitness_idx = np.argmax(fitnesses)
        best_genotype = valid_genotypes[best_fitness_idx]
        best_filename = filenames[best_fitness_idx]

        # Step 2: Sample `n_sample` other valid genotypes (and filenames)
        # Remove the best genotype from the valid genotypes list to avoid selecting it again
        remaining_genotypes = valid_genotypes[:best_fitness_idx] + valid_genotypes[best_fitness_idx+1:]
        remaining_filenames = filenames[:best_fitness_idx] + filenames[best_fitness_idx+1:]
        
        # Sample `n_sample` from the remaining genotypes and filenames
        sampled_genotypes_indices = random.sample(range(len(remaining_genotypes)), n_sample)
        sampled_genotypes = [remaining_genotypes[i] for i in sampled_genotypes_indices]
        sampled_filenames = [remaining_filenames[i] for i in sampled_genotypes_indices]

        # Combine the best genotype with the sampled ones
        combined_genotypes = [best_genotype] + sampled_genotypes
        combined_filenames = [best_filename] + sampled_filenames
    
    # Step 3: Generate images from combined genotypes
    final_images = generate_image_from_genotype(
        combined_genotypes,
        prompt,
        num_inference_steps,
        guidance_scale,
        scheduler,
        unet,
        vae,
        tokenizer,
        text_encoder
    )
    
    # Use the existing save_images function to save the images
    save_images(final_images, filenames=combined_filenames, folder=folder)
