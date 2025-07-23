import torch
from PIL import Image
import jax.numpy as jnp
import numpy as np
import os
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm

def expand_prompt_embedding(prompt_embedding, batch_size):
    """
    Takes a [2, 77, 1024] prompt embedding (uncond, cond) and returns
    [2*batch_size, 77, 1024] with batch_size uncond followed by batch_size cond.
    """
    uncond = prompt_embedding[0].unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, 77, 1024]
    cond = prompt_embedding[1].unsqueeze(0).repeat(batch_size, 1, 1)    # [batch_size, 77, 1024]
    expanded = torch.cat([uncond, cond], dim=0)  # [2 * batch_size, 77, 1024]
    return expanded

# Function to encode text prompt into embeddings
def get_text_embeddings(prompt,
        batch_size,
        tokenizer,
        text_encoder,
        with_grad=False
    ):
    text_input = tokenizer([str(prompt)] * batch_size, padding="max_length", max_length=77, return_tensors="pt")

    if with_grad:
        # Enable gradients for text encoding
        text_embeddings = text_encoder(text_input.input_ids.cuda())[0]
    else:
        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.cuda())[0]
    
    # Generate unconditional embeddings for classifier-free guidance
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    
    if with_grad:
        uncond_embeddings = text_encoder(uncond_input.input_ids.cuda())[0]
    else:
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.cuda())[0]
    
    # Concatenate conditional & unconditional embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings]) 
    return text_embeddings.detach()


def generate_images_from_latents(
    latents,
    prompt_embedding,
    num_inference_steps,
    guidance_scale,
    scheduler,
    unet,
    vae,
    with_grad=False,
):

    # Convert JAX latents to PyTorch tensor
    if isinstance(prompt_embedding, np.ndarray):
        prompt_embedding = torch.tensor(prompt_embedding, dtype=torch.float32).cuda()
    elif isinstance(prompt_embedding, torch.Tensor):
        prompt_embedding = prompt_embedding.float().cuda()
    else:
        prompt_embedding = torch.tensor(np.array(prompt_embedding), dtype=torch.float32).cuda()
    
    # Ensure prompt_embedding is a PyTorch tensor
    if isinstance(latents, np.ndarray):
        latents = torch.tensor(latents, dtype=torch.float32).cuda()
    elif isinstance(latents, torch.Tensor):
        latents = latents.float().cuda()
    else:
        latents = torch.tensor(np.array(latents), dtype=torch.float32).cuda()
            
    # Get text embeddings
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma  # Scale initial noise

    for t in scheduler.timesteps:
        # Expand latents for classifier-free guidance (double batch size)
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # Predict noise
        if with_grad:
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embedding).sample
        else:
            # Disable gradients for inference
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embedding).sample

        # Apply classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Denoising step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents into images
    if with_grad:
        images = vae.decode(latents / 0.18215).sample
    else:
        # Disable gradients for decoding    
        with torch.no_grad():
            images = vae.decode(latents / 0.18215).sample  # Scale down before decoding
   
    return images




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
        fitnesses_np = fitnesses_torch.detach().cpu().numpy()  # Convert to NumPy
        fitnesses = jnp.array(fitnesses_np, dtype=jnp.float32)  # Convert to JAX dtype
        
        return fitnesses


def save_images(images, iteration="", filenames=None, folder="results"):
    """Saves a batch of images into a results folder."""
    
    # Ensure the directory exists
    os.makedirs(folder, exist_ok=True)
    
    if filenames is None:
        filenames = [f"iteration_{iteration}_sample_{i+1}.png" for i in range(len(images))]
      
    for i, image_array in enumerate(images):
        # Convert to NumPy if needed
        image_array = np.array(image_array.cpu())

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
        num_inference_steps,
        guidance_scale,
        scheduler,
        unet,
        vae,
        n_sample=5,
        folder="results",
        init_embedding=None,
        init_latent=None,
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
        
    num_genotypes = len(combined_genotypes)

    if init_embedding is not None:
        
        prompt_embeddings = init_embedding.repeat(num_genotypes, 1, 1)
        prompt_embeddings = prompt_embeddings.reshape(num_genotypes*2, 77, 1024)

        final_images = generate_images_from_latents(
            combined_genotypes,
            prompt_embeddings,
            num_inference_steps,
            guidance_scale,
            scheduler,
            unet,
            vae,
        )
    
    elif init_latent is not None:
        
        latents = init_latent.repeat(num_genotypes, axis=0)  # Repeat for batch size
        combined_genotypes = jnp.array(combined_genotypes).reshape(num_genotypes*2, 77, 1024)
        
        final_images = generate_images_from_latents(
            latents,
            combined_genotypes,
            num_inference_steps,
            guidance_scale,
            scheduler,
            unet,
            vae,
        )        
    
    # Use the existing save_images function to save the images
    save_images(final_images, filenames=combined_filenames, folder=folder)



# Plot valid thumbnails in descriptor space for a repertoire
def plot_thumbnails(
    repertoire,
    num_inference_steps,
    guidance_scale,
    scheduler,
    unet,
    vae,
    batch_size,
    save_path,
    init_embedding=None,
    init_latent=None,
):
    """
    Filters valid genotypes and plots their decoded images in descriptor space.

    Parameters:
    - repertoire: Dictionary containing 'genotypes', 'descriptors', and 'fitnesses'.
    - init_embedding: Optional, used for image generation when given.
    - init_latent: Optional, used for image generation when given.
    - All other params: Required for image generation and plotting.
    """

    valid_genotypes = []
    valid_descriptors = []
    valid_fitnesses = []

    for fitness, descriptor, genotype in zip(repertoire.fitnesses, repertoire.descriptors, repertoire.genotypes):
        if fitness == -jnp.inf:
            continue
        valid_genotypes.append(genotype)
        valid_descriptors.append(descriptor)
        valid_fitnesses.append(fitness)

    if len(valid_genotypes) == 0:
        print("No valid genotypes to plot.")
        return

    valid_descriptors = jnp.array(valid_descriptors)

    if init_embedding is not None:
        
        prompt_embeddings = init_embedding.repeat(len(valid_genotypes), 1, 1)
        prompt_embeddings = prompt_embeddings.reshape(len(valid_genotypes)*2, 77, 1024)

        all_images = generate_images_from_latents(
            valid_genotypes,
            prompt_embeddings,
            num_inference_steps,
            guidance_scale,
            scheduler,
            unet,
            vae,
        )
    
    elif init_latent is not None:
        
        latents = init_latent.repeat(len(valid_genotypes), axis=0)  # Repeat for batch size
        valid_genotypes = jnp.array(valid_genotypes).reshape(len(valid_genotypes)*2, 77, 1024)
        
        all_images = generate_images_from_latents(
            latents,
            valid_genotypes,
            num_inference_steps,
            guidance_scale,
            scheduler,
            unet,
            vae,
        )   

    else:
        raise ValueError("Either init_embedding or init_latent must be provided.")

    assert len(valid_descriptors) == len(all_images), "Mismatch between descriptors and images"

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(valid_descriptors[:, 0], valid_descriptors[:, 1], s=5, c='lightgray', alpha=0.5)

    for (x, y), img in zip(valid_descriptors, all_images):
        try:
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img)
            elif isinstance(img, np.ndarray):
                img = (img * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img)

            thumbnail = img.resize((256, 256))
            imagebox = OffsetImage(thumbnail, zoom=0.15)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.1)
            ax.add_artist(ab)
        except Exception as e:
            print(f"Failed to plot image at ({x}, {y}): {e}")

    ax.set_title("Repertoire Descriptor Space with Image Thumbnails")
    ax.set_xlabel("Descriptor 1")
    ax.set_ylabel("Descriptor 2")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path+"thumbnail_scatter.png", dpi=300)
    plt.close()
    print(f"Saved plot to {save_path}")
