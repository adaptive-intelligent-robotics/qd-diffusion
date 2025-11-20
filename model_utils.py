import torch
from PIL import Image
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from image_utils import save_image

def toggle_prompt_embedding_shape(embedding):
    """
    Toggle shape between:
    - [2*B, 77, 1024] -> [B, 2, 77, 1024]
    - [B, 2, 77, 1024] -> [2*B, 77, 1024]

    Works with both PyTorch tensors and NumPy arrays.
    Assumes uncond + cond are stacked vertically in the 2*B form.
    """
    # Determine backend
    if isinstance(embedding, torch.Tensor):
        backend = "torch"
    elif isinstance(embedding, np.ndarray):
        backend = "numpy"
    else:
        raise TypeError(f"Unsupported type: {type(embedding)}. Must be torch.Tensor or np.ndarray.")

    # Get number of dimensions
    ndim = embedding.ndim

    # Toggle shapes
    if ndim == 3:
        # Shape is [2*B, 77, 1024], convert to [B, 2, 77, 1024]
        B = embedding.shape[0] // 2
        uncond = embedding[:B]
        cond = embedding[B:]
        
        if backend == "torch":
            return torch.stack([uncond, cond], dim=1)
        else: # backend == "numpy"
            return np.stack([uncond, cond], axis=1)

    elif ndim == 4:
        # Shape is [B, 2, 77, 1024], convert to [2*B, 77, 1024]
        uncond = embedding[:, 0]
        cond = embedding[:, 1]
        
        if backend == "torch":
            return torch.cat([uncond, cond], dim=0)
        else: # backend == "numpy"
            return np.concatenate([uncond, cond], axis=0)

    else:
        raise ValueError(f"Expected input with 3 or 4 dimensions, got {ndim}")
    
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

def save_offspring_data(solutions, fitnesses, descriptors, images, itr, config, offspring_dir, images_dir, model_name):
    
    if config.save_offspring_data:
        torch.save(solutions["prompt_embedding"], os.path.join(offspring_dir, f"{itr}_prompt_embeddings.pt"))
        torch.save(solutions["latent"], os.path.join(offspring_dir, f"{itr}_latents.pt"))
        if model_name == "sdxl-turbo":
            torch.save(solutions["pooled_prompt_embedding"], os.path.join(offspring_dir, f"{itr}_pooled_prompt_embeddings.pt"))
        torch.save(fitnesses, os.path.join(offspring_dir, f"{itr}_fitnesses.pt"))
        torch.save(descriptors, os.path.join(offspring_dir, f"{itr}_descriptors.pt"))
        
    if config.save_images:
        itr_image_dir = os.path.join(images_dir, f"offspring_{itr}/")
        os.makedirs(itr_image_dir, exist_ok=True)
        for i, img in enumerate(images):
            save_image(img, os.path.join(itr_image_dir, f"offspring_image_{i}.png"))
    
    return


def save_repertoire(repertoire,
                    itr,
                    save_dir,
                    config
):  
    prompt_embeddings = repertoire.data()["solution"]["prompt_embedding"]
    latents = repertoire.data()["solution"]["latent"]
    torch.save(prompt_embeddings, os.path.join(save_dir, f"{itr}_prompt_embeddings.pt"))
    torch.save(latents, os.path.join(save_dir, f"{itr}_latents.pt"))
    
    if config.model.name == "sdxl-turbo":
        pooled_prompt_embeddings = repertoire.data()["solution"]["pooled_prompt_embedding"]
        torch.save(pooled_prompt_embeddings, os.path.join(save_dir, f"{itr}_pooled_prompt_embeddings.pt"))
        
    torch.save(repertoire.data()["objective"], os.path.join(save_dir, f"{itr}_fitnesses.pt"))
    torch.save(repertoire.data()["measures"], os.path.join(save_dir, f"{itr}_descriptors.pt"))

    return


def plot_repertoire_images(
    archive,
    itr,
    config,
    generate_images_from_latents_fun,
    save_dir,
):
    latents = archive.data()["solution"]["latent"]
    num_sols = latents.shape[0]
    prompt_embeddings = archive.data()["solution"]["prompt_embedding"]
    toggled_prompt_embeddings = toggle_prompt_embedding_shape(prompt_embeddings)
    
    if config.model.name == "sd-1-5":
        generated_images = generate_images_from_latents_fun(latents, toggled_prompt_embeddings)
        
    elif config.model.name == "sdxl-turbo":
        pooled_prompt_embeddings = archive.data()["solution"]["pooled_prompt_embedding"]
        generated_images = generate_images_from_latents_fun(
            latents,
            toggled_prompt_embeddings,
            pooled_prompt_embeddings,
        )

    generated_images_np = np.array(generated_images.cpu())
    generated_images_np = generated_images_np.transpose(0, 2, 3, 1)
    generated_images_np = (generated_images_np + 1) / 2
    generated_images_np = np.clip(generated_images_np, 0, 1)
    
    # Determine grid size
    cols = math.ceil(np.sqrt(config.algorithm.max_pop_size))
    rows = math.ceil(config.algorithm.max_pop_size / cols)
    
    # Create the figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))  # adjust figsize for visibility
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    
    # Plot each image
    for i, ax in enumerate(axes):
        if i < num_sols:
            ax.imshow(generated_images_np[i])
            ax.axis("off")
        else:
            ax.axis("off")  # hide any unused subplots
    
    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(save_dir, f"repertoire_{itr}"))

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
        

