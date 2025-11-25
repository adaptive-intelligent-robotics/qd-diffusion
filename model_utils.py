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

def get_sdxl_prompt_embeddings(
    prompt,
    tokenizer_1,
    tokenizer_2,
    text_encoder_1,
    text_encoder_2,
    device="cuda"
):
    """
    Manually encodes prompts for SDXL using the dual-encoder architecture.
    Returns:
        prompt_embeds: [batch, 77, 2048] (The 'encoder_hidden_states')
        pooled_prompt_embeds: [batch, 1280] (The 'text_embeds')
    """
    
    # 1. Tokenize the prompts
    # We assume prompts is a list of strings.
    # SDXL uses a standard sequence length of 77.
    
    # Tokenizer 1 (CLIP ViT-L)
    text_inputs_1 = tokenizer_1(
        prompt, 
        padding="max_length", 
        max_length=77, 
        truncation=True, 
        return_tensors="pt"
    )
    
    # Tokenizer 2 (OpenCLIP ViT-G)
    text_inputs_2 = tokenizer_2(
        prompt, 
        padding="max_length", 
        max_length=77, 
        truncation=True, 
        return_tensors="pt"
    )
    
    input_ids_1 = text_inputs_1.input_ids.to(device)
    input_ids_2 = text_inputs_2.input_ids.to(device)

    # 2. Get Embeddings from Encoders
    # We need 'output_hidden_states=True' because SDXL uses the PENULTIMATE layer, 
    # not the last one.
    
    with torch.no_grad():
        # Encoder 1 output
        output_1 = text_encoder_1(
            input_ids_1, 
            output_hidden_states=True
        )
        # Access the penultimate layer (second to last)
        hidden_states_1 = output_1.hidden_states[-2] 

        # Encoder 2 output
        output_2 = text_encoder_2(
            input_ids_2, 
            output_hidden_states=True
        )
        # Access the penultimate layer
        hidden_states_2 = output_2.hidden_states[-2]
        
        # For the Pooled Output (style/vibe), we explicitly need the pooled output 
        # from Encoder 2 only.
        pooled_prompt_embeds = output_2.text_embeds

    # 3. Concatenate Hidden States
    # Encoder 1 (768 dim) + Encoder 2 (1280 dim) = 2048 dim
    prompt_embeds = torch.cat([hidden_states_1, hidden_states_2], dim=-1)

    return prompt_embeds, pooled_prompt_embeds

def save_offspring_data(solutions, fitnesses, descriptors, images, itr, config, offspring_dir, images_dir, model_name):
    
    if config.save_offspring_data:
        torch.save(solutions["prompt_embedding"], os.path.join(offspring_dir, f"{itr}_prompt_embeddings.pt"))
        torch.save(solutions["latent"], os.path.join(offspring_dir, f"{itr}_latents.pt"))
        if model_name == "sdxl-turbo":
            torch.save(solutions["pooled_prompt_embedding"], os.path.join(offspring_dir, f"{itr}_pooled_prompt_embeddings.pt"))
        torch.save(fitnesses, os.path.join(offspring_dir, f"{itr}_fitnesses.pt"))
        torch.save(descriptors, os.path.join(offspring_dir, f"{itr}_descriptors.pt"))
        torch.save(images, os.path.join(offspring_dir, f"{itr}_images.pt"))
        
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
    images = archive.data()["solution"]["image"]
    num_sols = images.shape[0]
    
    generated_images_np = images.transpose(0, 2, 3, 1)
    generated_images_np = (generated_images_np + 1) / 2
    generated_images_np = np.clip(generated_images_np, 0, 1)
    
    # Determine grid size
    cols = math.ceil(np.sqrt(config.max_pop_size))
    rows = math.ceil(config.max_pop_size / cols)
    
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

    if isinstance(prompt_embedding, np.ndarray):
        prompt_embedding = torch.tensor(prompt_embedding, dtype=unet.dtype).cuda()
    elif isinstance(prompt_embedding, torch.Tensor):
        prompt_embedding = prompt_embedding.float().cuda()
    else:
        prompt_embedding = torch.tensor(np.array(prompt_embedding), dtype=unet.dtype).cuda()
    
    # Ensure prompt_embedding is a PyTorch tensor
    if isinstance(latents, np.ndarray):
        latents = torch.tensor(latents, dtype=unet.dtype).cuda()
    elif isinstance(latents, torch.Tensor):
        latents = latents.float().cuda()
    else:
        latents = torch.tensor(np.array(latents), dtype=unet.dtype).cuda()
            
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

def generate_images_from_latents_sdxl_turbo(
    latents,
    prompt_embedding,         
    pooled_prompt_embedding,  
    time_ids,                 
    num_inference_steps,
    scheduler,
    unet,
    vae,
    with_grad=False,
    device="cuda"
):
    dtype = unet.dtype  # Match U-Net precision (fp16 or fp32)

    # --- 1. Helper to standardize tensors ---
    def to_correct_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        # Ensure correct device and dtype matches the model
        return x.to(device=device, dtype=dtype)

    prompt_embedding = to_correct_tensor(prompt_embedding)
    pooled_prompt_embedding = to_correct_tensor(pooled_prompt_embedding)
    time_ids = to_correct_tensor(time_ids)
    latents = to_correct_tensor(latents)

    # --- 2. Setup Scheduler ---
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        
        # FIX 1: Use a temporary variable for the input, DO NOT overwrite 'latents'
        latent_model_input = scheduler.scale_model_input(latents, timestep=t)

        # Prepare SDXL added conditioning
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embedding,
            "time_ids": time_ids,
        }

        # --- 3. Predict Noise ---
        if with_grad:
            noise_pred = unet(
                latent_model_input, # Use the scaled input here
                t, 
                encoder_hidden_states=prompt_embedding,
                added_cond_kwargs=added_cond_kwargs
            ).sample
        else:
            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input, # Use the scaled input here
                    t, 
                    encoder_hidden_states=prompt_embedding,
                    added_cond_kwargs=added_cond_kwargs
                ).sample

        # FIX 2: Un-indent this so it runs regardless of with_grad
        # Use the ORIGINAL 'latents' variable here
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # --- 4. Decode ---
    scaling_factor = 0.13025 
    
    if with_grad:
        images = vae.decode(latents / scaling_factor).sample
    else:
        with torch.no_grad():
            images = vae.decode(latents / scaling_factor).sample

    return images