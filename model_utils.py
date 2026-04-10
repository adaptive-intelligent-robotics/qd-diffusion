import gc
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from diffusers import (
    AutoPipelineForText2Image,
    BitsAndBytesConfig,
    DDIMScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
    StableDiffusionPipeline
)
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from functools import partial
from image_utils import save_image
from optimum.quanto import freeze, qfloat8, quantize

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
    images = repertoire.data()["solution"]["image"] 
    torch.save(prompt_embeddings, os.path.join(save_dir, f"{itr}_prompt_embeddings.pt"))
    torch.save(latents, os.path.join(save_dir, f"{itr}_latents.pt"))
    torch.save(images,  os.path.join(save_dir, f"{itr}_images.pt"))
    
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
    # Normalize images based on model type
    if config.model.name == "flux":
        # Flux images are already in [0, 1] range after postprocessing
        generated_images_np = np.clip(generated_images_np, 0, 1)
    else:
        # SD models output in [-1, 1] range, need to normalize to [0, 1]
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
    decode_chunk_size=4,
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
            # Decode in chunks to avoid large transient VAE allocations.
            latents_scaled = latents / scaling_factor
            if latents_scaled.shape[0] <= decode_chunk_size:
                images = vae.decode(latents_scaled).sample
            else:
                decoded_chunks = []
                for i in range(0, latents_scaled.shape[0], decode_chunk_size):
                    decoded_chunks.append(
                        vae.decode(latents_scaled[i : i + decode_chunk_size]).sample
                    )
                images = torch.cat(decoded_chunks, dim=0)

    return images


def get_flux_mu(pipe, latents):
    """
    Calculates the resolution-dependent density shift (mu).
    Required by Flux to ensure timesteps match the image sequence length.
    """
    return calculate_shift(
        latents.shape[1],
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    
def generate_images_from_latents_flux(
    latents,
    prompt_embedding,         
    pooled_prompt_embedding,  
    pipe,
    text_ids,
    latent_img_ids,
    num_inference_steps,
    guidance_scale,
    height,
    width,
    micro_batch_size=None,
    device="cuda"
):
    """
    Complete Flux generation function.
    Handles NumPy/Tensor inputs, VRAM offloading, and batch dimension alignment.
    """    
    dtype = pipe.transformer.dtype
    
    # --- 1. Robust Type Handling Helper ---
    def to_correct_tensor(x, is_pooled=False):
        # Converts NumPy to Tensor or passes existing Tensors through
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        
        x = x.to(device=device, dtype=dtype)
        
        # Remove redundant dimensions: [Batch, 1, Seq, Dim] -> [Batch, Seq, Dim]
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
            
        # Ensure pooled embedding is 2D [Batch, Dim]
        if is_pooled and x.ndim > 2:
            x = x.view(x.shape[0], -1)
            
        return x

    # Initial full-batch conversion (tensors stay on CPU until sliced if needed)
    latents_all = to_correct_tensor(latents)
    prompt_all = to_correct_tensor(prompt_embedding)
    pooled_all = to_correct_tensor(pooled_prompt_embedding, is_pooled=True)
    
    total_samples = latents_all.shape[0]
    if micro_batch_size is None:
        micro_batch_size = min(total_samples, 4)
    all_images = []

    # --- 2. Micro-Batch Loop ---
    for i in range(0, total_samples, micro_batch_size):
        # Slice the batch
        m_latents = latents_all[i : i + micro_batch_size]
        m_prompt = prompt_all[i : i + micro_batch_size]
        m_pooled = pooled_all[i : i + micro_batch_size]
        curr_batch_size = m_latents.shape[0]

        # Setup Scheduler for this specific resolution/batch
        mu = get_flux_mu(pipe, m_latents)
        pipe.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
        timesteps = pipe.scheduler.timesteps
        
        # Setup Guidance
        if pipe.transformer.config.guidance_embeds:
            guidance = torch.full([curr_batch_size], guidance_scale, device=device, dtype=torch.float32)
        else:
            guidance = None

        # --- 3. Denoising Loop ---
        for t in timesteps:
            timestep_input = t.expand(curr_batch_size).to(m_latents.dtype)

            with torch.no_grad(), pipe.transformer.cache_context("cond"):
                noise_pred = pipe.transformer(
                    hidden_states=m_latents,
                    timestep=timestep_input / 1000,
                    guidance=guidance,
                    pooled_projections=m_pooled,
                    encoder_hidden_states=m_prompt,
                    txt_ids=text_ids,
                    img_ids=latent_img_ids,
                    return_dict=False,
                )[0]

            # Flow Matching Euler Step
            m_latents = pipe.scheduler.step(noise_pred, t, m_latents, return_dict=False)[0]

        # --- 4. Decode Micro-Batch ---
        # Unpack Flux latents (2x2) back to standard shapes
        m_latents_unpacked = pipe._unpack_latents(m_latents, height, width, pipe.vae_scale_factor)
        
        # VAE Scaling/Shift
        m_latents_unpacked = (m_latents_unpacked / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        
        with torch.no_grad():
            m_imgs = pipe.vae.decode(m_latents_unpacked, return_dict=False)[0]
            # Post-process and cast to float to avoid BFloat16 NumPy errors
            m_imgs = pipe.image_processor.postprocess(m_imgs, output_type="pt").float()
            all_images.append(m_imgs)

        # --- 5. Clean up Micro-Batch ---
        del m_latents, m_prompt, m_pooled, noise_pred
        torch.cuda.empty_cache()

    # Final concat of all micro-batches
    return torch.cat(all_images, dim=0)

def setup_models(
    config,
    prompt,
    generator,
    initial_values_dir,
    device,
):
    if config.model.name == "sd-1-5":
        init_latent = torch.randn(1, 4, 64, 64, device=device, generator=generator)
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
        scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        solution_schema = {
            "latent": (4, 64, 64),
            "prompt_embedding": (2, 77, 768),
            "image": (3, 512, 512),
        }
        generate_images_from_latents_fn = partial(generate_images_from_latents,
            num_inference_steps=config.model.num_inference_steps,
            guidance_scale=config.model.guidance_scale,
            scheduler=scheduler,
            unet=pipe.unet,
            vae=pipe.vae,
        )        
        
        init_prompt_embedding = get_text_embeddings(
            prompt=prompt,
            batch_size=1,
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            with_grad=False,
        ) # [batch_size * 2, 77, 768]
        init_latent *= scheduler.init_noise_sigma
        init_image = generate_images_from_latents_fn(
            latents=init_latent,
            prompt_embedding=init_prompt_embedding
        )
        
        # Save seed image, latent and prompt embedding
        torch.save(init_prompt_embedding.cpu(), os.path.join(initial_values_dir, "initial_prompt_embedding.pt"))
        torch.save(init_latent.cpu(), os.path.join(initial_values_dir, "initial_latent.pt"))
        save_image(init_image[0], os.path.join(initial_values_dir, "initial_image.png"))
        
        # Create initial solution dict
        initial_solution = {
            "latent": init_latent.cpu(),
            "prompt_embedding": init_prompt_embedding.cpu(),
        }
        
        del init_image, init_latent, init_prompt_embedding
        text_ids = None
        latent_img_ids = None
        gc.collect()
            
    elif config.model.name == "sdxl-turbo":
        init_latent = torch.randn(1, 4, 64, 64, device=device, generator=generator)
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
        pipe.to(device)
        solution_schema = {
            "latent": (4, 64, 64),
            "prompt_embedding": (77, 2048),
            "pooled_prompt_embedding": (1280,),
            "image": (3, 512, 512),
        }
    
        generate_images_from_latents_fn = partial(generate_images_from_latents_sdxl_turbo,
            num_inference_steps=config.model.num_inference_steps,
            scheduler=pipe.scheduler,
            unet=pipe.unet,
            vae=pipe.vae,
        )
        
        init_prompt_embedding, init_pooled_prompt_embedding = get_sdxl_prompt_embeddings(
            prompt=prompt,
            tokenizer_1=pipe.tokenizer,          
            tokenizer_2=pipe.tokenizer_2,
            text_encoder_1=pipe.text_encoder,    
            text_encoder_2=pipe.text_encoder_2,
            device=device
        )

        init_image = generate_images_from_latents_fn(
            latents=init_latent,
            prompt_embedding=init_prompt_embedding,
            pooled_prompt_embedding=init_pooled_prompt_embedding,
            time_ids=[1024, 1024, 0, 0, 512, 512],
        )
        
        # Save seed image, latent and prompt embedding
        torch.save(init_prompt_embedding.cpu(), os.path.join(initial_values_dir, "initial_prompt_embedding.pt"))
        torch.save(init_latent.cpu(), os.path.join(initial_values_dir, "initial_latent.pt"))
        torch.save(init_pooled_prompt_embedding.cpu(), os.path.join(initial_values_dir, "initial_pooled_prompt_embedding.pt"))
        save_image(init_image[0], os.path.join(initial_values_dir, "initial_image.png"))
        
        # Create initial solution dict
        initial_solution = {
            "latent": init_latent.cpu(),
            "prompt_embedding": init_prompt_embedding.cpu(),
            "pooled_prompt_embedding": init_pooled_prompt_embedding.cpu(),
        }
        
        del init_image, init_latent, init_prompt_embedding, init_pooled_prompt_embedding    
        gc.collect()
        scheduler = None # already part of the pipeline object
        text_ids = None
        latent_img_ids = None
    
    elif config.model.name == "flux":
        # Set up dir to write quanto cache
        # os.makedirs("/tmp/quanto_cache", exist_ok=True)
        
        # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

        # Quantize the transformer only (this is where the bulk of the VRAM is)
        # quantize(pipe.transformer, weights=qfloat8)
        # freeze(pipe.transformer)
        
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16
        )

        # 2. Load the Pipeline and pass the pre-quantized transformer
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        
        # Flux uses packed latents; standard height/width are divided by 2
        height, width = 1024, 1024 # Standard for Flux
        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents_h = 2 * (int(height) // (pipe.vae_scale_factor * 2))
        latents_w = 2 * (int(width) // (pipe.vae_scale_factor * 2))
        
        # Initial Noise
        init_latent = torch.randn(1, num_channels_latents, latents_h, latents_w, 
                                device=device, generator=generator, dtype=torch.bfloat16)
        
        # Pack and prepare IDs
        init_latent = pipe._pack_latents(init_latent, 1, num_channels_latents, latents_h, latents_w)
        latent_img_ids = pipe._prepare_latent_image_ids(1, latents_h // 2, latents_w // 2, 
                                                    device=device, dtype=torch.float32)
        
        # Encode Prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(prompt, device=device)

        solution_schema = {
            "latent": init_latent.squeeze(0).shape,
            "prompt_embedding": prompt_embeds.squeeze(0).shape,
            "pooled_prompt_embedding": pooled_prompt_embeds.squeeze(0).shape,
            "image": (3, height, width),
        }

        # Partial for generation
        generate_images_from_latents_fn = partial(generate_images_from_latents_flux,
            pipe=pipe,
            text_ids=text_ids,
            latent_img_ids=latent_img_ids,
            num_inference_steps=config.model.num_inference_steps,
            guidance_scale=config.model.guidance_scale,
            micro_batch_size=min(config.model.micro_batch_size, config.batch_size),
            height=height,
            width=width,
            device=device
        )

        init_image = generate_images_from_latents_fn(
            latents=init_latent,
            prompt_embedding=prompt_embeds,
            pooled_prompt_embedding=pooled_prompt_embeds
        )

        # Save values
        torch.save(prompt_embeds.cpu(), os.path.join(initial_values_dir, "initial_prompt_embedding.pt"))
        torch.save(init_latent.cpu(), os.path.join(initial_values_dir, "initial_latent.pt"))
        torch.save(pooled_prompt_embeds.cpu(), os.path.join(initial_values_dir, "initial_pooled_prompt_embedding.pt"))
        save_image(init_image[0], os.path.join(initial_values_dir, "initial_image.png"))

        initial_solution = {
            "latent": init_latent.cpu(),
            "prompt_embedding": prompt_embeds.cpu(),
            "pooled_prompt_embedding": pooled_prompt_embeds.cpu(),
        }

        # --- VRAM Cleanup ---
        # Deleting encoders and tokenizers as they are no longer needed in the QD loop
        del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2
        gc.collect()
        torch.cuda.empty_cache()
        
        scheduler = pipe.scheduler
    
    return pipe, scheduler, generate_images_from_latents_fn, solution_schema, initial_solution, text_ids, latent_img_ids