import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
)
from functools import partial
from dreamsim import dreamsim

def dino_feature_extractor(image_np, processor, model, projection_matrix, device):
    if image_np.ndim == 3 and image_np.shape[0] == 3:
        # Convert from CHW → HWC
        image_np = np.transpose(image_np, (1, 2, 0))
    # Convert NumPy float32 image in [-1, 1] or [0, 1] to [0, 255] uint8
    if image_np.dtype != np.uint8:
        if image_np.min() < 0:
            # Assume [-1, 1] → scale to [0, 255]
            image_np = ((image_np + 1.0) / 2.0 * 255).clip(0, 255)
        else:
            # Assume [0, 1] → scale to [0, 255]
            image_np = (image_np * 255).clip(0, 255)
        image_np = image_np.astype(np.uint8)

    # Convert to PIL
    image_pil = Image.fromarray(image_np, mode='RGB')
   
    # Process and get embedding
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
    embedding = output.last_hidden_state.mean(dim=1).squeeze(0).to(device)
    
    # Project to 2D
    if projection_matrix is not None:
        descriptors = torch.matmul(projection_matrix, embedding)
    else:
        descriptors = embedding
        
    return descriptors.cpu().numpy() # Return as NumPy array


def dream_sim_feature_extractor(image_np, processor, model, projection_matrix, device):
    if image_np.ndim == 3 and image_np.shape[0] == 3:
        # Convert from CHW → HWC
        image_np = np.transpose(image_np, (1, 2, 0))
    # Convert NumPy float32 image in [-1, 1] or [0, 1] to [0, 255] uint8
    if image_np.dtype != np.uint8:
        if image_np.min() < 0:
            # Assume [-1, 1] → scale to [0, 255]
            image_np = ((image_np + 1.0) / 2.0 * 255).clip(0, 255)
        else:
            # Assume [0, 1] → scale to [0, 255]
            image_np = (image_np * 255).clip(0, 255)
        image_np = image_np.astype(np.uint8)

    # Convert to PIL
    image_pil = Image.fromarray(image_np, mode='RGB')
   
    # Process and get embedding
    inputs = processor(image_pil).to(device)
    embedding = model.embed(inputs).squeeze(0).to(device)
    
    # Project to 2D
    if projection_matrix is not None:
        descriptors = torch.matmul(projection_matrix, embedding)
    else:
        descriptors = embedding
        
    return descriptors.cpu().numpy() # Return as NumPy array


def get_descriptor_fun(
    descriptor_name: str,
    project_desc: bool,
    projection_dim: int = 2,
    generator: torch.Generator = None,
    save_dir: str = None,
):

    if descriptor_name == "dino-v2":
        dino_model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to("cuda")
        dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        
        if project_desc == True:
            projection_matrix = torch.randn(projection_dim, 768, device="cuda", generator=generator)
            torch.save(projection_matrix, f"{save_dir}/descriptor_projection_matrix.pt")
            descriptor_dim = projection_dim
        else:
            projection_matrix = None
            descriptor_dim = 768
            
        descriptor_fn =  partial(
            dino_feature_extractor,
            processor=dino_processor,
            model=dino_model,
            projection_matrix=projection_matrix,
            device="cuda"
        )

    elif descriptor_name == "dino-v3":
        print("!!! ERROR!!: Dino-v3 descriptor not currently supported. Please use v2 or dreamsim descriptor instead. !!!")
        # dino_model = AutoModel.from_pretrained("facebook/dinov3-convnext-tiny-pretrain-lvd1689m").eval().to("cuda")
        # dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
        
        # if project_desc == True:
        #     projection_matrix = torch.randn(projection_dim, 1024, device="cuda")
        # else:
        #     projection_matrix = None
            
        # descriptor_fn =  partial(
        #     dino_feature_extractor,
        #     processor=dino_processor,
        #     model=dino_model,
        #     projection_matrix=projection_matrix,
        #     device="cuda"
        # )

    elif descriptor_name == "dreamsim":
        dreamsim_dino_model, preprocess = dreamsim(pretrained=True, dreamsim_type="dino_vitb16")
        if project_desc == True:
            projection_matrix = torch.randn(projection_dim, 768, device="cuda", generator=generator)
            torch.save(projection_matrix, f"{save_dir}/descriptor_projection_matrix.pt")
            descriptor_dim = projection_dim
        else:
            projection_matrix = None
            descriptor_dim = 768
        
        descriptor_fn =  partial(
            dream_sim_feature_extractor,
            processor=preprocess,
            model=dreamsim_dino_model,
            projection_matrix=projection_matrix,
            device="cuda"
        )
    
    return descriptor_fn, descriptor_dim