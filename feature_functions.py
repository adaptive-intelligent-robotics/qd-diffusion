import jax.numpy as jnp
import numpy as np
from APDDv2.demo import get_score, load_model
import torch
from PIL import Image
import APDDv2.models.clip as clip
from functools import partial

def brightness(image):
    """Compute brightness by averaging pixel values."""
    return jnp.mean(image)  # No OpenCV needed!

def average_hue(image):
    """Compute an approximate average hue using JAX operations."""
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    max_c = jnp.max(image, axis=-1)
    min_c = jnp.min(image, axis=-1)
    delta = max_c - min_c

    hue = jnp.where(
        delta == 0, 0,
        jnp.where(max_c == r, (60 * ((g - b) / delta) + 360) % 360,
        jnp.where(max_c == g, (60 * ((b - r) / delta) + 120) % 360,
                             (60 * ((r - g) / delta) + 240) % 360
        )))
    
    return jnp.mean(hue) 

    
def artclip_score(image, preprocess_model, score_model, device):
    image =  Image.fromarray(np.array(image).astype('uint8'), 'RGB')
    image_input = preprocess_model(image).unsqueeze(0).to(device)
    artclip_score = score_model(image_input)
    _, score = get_score(None, artclip_score)
    return score



def get_artclip_score_model(feature_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess_model = clip.load('ViT-B/16', device)

    if feature_name == "artclip_score":
        model = load_model("APDDv2/model_weights/1.Score_reg_weight--e4-train0.4393-test0.6835_best.pth", device)
    elif feature_name == "theme_and_logic":
        model = load_model("APDDv2/model_weights/2.Theme and logic_reg_weight--e5-train0.3792-test0.5953_best.pth", device)
    elif feature_name == "creativity":
        model = load_model("APDDv2/model_weights/3.Creativity_reg_weight--e5-train0.4212-test0.7122_best.pth", device)
    elif feature_name == "layout_and_composition":
        model = load_model("APDDv2/model_weights/4.Layout and composition_reg_weight--e6-train0.2783-test0.6342_best.pth", device)
    elif feature_name == "space_and_perspective":
        model = load_model("APDDv2/model_weights/5.Space and perspective_reg_weight--e7-train0.2168-test0.5998_best.pth", device)
    elif feature_name == "sense_of_order":
        model = load_model("APDDv2/model_weights/6.The sense of order_reg_weight--e5-train0.3708-test0.6206_best.pth", device)
    elif feature_name == "light_and_shadow":
        model = load_model("APDDv2/model_weights/7.Light and shadow_reg_weight--e7-train0.1937-test0.6518_best.pth", device)
    elif feature_name == "color":
        model = load_model("APDDv2/model_weights/8.Color_reg_weight--e5-train0.2905-test0.5871_best.pth", device)
    elif feature_name == "details_and_texture":
        model = load_model("APDDv2/model_weights/9.Details and texture_reg_weight--e4-train0.4385-test0.7034_best.pth", device)
    elif feature_name == "overall":
        model = load_model("APDDv2/model_weights/10.The overall_reg_weight--e3-train0.5131-test0.6343_best.pth", device)
    elif feature_name == "mood":
        model = load_model("APDDv2/model_weights/11.Mood_reg_weight--e7-train0.3108-test0.7097_best.pth", device)
    else:
        raise ValueError(f"ArtCLIP Feature name {feature_name} not found")
    
    scoring_function = partial(artclip_score,
        preprocess_model=preprocess_model,  
        score_model=model,
        device=device
    )
    
    return scoring_function


def dino_feature_extractor(image_np, processor, model, projection_matrix, device):
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
    embedding = output.last_hidden_state.mean(dim=1).squeeze(0)  # (768,)
    
    # Project to 2D
    descriptors = projection_matrix @ embedding
    return descriptors.cpu().numpy()  # Return as NumPy array