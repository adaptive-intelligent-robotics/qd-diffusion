import jax.numpy as jnp

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
