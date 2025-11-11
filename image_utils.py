
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def show_image(image):
    image_array = np.array(image.cpu())
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    image_array = (image_array * 255).astype(np.uint8)
    image_array = np.transpose(image_array, (1, 2, 0))
    image_plot = Image.fromarray(image_array, mode="RGB")

    # Show with corrected aspect ratio
    plt.figure(figsize=(10, 2))  # Adjust figsize to better match the image shape
    plt.imshow(image_plot)
    plt.axis('off')  # Optional: hides axis for better viewing
    plt.tight_layout()
    plt.show()

def save_image(image, path):
    image_array = np.array(image.cpu())
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    image_array = (image_array * 255).astype(np.uint8)
    image_array = np.transpose(image_array, (1, 2, 0))
    image_plot = Image.fromarray(image_array, mode="RGB")
    image_plot.save(path)
    

def load_images_from_folder(folder, max_images=None):
    print("Loading images...")

    supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG"]
    image_paths = []
    for ext in supported_extensions:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))

    if max_images:
        image_paths = image_paths[:max_images]

    images = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")  # Important: force RGB
                images.append(img.copy())  # .copy() keeps image after context closes
        except Exception as e:
            print(f"Error loading {path}: {e}")

    return images
