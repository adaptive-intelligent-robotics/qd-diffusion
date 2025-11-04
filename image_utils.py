
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


def grid_plot_images(test_images):
    """
    Plot all images in test_images in a grid using matplotlib.
    Up to 1000 images, arranged 20 per row.
    Each image is displayed small to fit the grid nicely.
    """
    num_images = len(test_images)
    if num_images == 0:
        print("No images to display.")
        return
    
    # Determine grid size
    cols = 20
    rows = math.ceil(num_images / cols)
    
    # Create the figure
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows))  # adjust figsize for visibility
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    
    # Plot each image
    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(test_images[i])
            ax.axis("off")
        else:
            ax.axis("off")  # hide any unused subplots
    
    plt.tight_layout(pad=0.1)
    plt.show()