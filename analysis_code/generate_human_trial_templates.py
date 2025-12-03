import os
import glob
import random
import json
import sys
import textwrap
sys.path.append('/workspace/src/prompts')

from prompts import prompts_dict
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 1. Get the absolute path to the folder where this script is currently sitting
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to get the Repository Root
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# 3. Define paths relative to the Repo Root
ROOT_DATA_DIR = os.path.join(REPO_ROOT, "results/final_images") 
OUTPUT_DIR = os.path.join(REPO_ROOT, "figures/human_trials")

# Algorithm Names
MY_ALGO_NAME = "imagebreeder"
BASELINE_ALGOS = ["gaussian-noise"]

# List of prompts (must match your folder names exactly)
PROMPTS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10","11", "12", "13", "14", "15", "16",
              "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
              "31", "32", "33", "34", "35", "36"]

# The directory name inside the algorithm folder containing the pngs
FINAL_IMG_DIR = ""

# Image Settings
THUMBNAIL_SIZE = (512, 512) 

# Aesthetic Settings
GRID_PADDING = 10       # White space between images in the 4x4 grid
OUTER_MARGIN = 100      # Space on Left, Right, and Bottom edges
SECTION_GAP = 100       # White space between the Left and Right grids
HEADER_HEIGHT = 400     # Space for the text at the top
BACKGROUND_COLOR = "white"
TEXT_COLOR = "black"
FONT_SIZE = 60
FONT_PATH = None  # Set to None to auto-detect system font for autosizing

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_system_font_path():
    """
    Attempts to find a standard system font (Arial/Helvetica/DejaVu)
    so autosizing works out of the box.
    """
    common_fonts = [
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        # Mac
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Tahoma.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    ]
    
    for font in common_fonts:
        if os.path.exists(font):
            return font
    return None

def fit_text_to_header(draw, text, font_path, max_width, max_height):
    """
    Finds the best font size and line wrapping to fill the header area.
    Returns: (font_object, wrapped_text_string)
    """
    # Start with a large font size (relative to header height)
    font_size = int(max_height * 0.6) 
    min_font_size = 20
    
    # Safety margin so text doesn't touch edges
    safe_width = max_width - 100 
    safe_height = max_height - 20

    if not font_path:
        return ImageFont.load_default(), text

    while font_size > min_font_size:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            return ImageFont.load_default(), text

        # 1. Estimate characters per line based on average char width
        # (This is an approximation, but fast)
        avg_char_width = font.getlength("x")
        if avg_char_width == 0: avg_char_width = 10 # fallback
        
        chars_per_line = int(safe_width / avg_char_width)
        
        # 2. Wrap the text
        lines = textwrap.wrap(text, width=chars_per_line)
        wrapped_text = "\n".join(lines)
        
        # 3. Calculate total size of this multi-line block
        left, top, right, bottom = draw.multiline_textbbox((0,0), wrapped_text, font=font, align='center')
        text_w = right - left
        text_h = bottom - top

        # 4. Check if it fits
        if text_w <= safe_width and text_h <= safe_height:
            return font, wrapped_text
        
        # If not, reduce font and try again
        font_size -= 2

    # Fallback to min size
    return ImageFont.truetype(font_path, min_font_size), text

def get_image_paths(folder_path):
    """
    Retrieves all .png images from a folder. 
    Returns the first 16 found.
    """
    search_path = os.path.join(folder_path, "*.png")
    images = glob.glob(search_path)
    # Sort to ensure consistent ordering if needed, or shuffle if you prefer internal randomness
    images.sort() 
    
    if len(images) < 16:
        print(f"WARNING: Found less than 16 images in {folder_path}. Filling with placeholders.")
    
    return images[:16]

def create_4x4_grid(image_paths, thumb_size, padding):
    """
    Stitches 16 images into a single 4x4 image object.
    """
    # Calculate total grid size
    grid_w = (thumb_size[0] * 4) + (padding * 3)
    grid_h = (thumb_size[1] * 4) + (padding * 3)
    
    grid_img = Image.new('RGB', (grid_w, grid_h), BACKGROUND_COLOR)
    
    for idx, img_path in enumerate(image_paths):
        if idx >= 16: break
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(thumb_size)
            
            # Calculate x, y position
            col = idx % 4
            row = idx // 4
            
            x = col * (thumb_size[0] + padding)
            y = row * (thumb_size[1] + padding)
            
            grid_img.paste(img, (x, y))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    return grid_img

def draw_text_centered(draw, text, font, canvas_width, y_pos):
    """
    Draws text centered horizontally on the canvas.
    """
    # Get text bounding box to calculate width
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w = right - left
    text_h = bottom - top
    
    x_pos = (canvas_width - text_w) / 2
    draw.text((x_pos, y_pos), text, font=font, fill=TEXT_COLOR)

# ==========================================
# 3. MAIN SCRIPT
# ==========================================

def main():
    # ... (Directory creation code remains the same) ...
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # NEW: Try to find a valid font path automatically if one isn't set
    global FONT_PATH
    if FONT_PATH is None:
        FONT_PATH = get_system_font_path()
        if FONT_PATH:
            print(f"Autosizing enabled using system font: {FONT_PATH}")
        else:
            print("WARNING: No .ttf font found. Autosizing disabled (using fixed default font).")

    trial_counter = 1
    trial_metadata = {}

    print("--- Starting Generation ---")

    for prompt in PROMPTS:
        for baseline in BASELINE_ALGOS:
            
            # ... (Path loading and Grid creation code remains the same) ...
            
            # [PASTE THE PREVIOUS GRID CREATION CODE HERE]
            # (Assuming you have my_grid, base_grid setup from previous step)
            
            # --- CODE CONTINUES HERE ---
            
            my_path = os.path.join(ROOT_DATA_DIR, prompt, MY_ALGO_NAME, FINAL_IMG_DIR)
            base_path = os.path.join(ROOT_DATA_DIR, prompt, baseline, FINAL_IMG_DIR)
            
            if not os.path.exists(my_path) or not os.path.exists(base_path): continue
            
            my_imgs_paths = get_image_paths(my_path)
            base_imgs_paths = get_image_paths(base_path)
            
            if not my_imgs_paths or not base_imgs_paths: continue

            my_grid = create_4x4_grid(my_imgs_paths, THUMBNAIL_SIZE, GRID_PADDING)
            base_grid = create_4x4_grid(base_imgs_paths, THUMBNAIL_SIZE, GRID_PADDING)

            is_mine_left = random.choice([True, False])
            left_grid = my_grid if is_mine_left else base_grid
            right_grid = base_grid if is_mine_left else my_grid

            total_w = OUTER_MARGIN + left_grid.width + SECTION_GAP + right_grid.width + OUTER_MARGIN
            total_h = HEADER_HEIGHT + max(left_grid.height, right_grid.height) + OUTER_MARGIN

            final_sheet = Image.new('RGB', (total_w, total_h), BACKGROUND_COLOR)
            final_sheet.paste(left_grid, (OUTER_MARGIN, HEADER_HEIGHT))
            final_sheet.paste(right_grid, (OUTER_MARGIN + left_grid.width + SECTION_GAP, HEADER_HEIGHT))

            # --- UPDATED TEXT SECTION ---
            draw = ImageDraw.Draw(final_sheet)
            prompt_title = prompts_dict[int(prompt)]
            title_text = f'"{prompt_title}"'
            
            # DYNAMICALLY CALCULATE FONT SIZE HERE
           # 1. Calculate best font and wrapping
            font, final_text = fit_text_to_header(draw, f'"{prompt_title}"', FONT_PATH, total_w, HEADER_HEIGHT)
            
            # 2. Calculate position to center the multiline block
            # Get the bounding box of the final wrapped text
            left, top, right, bottom = draw.multiline_textbbox((0,0), final_text, font=font, align='center')
            text_width = right - left
            text_height = bottom - top
            
            x_pos = (total_w - text_width) / 2
            y_pos = (HEADER_HEIGHT - text_height) / 2
            
            # 3. Draw
            draw.multiline_text(
                (x_pos, y_pos), 
                final_text, 
                font=font, 
                fill=TEXT_COLOR, 
                align="center"
            )
            
            # ... (Save file and logging code remains the same) ...
            
            trial_id = f"t{trial_counter:03}"
            filename = f"{trial_id}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            final_sheet.save(save_path)
            
            trial_metadata[trial_id] = {
                "prompt": prompt,
                "baseline_algo": baseline,
                "my_algo_side": "left" if is_mine_left else "right",
                "left_content": MY_ALGO_NAME if is_mine_left else baseline,
                "right_content": baseline if is_mine_left else MY_ALGO_NAME
            }
            print(f"Generated {trial_id} ({prompt})")
            trial_counter += 1

    # ... (JSON dump code remains the same) ...
    json_path = os.path.join(OUTPUT_DIR, "trial_key.json")
    with open(json_path, 'w') as f:
        json.dump(trial_metadata, f, indent=4)
        
if __name__ == "__main__":
    main()