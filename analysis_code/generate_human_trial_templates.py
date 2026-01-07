import os
import random
import json
import textwrap
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List

# --- AESTHETIC CONFIG ---
OUTER_MARGIN = 100      
SECTION_GAP = 100       
HEADER_HEIGHT = 400     
BACKGROUND_COLOR = "white"
TEXT_COLOR = "black"

def get_experiment_subdir(base_path: str) -> str:
    if not os.path.exists(base_path): return None
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs: return None
    subdirs.sort() 
    return os.path.join(base_path, subdirs[-1])

def get_system_font_path():
    common_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for font in common_fonts:
        if os.path.exists(font): return font
    return None

def fit_text_to_header(draw, text, font_path, max_width, max_height):
    font_size = int(max_height * 0.6) 
    min_font_size = 20
    safe_width = max_width - 100 
    safe_height = max_height - 20

    if not font_path: return ImageFont.load_default(), text

    while font_size > min_font_size:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            return ImageFont.load_default(), text

        avg_char_width = font.getlength("x") or 10
        chars_per_line = int(safe_width / avg_char_width)
        lines = textwrap.wrap(text, width=chars_per_line)
        wrapped_text = "\n".join(lines)
        
        left, top, right, bottom = draw.multiline_textbbox((0,0), wrapped_text, font=font, align='center')
        if (right - left) <= safe_width and (bottom - top) <= safe_height:
            return font, wrapped_text
        font_size -= 2
    return ImageFont.truetype(font_path, min_font_size), text

def generate_human_trial_templates(
    root_dir, 
    seeds, 
    algorithm_dicts, 
    main_algo_key, 
    prompts_dict, 
    output_dir,
    k=16,
    metric="fitness"
):
    # Adjust Output Directory based on metric
    # e.g. analysis_output/human_trials_dns
    final_output_dir = f"{output_dir}_{metric}" if metric != "fitness" else output_dir
    os.makedirs(final_output_dir, exist_ok=True)

    print("\n-------------------------------------------------------------------------")
    print(f"      Generating Human Trial Templates ({metric.upper()})")
    print(f"      Output: {final_output_dir}")
    print("-------------------------------------------------------------------------")

    font_path = get_system_font_path()
    trial_metadata = {}
    trial_counter = 1
    
    baselines = [key for key in algorithm_dicts.keys() if key != main_algo_key]
    if not baselines:
        print("No baselines found.")
        return

    # Filename to look for
    grid_filename = f"top_{k}_{metric}_grid.png"

    for seed in seeds:
        if seed not in prompts_dict: continue
        prompt_text = prompts_dict[seed]
        
        # 1. Load Main Algo Grid
        main_path_init = os.path.join(root_dir, str(seed), algorithm_dicts[main_algo_key]["path"])
        main_exp_dir = get_experiment_subdir(main_path_init)
        if not main_exp_dir: continue
        
        main_grid_path = os.path.join(main_exp_dir, "images", grid_filename)
        if not os.path.exists(main_grid_path): continue
        try: main_grid_img = Image.open(main_grid_path).convert('RGB')
        except: continue

        # 2. Loop over Baselines
        for base_key in baselines:
            base_config = algorithm_dicts[base_key]
            base_path_init = os.path.join(root_dir, str(seed), base_config["path"])
            base_exp_dir = get_experiment_subdir(base_path_init)
            if not base_exp_dir: continue
            
            base_grid_path = os.path.join(base_exp_dir, "images", grid_filename)
            if not os.path.exists(base_grid_path): continue
            try: base_grid_img = Image.open(base_grid_path).convert('RGB')
            except: continue

            # 3. Create Comparison
            is_main_left = random.choice([True, False])
            left_img = main_grid_img if is_main_left else base_grid_img
            right_img = base_grid_img if is_main_left else main_grid_img
            
            total_w = OUTER_MARGIN + left_img.width + SECTION_GAP + right_img.width + OUTER_MARGIN
            total_h = HEADER_HEIGHT + max(left_img.height, right_img.height) + OUTER_MARGIN
            
            final_sheet = Image.new('RGB', (total_w, total_h), BACKGROUND_COLOR)
            final_sheet.paste(left_img, (OUTER_MARGIN, HEADER_HEIGHT))
            final_sheet.paste(right_img, (OUTER_MARGIN + left_img.width + SECTION_GAP, HEADER_HEIGHT))
            
            draw = ImageDraw.Draw(final_sheet)
            font, final_text = fit_text_to_header(draw, f'"{prompt_text}"', font_path, total_w, HEADER_HEIGHT)
            
            # Center text
            left, top, right, bottom = draw.multiline_textbbox((0,0), final_text, font=font, align='center')
            x_pos = (total_w - (right-left)) / 2
            y_pos = (HEADER_HEIGHT - (bottom-top)) / 2
            
            draw.multiline_text((x_pos, y_pos), final_text, font=font, fill=TEXT_COLOR, align="center")
            
            trial_id = f"t{trial_counter:04}"
            save_path = os.path.join(final_output_dir, f"{trial_id}.png")
            final_sheet.save(save_path)
            
            trial_metadata[trial_id] = {
                "seed": seed,
                "prompt": prompt_text,
                "metric": metric,
                "main_algo": algorithm_dicts[main_algo_key]["label"],
                "baseline_algo": base_config["label"],
                "left_algo": algorithm_dicts[main_algo_key]["label"] if is_main_left else base_config["label"],
                "right_algo": base_config["label"] if is_main_left else algorithm_dicts[main_algo_key]["label"],
            }
            trial_counter += 1
            
    with open(os.path.join(final_output_dir, "trial_key.json"), 'w') as f:
        json.dump(trial_metadata, f, indent=4)
        
    print(f"Generated {trial_counter-1} templates in {final_output_dir}")