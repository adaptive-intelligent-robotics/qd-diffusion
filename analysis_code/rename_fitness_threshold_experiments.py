import os
import shutil
from pathlib import Path

# --- Configuration ---

# BASE_ROOT = Path("/workspace/src/hpc_results/output/sdxl-turbo")
# BASE_ROOT = Path("/workspace/src/output/sdxl-turbo")
BASE_ROOT = Path("/workspace/src/new_hpc_flux/output/flux")

EXP_NAME = "dns"
OLD_PARENT_NAME = "flux-hidden-layer-latent-svd"
NEW_PARENT_NAME = "flux-hidden-layer-latent-svd-threshold-0.2"
TARGET_STRING = "fitness_threshold: 0.2"

# Set to False to actually move files. Keep True to test safely first.
DRY_RUN = False 

def restructure_directories():
    print(f"--- Starting {'DRY RUN' if DRY_RUN else 'LIVE EXECUTION'} ---")
    
    # Iterate through PROMPT 1 to 36
    for prompt_key in range(1, 37):
        # Construct the path: .../sdxl-turbo/{PROMPT}/{EXP_NAME}/multi-hidden-layer-multi-svd/
        current_parent_dir = BASE_ROOT / str(prompt_key) / EXP_NAME / OLD_PARENT_NAME
        
        # Check if this directory actually exists
        if not current_parent_dir.exists():
            print(f"Skipping missing directory: {current_parent_dir}")
            continue

        # Iterate through experiment folders (e.g., 2026-01-01_073428_1)
        # We look for directories inside the current parent
        for experiment_dir in current_parent_dir.iterdir():
            if not experiment_dir.is_dir():
                continue

            yaml_path = experiment_dir / "config.yaml"
            
            # Check if config.yaml exists
            if not yaml_path.exists():
                print(f"[Warning] No config.yaml found in {experiment_dir}")
                continue

            # Read the yaml file to check for fitness_threshold
            try:
                content = yaml_path.read_text()
                
                if TARGET_STRING in content:
                    # Construct the new destination path
                    # .../sdxl-turbo/{PROMPT}/{EXP_NAME}/multi-hidden-layer-multi-svd-threshold-0/
                    new_parent_dir = current_parent_dir.parent / NEW_PARENT_NAME
                    destination_path = new_parent_dir / experiment_dir.name

                    if DRY_RUN:
                        print(f"[Would Move] {experiment_dir.name}")
                        print(f"    From: {current_parent_dir}")
                        print(f"    To:   {new_parent_dir}")
                    else:
                        # Create the new parent directory if it doesn't exist
                        new_parent_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Move the directory
                        try:
                            shutil.move(str(experiment_dir), str(destination_path))
                            print(f"[Moved] {experiment_dir.name} -> {new_parent_dir}")
                        except Exception as e:
                            print(f"[Error] Failed to move {experiment_dir.name}: {e}")
                            
            except Exception as e:
                print(f"[Error] reading {yaml_path}: {e}")

    print("--- Process Complete ---")

if __name__ == "__main__":
    restructure_directories()