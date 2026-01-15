import os
import shutil
import sys
from pathlib import Path

# --- CONFIGURATION ---
# Source: The new results you want to merge IN
# SRC_ROOT = Path("/workspace/src/hpc_flux/container_2026-01-07_115243_5205197/output/sdxl-turbo")
SRC_ROOT = Path("/workspace/src/output/sdxl-turbo")

# Destination: Your main results folder
DEST_ROOT = Path("/workspace/src/hpc_results/output/sdxl-turbo")

# Set this to True to DELETE the source files after copying (Move operation)
# Set to False to keep them (Copy operation)
DELETE_SOURCE_AFTER_COPY = True 

def merge_folders(src_dir, dst_dir):
    """
    Recursively copies contents from src_dir to dst_dir.
    Merges directories if they already exist.
    """
    if not src_dir.exists():
        return

    # Ensure destination exists
    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in os.listdir(src_dir):
        s = src_dir / item
        d = dst_dir / item
        
        if s.is_dir():
            # Recurse if directory
            merge_folders(s, d)
        else:
            # Copy file (overwrite if exists)
            shutil.copy2(s, d)
            # print(f"  Copied: {s.name}")

def main():
    print(f"Source:      {SRC_ROOT}")
    print(f"Destination: {DEST_ROOT}")
    print("-" * 50)

    if not SRC_ROOT.exists():
        print(f"!! Error: Source path does not exist:\n{SRC_ROOT}")
        sys.exit(1)

    # Loop through seeds 1 to 36
    count = 0
    for i in range(1, 37):
        seed_src = SRC_ROOT / str(i)
        seed_dst = DEST_ROOT / str(i)

        if seed_src.exists():
            print(f"Merging Seed {i}...")
            merge_folders(seed_src, seed_dst)
            count += 1
            
            # Optional: Delete source after successful merge
            if DELETE_SOURCE_AFTER_COPY:
                shutil.rmtree(seed_src)
                print(f"  Deleted source seed {i}")

    print("-" * 50)
    print(f"Complete. Processed {count} seeds.")

if __name__ == "__main__":
    main()