import shutil
from pathlib import Path

# --- Configuration ---
# BASE_ROOT = Path("/workspace/src/hpc_results/output/sdxl-turbo")
BASE_ROOT = Path("/workspace/src/output/sdxl-turbo")

SUCCESS_MARKER = Path("repertoire/final_fitnesses.pt")

# Set to False to actually delete files. Keep True to test safely first.
DRY_RUN = False 

def cleanup_failed_experiments():
    print(f"--- Starting Cleanup ({'DRY RUN' if DRY_RUN else 'LIVE DELETION'}) ---\n")
    
    # 1. Iterate over PROMPT folders
    if not BASE_ROOT.exists():
        print(f"Error: Base path not found: {BASE_ROOT}")
        return

    for prompt_dir in BASE_ROOT.iterdir():
        if not prompt_dir.is_dir(): continue

        # 2. Iterate over ALGO folders (e.g., 'dns')
        for algo_dir in prompt_dir.iterdir():
            if not algo_dir.is_dir(): continue

            # 3. Iterate over EMITTER folders (e.g., 'multi-hidden-layer-svd')
            for emitter_dir in algo_dir.iterdir():
                if not emitter_dir.is_dir(): continue

                # 4. Inspect EXPERIMENT folders
                experiments = [e for e in emitter_dir.iterdir() if e.is_dir()]
                
                failed_experiments = []
                successful_experiments = []

                for exp_dir in experiments:
                    marker_path = exp_dir / SUCCESS_MARKER
                    
                    if marker_path.exists():
                        successful_experiments.append(exp_dir)
                    else:
                        failed_experiments.append(exp_dir)

                # --- Logic Checks ---
                
                # If there are no failed experiments, move to next emitter
                if not failed_experiments:
                    continue

                # WARNING CHECK: If we delete the failures, will anything remain?
                if len(successful_experiments) == 0:
                    print(f"⚠️  CRITICAL WARNING: All experiments failed in:")
                    print(f"    {emitter_dir}")
                    print(f"    (Deleting these would leave the emitter folder empty)\n")

                # Perform Deletion
                for failed_exp in failed_experiments:
                    if DRY_RUN:
                        print(f"[Would Delete] {failed_exp.name}")
                        print(f"    Path: {failed_exp}")
                        # visual separator for readability
                        if failed_exp == failed_experiments[-1]: print("") 
                    else:
                        try:
                            shutil.rmtree(failed_exp)
                            print(f"[Deleted] {failed_exp.name}")
                        except Exception as e:
                            print(f"[Error] Could not delete {failed_exp.name}: {e}")

    print("--- Cleanup Complete ---")

if __name__ == "__main__":
    cleanup_failed_experiments()