import os
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
# BASE_ROOT = Path("/workspace/src/output/sdxl-turbo")
BASE_ROOT = Path("/workspace/src/hpc_results/output/sdxl-turbo")

EXPECTED_SEEDS = range(1, 37)  # Seeds 1-36


ALGORITHM_PATHS = {
    "dns_svd": "dns/multi-hidden-layer-multi-svd-alpha-10",
    "cvt-map-elites": "cvt-map-elites/multi-hidden-layer-multi-svd-alpha-10",
    "ablation": "dns/multi-hidden-layer-multi-svd-threshold-0-alpha-10",
    "cma-es": "dns/multi-sep-cma-es",
    "adam": "dns/multi-adam",
    "qdhf": "qdhf/multi-hidden-layer-multi-svd-alpha-10",
    "dns_random": "dns/random-seeds",
    "dns_hidden_gaussian": "dns/multi-hidden-layer-gaussian",
    "dns_mixed": "dns/mixed-all",
    "dns_gaussian": "dns/multi-gaussian",
    "dns_isoline": "dns/multi-isoline",
}
# 
# {
#     "dns_svd": "dns/multi-hidden-layer-multi-svd-alpha-20",
#     "dns_svd_alpha_1": "dns/multi-hidden-layer-multi-svd-alpha-1",
#     "dns_svd_alpha_10": "dns/multi-hidden-layer-multi-svd-alpha-10",
#     "dns_svd_alpha_5": "dns/multi-hidden-layer-multi-svd-alpha-5",
#     "dns_random": "dns/random-seeds",
#     "dns_hidden_gaussian": "dns/multi-hidden-layer-gaussian",
#     "dns_gaussian": "dns/multi-gaussian",
#     "dns_mixed": "dns/mixed-all",
#     "dns_isoline": "dns/multi-isoline",
#     "cvt-map-elites": "cvt-map-elites/multi-hidden-layer-multi-svd-alpha-20",
#     "qdhf": "qdhf/multi-hidden-layer-multi-svd-alpha-20",
#     "adam": "dns/multi-adam",
#     "cma-es": "dns/multi-sep-cma-es",
#     "ablation": "dns/multi-hidden-layer-multi-svd-threshold-0-alpha-20",
#     "ablation-alpha-10": "dns/multi-hidden-layer-multi-svd-threshold-0-alpha-10",
# }

# ALGORITHM_PATHS = {
#     "dns_svd_latent": "dns/sdxl-hidden-layer-latent-svd",
#     "dns_svd_pooled": "dns/sdxl-hidden-layer-pooled-svd",
#     "dns_svd_prompt": "dns/sdxl-hidden-layer-prompt-embed-svd",
# }  

def count_seeds_for_algorithm(algo_path):
    """
    Count how many seeds exist for a given algorithm path.
    Returns a set of seed numbers that exist.
    """
    found_seeds = set()
    duplicate_seeds = set()
    
    # Iterate through all prompt directories (which are numbered as seeds)
    for prompt_dir in BASE_ROOT.iterdir():
        if not prompt_dir.is_dir():
            continue
        
        # Check if the directory name is a number (seed)
        try:
            seed_num = int(prompt_dir.name)
        except ValueError:
            continue
        
        # Check if this algorithm path exists for this seed
        algo_full_path = prompt_dir / algo_path
        if algo_full_path.exists() and algo_full_path.is_dir():
            # Optionally, check if there are any experiment folders inside
            experiments = [e for e in algo_full_path.iterdir() if e.is_dir()]
            if experiments:
                found_seeds.add(seed_num)
            if len(experiments) > 1:
                duplicate_seeds.add(seed_num)
    
    return found_seeds, duplicate_seeds

def main():
    print("=" * 80)
    print("ALGORITHM SEED COUNT REPORT")
    print("=" * 80)
    print(f"Base Directory: {BASE_ROOT}")
    print(f"Expected Seeds: {min(EXPECTED_SEEDS)}-{max(EXPECTED_SEEDS)} ({len(EXPECTED_SEEDS)} total)")
    print("=" * 80)
    print()
    
    results = {}
    
    for algo_name, algo_path in ALGORITHM_PATHS.items():
        found_seeds, duplicate_seeds = count_seeds_for_algorithm(algo_path)
        missing_seeds = set(EXPECTED_SEEDS) - found_seeds
        
        results[algo_name] = {
            'path': algo_path,
            'found': found_seeds,
            'missing': missing_seeds,
            'count': len(found_seeds),
            'duplicates': duplicate_seeds,
        }
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for algo_name in ALGORITHM_PATHS.keys():
        info = results[algo_name]
        status = "✓ COMPLETE" if len(info['missing']) == 0 else f"✗ INCOMPLETE"
        print(f"{algo_name:25s} | {info['count']:2d}/{len(EXPECTED_SEEDS)} seeds | {status} | Duplicates: {len(info['duplicates'])}")
    
    # Print detailed missing seeds for incomplete algorithms
    print("\n" + "=" * 80)
    print("MISSING SEEDS DETAILS")
    print("=" * 80)
    
    incomplete_found = False
    for algo_name in ALGORITHM_PATHS.keys():
        info = results[algo_name]
        if info['missing']:
            incomplete_found = True
            missing_sorted = sorted(info['missing'])
            print(f"\n{algo_name}:")
            print(f"  Path: {info['path']}")
            print(f"  Found: {info['count']}/{len(EXPECTED_SEEDS)} seeds")
            print(f"  Missing Seeds: {missing_sorted}")
    
    print("\n" + "=" * 80)
    print("DUPLICATE SEEDS DETAILS")
    print("=" * 80)
    for algo_name in ALGORITHM_PATHS.keys():
        info = results[algo_name]
        if info['duplicates']:
            duplicate_sorted = sorted(info['duplicates'])
            print(f"\n{algo_name}:")
            print(f"  Path: {info['path']}")
            print(f"  Duplicate Seeds: {duplicate_sorted}")
    
    if not incomplete_found:
        print("\n✓ All algorithms have complete seed coverage!")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
