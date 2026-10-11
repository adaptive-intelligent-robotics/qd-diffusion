"""Shared experiment configuration for all analysis scripts.

Every script selects a model (``sdxl-turbo`` or ``flux``) and gets the same
set of algorithms, seeds, results directory and output directory from here.
Edit this file to add an algorithm or change the seeds.

Paths are resolved relative to the repository root (the parent of this folder).
Algorithms whose results are missing for a given model or seed are skipped by
the scripts, so it is safe to list an algorithm that has not been run yet.
"""

import argparse
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "main_results"
COCO_VAL_DIR = REPO_ROOT / "coco" / "val2017"

# Seeds that were run for each model.
SDXL_SEEDS: List[int] = list(range(1, 37))
FLUX_SEEDS: List[int] = [1, 4, 7, 10, 13, 16, 20, 22, 25, 28, 31, 34]

# Algorithm key -> display label. Keys are shared across models.
ALGORITHM_LABELS: Dict[str, str] = {
    "dns_svd": "Spectral-Guided QD-Diffusion (Ours)",
    "cvt-map-elites": "CVT MAP-Elites",
    "ablation": "Fitness Threshold Ablation",
    "cma-es": "Sep-CMA-ES",
    "cma-mae": "CMA-MAE",
    "adam": "Adam",
    "qdhf": "QDHF",
    "cads": "CADS",
    "particle-guidance": "Particle Guidance",
    "diverse-diffusion": "Diverse Diffusion",
    "dns_mixed": "ImageBreeder",
    "dns_random": "Random Seeds",
    "dns_hidden_gaussian": "Intermed.-layer Gaussian",
    "dns_gaussian": "Gaussian",
    "dns_isoline": "Isoline-DD",
}

# Algorithm key -> results sub-path (relative to <results>/<model>/<seed>/).
SDXL_PATHS: Dict[str, str] = {
    "dns_svd": "dns/multi-svd-hidden-layer",
    "cvt-map-elites": "cvt-map-elites/multi-svd-hidden-layer",
    "ablation": "dns/multi-svd-hidden-layer-threshold-0",
    "cma-es": "dns/multi-sep-cma-es",
    "cma-mae": "cma-mae/cma-mae",
    "adam": "dns/multi-adam",
    "qdhf": "qdhf/multi-svd-hidden-layer",
    "particle-guidance": "dns/particle-guidance",
    "diverse-diffusion": "dns/diverse-diffusion",
    "dns_mixed": "dns/mixed-all",
    "dns_random": "dns/random-seeds",
    "dns_hidden_gaussian": "dns/multi-hidden-layer-gaussian",
    "dns_gaussian": "dns/multi-gaussian",
    "dns_isoline": "dns/multi-isoline",
}

FLUX_PATHS: Dict[str, str] = {
    "dns_svd": "dns/flux-multi-svd",
    "cvt-map-elites": "cvt-map-elites/flux-multi-svd",
    "cma-es": "dns/multi-sep-cma-es",
    "cma-mae": "cma-mae/cma-mae",
    "qdhf": "qdhf/flux-multi-svd",
    "cads": "dns/cads-flux",
    "particle-guidance": "dns/particle-guidance-flux",
    "diverse-diffusion": "dns/diverse-diffusion",
    "dns_mixed": "dns/mixed-all",
    "dns_random": "dns/random-seeds",
    "dns_hidden_gaussian": "dns/flux-hidden-layer-multi-gaussian",
    "dns_gaussian": "dns/multi-gaussian",
    "dns_isoline": "dns/multi-isoline",
}


def _algorithms(paths: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    return {key: {"path": path, "label": ALGORITHM_LABELS[key]} for key, path in paths.items()}


MODELS: Dict[str, Dict] = {
    "sdxl-turbo": {
        "root_dir": str(RESULTS_DIR / "sdxl-turbo"),
        "output_dir": str(REPO_ROOT / "analysis_output"),
        "seeds": SDXL_SEEDS,
        "algorithms": _algorithms(SDXL_PATHS),
    },
    "flux": {
        "root_dir": str(RESULTS_DIR / "flux"),
        "output_dir": str(REPO_ROOT / "analysis_output_flux"),
        "seeds": FLUX_SEEDS,
        "algorithms": _algorithms(FLUX_PATHS),
    },
}

# The algorithm that all others are compared against in significance tests.
BASE_ALGO_KEY = "dns_svd"


def add_model_argument(parser: argparse.ArgumentParser) -> None:
    """Add the required ``--model`` option shared by all analysis scripts."""
    parser.add_argument("--model", required=True, choices=sorted(MODELS), help="Which model's results to analyse.")


def get_model_config(model: str) -> Dict:
    """Return the config (root_dir, output_dir, seeds, algorithms) for a model."""
    return MODELS[model]
