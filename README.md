# Latent Exploration of Diffusion Models with Quality-Diversity

Code for generating *diverse* sets of images from a single text prompt by running
Quality-Diversity (QD) search over the internal latent space of a text-to-image diffusion model.

Instead of only varying the initial noise seed, the search perturbs the initial latent, the prompt
embeddings and (optionally) intermediate hidden-layer activations of the model. Perturbations along
SVD-derived directions of those hidden layers are the main method (labelled "Spectral-Guided
QD-Diffusion" in the analysis code). Images are scored for **fitness** by CLIP similarity to the
prompt, and compared with each other using a perceptual **descriptor** (DreamSim or DINOv2).

Supported diffusion models: **SDXL-Turbo**, **FLUX**, **Stable Diffusion 1.5**.

## Repository layout

| Path | Contents |
| --- | --- |
| [main.py](main.py) | Entry point. Hydra app that builds the model, archive, emitters and runs the QD loop. |
| [config/](config/) | Hydra configs, one group each for `model`, `algorithm` and `emitter`, plus [config.yaml](config/config.yaml) with the global defaults. |
| [emitters/](emitters/) | Variation operators: Gaussian, iso-line, Sep-CMA-ES, Adam, hidden-layer (SVD / Gaussian) for SD-1.5, SDXL and FLUX, ImageBreeder, random seeds, plus CADS, Particle Guidance and Diverse Diffusion baselines. |
| [repertoires/](repertoires/) | Archives: DNS, CVT-MAP-Elites, a passive population, and a custom grid. |
| [scoring_function.py](scoring_function.py) | Image generation, CLIP fitness and descriptor computation. |
| [descriptors.py](descriptors.py) | DreamSim and DINOv2 feature extractors. |
| [metrics.py](metrics.py) | QD metrics (QD score, coverage, diversity, top-k statistics) and CSV logging. |
| [model_utils.py](model_utils.py), [image_utils.py](image_utils.py) | Model loading/hooks, saving and plotting of repertoires. |
| [prompts.py](prompts.py) | The 36 benchmark prompts, selected with `prompt_key`. |
| [analysis_code/](analysis_code/) | Post-hoc analysis scripts (tables, significance tests, FID, plots). |
| [apptainer/](apptainer/) | Container definition and an HPC job spec. |
| [run_multiple.sh](run_multiple.sh) | Example of sweeping prompts/seeds. |

## Installation

Python 3.10 and a CUDA GPU are expected. The pinned stack is JAX 0.5.1 (CUDA 12) with
PyTorch 2.5.1 (cu121).

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Replace the CPU/default torch with the CUDA build
pip install --force-reinstall \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    xformers==0.0.28.post3 --extra-index-url https://download.pytorch.org/whl/cu121
pip install pytorch-fid torchmetrics
```

The container recipe in [apptainer/container.def](apptainer/container.def) does the same thing. It
needs `gitlab_user`, `gitlab_token` and `commit` template variables to clone the repository.

The container definition also adds `QDax` and `APDDv2` to `PYTHONPATH` (expected under
`/workspace/src/`), so make sure these are available if you build from it.

Pretrained weights (Hugging Face models, CLIP, DreamSim, DINO) are downloaded on first use. HF caches
are written to `/var/tmp/hf`. Model weights, containers, `coco/` and `results/` are git-ignored.

## Running an experiment

```bash
python main.py model=sdxl-turbo emitter=multi-hidden-layer-svd algorithm=dns prompt_key=5 seed=42
```

Everything is a [Hydra](https://hydra.cc) override, so any value in [config/config.yaml](config/config.yaml)
can be changed on the command line.

| Option | Choices / meaning |
| --- | --- |
| `model` | `sdxl-turbo` (default), `flux`, `sd-1-5` |
| `algorithm` | `dns` (default), `cvt-map-elites`, `cma-mae`, `qdhf` |
| `emitter` | any file in [config/emitter/](config/emitter/), e.g. `multi-hidden-layer-svd` (default), `multi-gaussian`, `multi-isoline`, `multi-sep-cma-es`, `multi-adam`, `random-seeds`, `cads-flux`, `particle-guidance-flux`, `diverse-diffusion`, `mixed-all` |
| `prompt_key` | Index into [prompts.py](prompts.py) (1-36) |
| `prompt` | Free-text prompt; if non-empty it overrides `prompt_key` |
| `descriptor` | `dreamsim` (default) or `dino-v2` |
| `max_pop_size`, `batch_size`, `max_iterations` | Population size (64), offspring per iteration (16) and iterations (125) |
| `fitness_threshold` | Minimum CLIP fitness for an image to be kept (`-1` disables) |
| `top_k` | `k` used for the top-k diversity metrics |
| `save_images`, `save_offspring_data`, `save_repertoire_period` | What to write to disk, and how often |
| `wandb_mode` | `online`, `offline` or `disabled` |

Emitters are tied to a model (for example the `flux-*` and `sdxl-*` emitter configs), and `main.py`
asserts when `emitter.model_type` does not match `model`. Emitter-specific parameters can be
overridden in the same way, e.g. `emitter.svd_k=10`.

FLUX example:

```bash
python main.py model=flux emitter=flux-multi-svd prompt_key=4 seed=4 max_iterations=100 batch_size=4
```

[run_multiple.sh](run_multiple.sh) shows how to sweep several prompt/seed pairs, and
[apptainer/hpc.yaml](apptainer/hpc.yaml) lists the arguments used for the HPC runs.

### Outputs

Each run writes to a Hydra directory under `output/`:

```
output/<model>/<prompt_key>/<algorithm>/<emitter>/<timestamp>_<seed>/
```

It contains the resolved config, the metrics CSV, and the saved repertoires/images. Metrics are also
logged to Weights & Biases unless `wandb_mode=disabled`.

## Analysis

The scripts in [analysis_code/](analysis_code/) read results from `results/main_results/` (laid out as
`<model>/<seed>/<algorithm path>/<timestamp>/`) and write to `analysis_output/` (SDXL-Turbo) or
`analysis_output_flux/` (FLUX). Which algorithms and seeds are included, and where each one's
results live, is set in [analysis_code/config.py](analysis_code/config.py).

Every script takes `--model {sdxl-turbo,flux}`; run them from the repo root or from `analysis_code/`:

```bash
cd analysis_code
python generate_latex_table.py --model sdxl-turbo   # results table + p_values.csv
python calculate_p_values.py   --model sdxl-turbo   # Wilcoxon tests with Holm correction
python analyse_diversity.py    --model sdxl-turbo   # diversity statistics (CLIP / DINOv2 / DreamSim)
python calculate_fid.py        --model sdxl-turbo   # FID against COCO val2017 (expects coco/val2017)
python plot_pareto_front.py    --model sdxl-turbo   # fitness vs. diversity Pareto plot
python generate_top_k_plot.py  --model sdxl-turbo --top-k 16 --metric fitness   # image grids
python emitter_timings.py      --model sdxl-turbo   # per-emitter timing summary
```

Algorithms with missing results for a given model or seed are skipped.

## Reproducibility

`main.py` enables deterministic CUDA/cuDNN settings and disables TF32. Runs are seeded with `seed`.
Some operations without deterministic implementations only emit a warning, so small numerical
differences between GPUs are still possible.
