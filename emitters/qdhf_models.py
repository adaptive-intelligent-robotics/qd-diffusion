import time
import logging
import numpy as np
import torch
from torch import nn
from repertoires.custom_grid import GridDictArchive

logger = logging.getLogger(__name__)

# Define device inside the function or pass it
def get_div_proj_model(input_dim, latent_dim=2, device="cuda"):
    model = DivProj(input_dim, latent_dim).to(device)
    return model

class DivProj(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=latent_dim),
        )

    def forward(self, x):
        """Get diversity representations."""
        x = self.proj(x)
        return x

    def calc_dis(self, x1, x2):
        """Calculate diversity distance as (squared) L2 distance."""
        x1 = self.forward(x1)
        x2 = self.forward(x2)
        return torch.sum(torch.square(x1 - x2), -1)

    def triplet_delta_dis(self, ref, x1, x2):
        """Calculate delta distance comparing x1 and x2 to ref."""
        x1 = self.forward(x1)
        x2 = self.forward(x2)
        ref = self.forward(ref)
        return torch.sum(torch.square(ref - x1), -1) - torch.sum(
            torch.square(ref - x2), -1
        )
        
    
# Loss function from tutorial
# Triplet loss with margin 0.05.
# The binary preference labels are scaled to y = 1 or -1 for the loss, where y = 1 means x2 is more similar to ref than x1.
loss_fn = lambda y, delta_dis: torch.max(
    torch.tensor([0.0]).to(delta_dis.device), 0.05 - (y * 2 - 1) * delta_dis
).mean()


def fit_div_proj(inputs, dreamsim_features, latent_dim, batch_size=32):
    """Trains the DivProj model on ground-truth labels."""
    t = time.time()
    model = DivProj(input_dim=inputs.shape[-1], latent_dim=latent_dim)
    model.to(inputs.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_pref_data = inputs.shape[0]
    ref = inputs[:, 0]
    x1 = inputs[:, 1]
    x2 = inputs[:, 2]

    n_train = int(n_pref_data * 0.75)
    n_val = n_pref_data - n_train

    # Split data into train and val.
    ref_train = ref[:n_train]
    x1_train = x1[:n_train]
    x2_train = x2[:n_train]
    ref_val = ref[n_train:]
    x1_val = x1[n_train:]
    x2_val = x2[n_train:]

    # Split DreamSim features into train and val.
    ref_dreamsim_features = dreamsim_features[:, 0]
    x1_dreamsim_features = dreamsim_features[:, 1]
    x2_dreamsim_features = dreamsim_features[:, 2]
    ref_gt_train = ref_dreamsim_features[:n_train]
    x1_gt_train = x1_dreamsim_features[:n_train]
    x2_gt_train = x2_dreamsim_features[:n_train]
    ref_gt_val = ref_dreamsim_features[n_train:]
    x1_gt_val = x1_dreamsim_features[n_train:]
    x2_gt_val = x2_dreamsim_features[n_train:]

    val_acc = []
    n_iters_per_epoch = max((n_train) // batch_size, 1)
    for epoch in range(200):
        for _ in range(n_iters_per_epoch):
            optimizer.zero_grad()

            idx = np.random.choice(n_train, batch_size)
            batch_ref = ref_train[idx].float()
            batch1 = x1_train[idx].float()
            batch2 = x2_train[idx].float()

            # Get delta distance from model.
            delta_dis = model.triplet_delta_dis(batch_ref, batch1, batch2)

            # Get preference labels from DreamSim features.
            gt_dis = torch.nn.functional.cosine_similarity(
                ref_gt_train[idx], x2_gt_train[idx], dim=-1
            ) - torch.nn.functional.cosine_similarity(
                ref_gt_train[idx], x1_gt_train[idx], dim=-1
            )
            gt = (gt_dis > 0).to(inputs.dtype)

            loss = loss_fn(gt, delta_dis)
            loss.backward()
            optimizer.step()

        # Validate.
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            idx = np.arange(n_val)
            batch_ref = ref_val[idx].float()
            batch1 = x1_val[idx].float()
            batch2 = x2_val[idx].float()
            delta_dis = model.triplet_delta_dis(batch_ref, batch1, batch2)
            pred = delta_dis > 0
            gt_dis = torch.nn.functional.cosine_similarity(
                ref_gt_val[idx], x2_gt_val[idx], dim=-1
            ) - torch.nn.functional.cosine_similarity(
                ref_gt_val[idx], x1_gt_val[idx], dim=-1
            )
            gt = gt_dis > 0
            n_correct += (pred == gt).sum().item()
            n_total += len(idx)

        acc = n_correct / n_total
        val_acc.append(acc)

        # Early stopping if val_acc does not improve for 10 epochs.
        if epoch > 10 and np.mean(val_acc[-10:]) < np.mean(val_acc[-11:-1]):
            break

    print(
        f"{np.round(time.time() - t, 1)}s ({epoch + 1} epochs) | DivProj (n={n_pref_data}) fitted with val acc.: {acc}"
    )

    return model.to(inputs.dtype), acc



def get_qdhf_archive(
    solutions,
    scoring_function,
    diversity_model,
    solution_schema,
    config,
    warmup_clip_features=None,
):
    """
    Creates a new GridDictArchive based on the current solutions and diversity model.
    """
    # 1. Evaluate Solutions
    # solutions is a dict: {'latent': (B, ...), 'prompt': (B, ...)}
    dns_fitnesses, qdhf_fitnesses, dreamsim_features, clip_features, images = scoring_function(solutions)
    
    with torch.no_grad():
        measures = diversity_model(clip_features.float()).detach().cpu().numpy()
    
    # 2. Calculate Bounds (with Zero-Division Protection)
    lower_bounds = np.quantile(measures, 0.01, axis=0)
    upper_bounds = np.quantile(measures, 0.99, axis=0)
    
    # if it's first iteration, use the features from the "warmup" solutions
    if warmup_clip_features is not None:
        warmup_measures = diversity_model(warmup_clip_features.float()).detach().cpu().numpy()
        lower_bounds = np.quantile(warmup_measures, 0.01, axis=0)
        upper_bounds = np.quantile(measures, 0.99, axis=0)
    
    archive_bounds = np.array([lower_bounds, upper_bounds]).T
    
    solution_schema["dns_descriptor"] = (768,)
    solution_schema["dns_fitness"] = ()
        
    archive = GridDictArchive(
        solution_dim=solution_schema, # Pass the schema dict
        dims=[int(np.sqrt(config.max_pop_size)), config.max_pop_size // int(np.sqrt(config.max_pop_size))],   # e.g., [20, 20]
        ranges=archive_bounds,
        seed=config.seed,
        dtype=np.float32,
    )

    sols_np = {}
    for k, v in solutions.items():
        if isinstance(v, torch.Tensor):
            sols_np[k] = v.detach().cpu().numpy()
        else:
            sols_np[k] = v

    sols_np["image"] = images.detach().cpu().numpy()
    sols_np["dns_descriptor"] = dreamsim_features
    sols_np["dns_fitness"] = dns_fitnesses
    archive.add(sols_np, qdhf_fitnesses, measures)
    
    return archive
    
def update_qdhf_diversity_model(archive, init_solution, scoring_function, config, return_clip_features=False):
    """
    Projects all history into new space, calculates bounds, 
    creates a new Grid Archive
    """
    
    # Buffers for collecting features before training
    all_clip_features = []
    all_dreamsim_features = []
    
    if archive is None:
        # --- INITIALIZATION PHASE ---
        # We need to generate a "warmup" population by adding noise to the initial solution.
        
        total_warmup = config.algorithm.warmup_pop_size
        minibatch_size = min(config.algorithm.minibatch_size, config.batch_size)
        total_minibatches = int(np.ceil(total_warmup / minibatch_size))
        logger.warning(
            "QDHF warmup: generating %s samples in %s minibatches of size <= %s",
            total_warmup,
            total_minibatches,
            minibatch_size,
        )
        
        # Loop in mini-batches to generate noisy samples without OOM
        for i in range(0, total_warmup, minibatch_size):
            current_batch_size = min(minibatch_size, total_warmup - i)
            batch_idx = i // minibatch_size + 1
            if batch_idx == 1 or batch_idx == total_minibatches or batch_idx % 5 == 0:
                logger.warning(
                    "QDHF warmup minibatch %s/%s (size=%s)",
                    batch_idx,
                    total_minibatches,
                    current_batch_size,
                )
            
            # Create a batch dictionary from the single init_solution
            batch_sols = {}
            for k, v in init_solution.items():
                # Repeat (1, ...) -> (Batch, ...)
                # Assumes v is on GPU/Device already
                repeated = v.repeat(current_batch_size, *([1] * (v.ndim - 1)))
                
                # Add small noise to latents to ensure we get diverse features
                if k == "latent":
                    noise = torch.randn_like(repeated) * config.algorithm.warmup_latent_sigma
                    repeated = repeated + noise
                
                batch_sols[k] = repeated

            # Scoring returns 5 values: fitnesses, qdhf_fitness, dreamsim, clip, images
            _, _, dreamsim_feature, clip_feature, _ = scoring_function(batch_sols)
            
            # Move to CPU to save VRAM while collecting
            all_clip_features.append(clip_feature)
            all_dreamsim_features.append(dreamsim_feature)
    else:
        # --- UPDATE PHASE (FROM ARCHIVE) ---
        # Get all current solutions. 
        # archive.data() returns a dict: {'solution': {'latent': ...}, ...}
        archive_data = archive.data(return_type="dict")
        sols_dict = archive_data["solution"]
        
        # Get total number of solutions (length of first array in dict)
        first_key = next(iter(sols_dict.keys()))
        total_sols = len(sols_dict[first_key])
        indices = np.arange(total_sols)
        
        # Process in mini-batches
        minibatch_size = min(config.algorithm.minibatch_size, config.batch_size)
        n_batches = np.ceil(total_sols / minibatch_size).astype(int)
        logger.warning(
            "QDHF update: rescoring %s archive solutions in %s minibatches of size <= %s",
            total_sols,
            n_batches,
            minibatch_size,
        )
        
        for i in range(n_batches):
            start_idx = i * minibatch_size
            end_idx = min((i + 1) * minibatch_size, total_sols)
            batch_indices = indices[start_idx:end_idx]
            batch_idx = i + 1
            if batch_idx == 1 or batch_idx == n_batches or batch_idx % 5 == 0:
                logger.warning(
                    "QDHF update minibatch %s/%s (size=%s)",
                    batch_idx,
                    n_batches,
                    len(batch_indices),
                )
            
            # Construct batch dict for scoring
            batch_sols = {}
            for k, v in sols_dict.items():
                # Convert numpy -> torch -> device
                # We use init_solution[first_key].device as reference for the correct GPU
                ref_device = init_solution[first_key].device 
                batch_sols[k] = torch.tensor(v[batch_indices]).to(ref_device)

            # Evaluate batch
            _, _, dreamsim_features, clip_features, _ = scoring_function(batch_sols)
            
            all_clip_features.append(clip_features.cpu())
            all_dreamsim_features.append(dreamsim_features)

    # --- TRAIN MODEL ---
    
    # Concatenate all collected features
    # These become the pool we sample triplets from
    all_clip_features = torch.cat(all_clip_features, dim=0)
    all_dreamsim_features = torch.cat([torch.tensor(f) if not isinstance(f, torch.Tensor) else f for f in all_dreamsim_features], dim=0)
    
    # Prepare data for DivProj
    n_samples = all_clip_features.shape[0]
    
    # Safety check for tiny archives
    if n_samples < 3:
        all_clip_features = all_clip_features.repeat(3, 1)
        all_dreamsim_features = all_dreamsim_features.repeat(3, 1)
        n_samples = all_clip_features.shape[0]

    # Sample random triplets
    div_proj_data = []
    div_proj_labels = []
    
    # We sample indices (N, 3) directly instead of appending in a loop (faster/cleaner)
    # Each row is [Ref, X1, X2]
    triplet_indices = np.random.randint(0, n_samples, (config.algorithm.n_pref_data, 3))
    
    # Gather data using advanced indexing
    div_proj_data = all_clip_features[triplet_indices]       # Shape: (n_pref_data, 3, Feat_Dim)
    div_proj_labels = all_dreamsim_features[triplet_indices] # Shape: (n_pref_data, 3, Feat_Dim)

    logger.warning(
        "QDHF fitting diversity model on %s triplets from %s feature samples",
        config.algorithm.n_pref_data,
        n_samples,
    )

    # Fit model
    diversity_model, _div_proj_acc = fit_div_proj(
        div_proj_data,
        div_proj_labels,
        latent_dim=config.algorithm.latent_dims, # Using config value
    )
    
    if return_clip_features:
        return diversity_model, all_clip_features
    else:
        return diversity_model