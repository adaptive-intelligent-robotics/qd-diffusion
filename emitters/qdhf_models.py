import time
import numpy as np
import torch
from torch import nn
from repertoires.custom_grid import GridDictArchive

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
    config,
    seed,
):
    _, objs, _, clip_features, images = scoring_function(solutions)
    with torch.no_grad():
        measures = diversity_model(clip_features).detach().cpu().numpy()

    archive_bounds = np.array(
        [np.quantile(measures, 0.01, axis=0), np.quantile(measures, 0.99, axis=0)]
    ).T

    sols = solutions.detach().cpu().numpy()
    sols = sols.reshape(sols.shape[0], -1)

    # Set up archive.
    archive = GridDictArchive(
        solution_dim=len(sols[0]),
        dims=config.algorithm.latent_dims,
        ranges=archive_bounds,
        seed=seed,
    )

    # Add initial solutions to the archive.
    sols["image"] = images.cpu().numpy()
    archive.add(sols, objs, measures)
    
    return archive
    
def update_qdhf_diversity_model(archive, init_solution, scoring_function, config):
    """
    Projects all history into new space, calculates bounds, 
    creates a new Grid Archive
    """
    
    if archive is None:
        all_sols = []
        all_clip_features = []
        all_dreamsim_features = []
        all_objs = []
        
                
        _, qdhf_fitness, dreamsim_feature, clip_feature, _ = scoring_function(init_solution)
        
        all_sols.append(init_solution['latent'].cpu().numpy())
        all_clip_features.append(clip_feature)
        all_dreamsim_features.append(dreamsim_feature)
        all_objs.append(qdhf_fitness)
        
        # Initialize the diversity projection model.
        div_proj_data = []
        div_proj_labels = []
        for _ in range(config.n_pref_data):
            idx = np.random.choice(all_sols.shape[0], 3)
            div_proj_data.append(all_clip_features[idx])
            div_proj_labels.append(all_dreamsim_features[idx])
            div_proj_data = torch.concat(div_proj_data, dim=0)
            div_proj_labels = torch.concat(div_proj_labels, dim=0)
            div_proj_data = div_proj_data.reshape(config.n_pref_data, 3, -1)
            div_proj_label = div_proj_labels.reshape(config.n_pref_data, 3, -1)
            diversity_model, _div_proj_acc = fit_div_proj(
                div_proj_data,
                div_proj_label,
                latent_dim=config.latent_dims,
            )
    
    else:
        # Get all the current solutions and collect feedback.
        all_sols = archive.data("solution").detach().cpu().numpy().astype(np.float32)
        all_sols.reshape(sols.shape[0], -1)
        n_batches = np.ceil(len(all_sols) / config.algorithm.minibatch_size).astype(int)
        all_clip_features = []
        all_dreamsim_features = []
        all_objs = []
        for i in range(n_batches):
            sols = all_sols[i * config.algorithm.minibatch_size : (i + 1) * config.algorithm.minibatch_size]
            _, qdhf_fitness, dreamsim_features, clip_features, _ = scoring_function(init_solution)
            all_sols
            all_clip_features.append(clip_features)
            all_dreamsim_features.append(dreamsim_features)
            all_objs.append(qdhf_fitness)
        all_clip_features = torch.concat(
            all_clip_features, dim=0
        )  # n_pref_data * 3, dim
        all_dreamsim_features = torch.concat(all_dreamsim_features, dim=0)
        all_objs = np.concatenate(all_objs, axis=0)

        # Update the diversity projection model.
        additional_features = []
        additional_labels = []
        for _ in range(config.algorithm.n_pref_data):
            idx = np.random.choice(all_sols.shape[0], 3)
            additional_features.append(all_clip_features[idx])
            additional_labels.append(all_dreamsim_features[idx])
        additional_features = torch.concat(additional_features, dim=0)
        additional_labels = torch.concat(additional_labels, dim=0)
        additional_div_proj_data = additional_features.reshape(config.algorithm.n_pref_data, 3, -1)
        additional_div_proj_label = additional_labels.reshape(config.algorithm.n_pref_data, 3, -1)
        div_proj_data = torch.concat(
            (div_proj_data, additional_div_proj_data), axis=0
        )
        div_proj_label = torch.concat(
            (div_proj_label, additional_div_proj_label), axis=0
        )
        diversity_model, _div_proj_acc = fit_div_proj(
            div_proj_data,
            div_proj_label,
            latent_dim=2,
        )
        
    return diversity_model