import numpy as np
from model_utils import plot_thumbnails, save_sample_images
from qdax.utils.plotting import plot_map_elites_results, plot_2d_map_elites_repertoire

def analyse_repertoire(
    config,
    iteration,
    metrics,
    repertoire,
    init_prompt_embedding,
    init_latent,
    scheduler,
    unet,
    vae,
    batch_size,
    images_dir,
    repertoire_dir,
    save_name
):

    if save_name == "final":
        # Plot results
        fig, axes = plot_map_elites_results(env_steps=range(iteration),
            metrics=metrics,
            repertoire=repertoire,
            min_descriptor=config.algorithm.map_elites.min_descriptor,
            max_descriptor=config.algorithm.map_elites.max_descriptor
        )
        fig.savefig(images_dir + f"{save_name}-mapelites-results.png")
    
    else:
        fig, axes = plot_2d_map_elites_repertoire(
            centroids=repertoire.centroids,
            repertoire_fitnesses=repertoire.fitnesses,
            minval=config.algorithm.map_elites.min_descriptor,
            maxval=config.algorithm.map_elites.max_descriptor,
            repertoire_descriptors=repertoire.descriptors,
        )
        fig.savefig(images_dir + f"{save_name}-mapelites-repertoire.png")

    
    # Save final repertoire
    np.savez(repertoire_dir + f"{save_name}_repertoire.npz",
        genotypes=repertoire.genotypes,
        descriptors=repertoire.descriptors,
        fitnesses=repertoire.fitnesses,
        centroids=repertoire.centroids
    )
    
    # Save final images
    if config.algorithm.variation.genotype == "embedding":
        save_embedding = None
        save_latents = init_latent
        
    elif config.algorithm.variation.genotype == "latents":
        save_embedding = init_prompt_embedding
        save_latents = None
        
    save_sample_images(repertoire=repertoire,
        num_inference_steps=config.generation.num_inference_steps,
        guidance_scale=config.generation.guidance_scale,
        scheduler=scheduler,
        unet=unet,
        vae=vae,
        n_sample=5,
        folder=images_dir + f"{save_name}/",
        init_embedding=save_embedding,
        init_latent=save_latents,
    )

    plot_thumbnails(
        repertoire=repertoire,
        num_inference_steps=config.generation.num_inference_steps,
        guidance_scale=config.generation.guidance_scale,
        scheduler=scheduler,
        unet=unet,
        vae=vae,
        batch_size=batch_size,
        save_path=images_dir + f"{save_name}/",
        init_embedding=save_embedding,
        init_latent=save_latents,
    )
    
    return




