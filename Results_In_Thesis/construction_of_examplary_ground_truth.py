import numpy as np
import matplotlib.pyplot as plt

from Cheeger.compute_cheeger import calculate_target_function


seed1 = 13
seed2 = 100

grid_size= 100
deltas = [0.08-0.005, 0.09+0.005]
max_jumps = 3 
cut_off = 10


def construction_of_two_ground_truths(seed1, seed2, grid_size, deltas, max_jumps, cut_off):
    _, ground_truth1, _ = calculate_target_function(grid_size, deltas, max_jumps, cut_off, seed=seed1, plot=False)
    _, ground_truth2, _ = calculate_target_function(grid_size, deltas, max_jumps, cut_off, seed=seed2, plot=False)


    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im1 = axes[0].imshow(ground_truth1, cmap='bwr')
    #axes[0].set_title(f"Ground Truth Seed {seed1}")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(ground_truth2, cmap='bwr')
    #axes[1].set_title(f"Ground Truth Seed {seed2}")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    #plt.savefig("ground_truths_two_versions.png", dpi=300)  # Hier speichern
    plt.savefig(r"C:\Lena\Universität\Inhaltlich\Master\AMasterarbeit\Masterarbeit_Dokument\ground_truth_examples.png", dpi=300)
    
    plt.show()