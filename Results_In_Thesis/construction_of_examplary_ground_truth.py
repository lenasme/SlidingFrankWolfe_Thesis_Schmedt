import numpy as np
import matplotlib.pyplot as plt

from Cheeger.compute_cheeger import calculate_target_function
from Setup.ground_truth import GroundTruth


seed1 = 13
seed2 = 100

grid_size= 100
deltas = [0.08-0.005, 0.09+0.005]
max_jumps = 3 
cut_off = 10

def construct_jump_points(seed, grid_size, deltas, max_jumps, cut_off):
	original = GroundTruth(grid_size, max_jumps, seed)

	# Einen Satz Jump-Points aus einer Delta-Bin erzeugen
	jump_points = original.get_jump_points_bin(deltas)[0]  # [points_x, points_y]

	# Visualisierung
	fig, ax = plt.subplots(figsize=(5, 5))
	ax.set_xlim(0, grid_size)
	ax.set_ylim(0, grid_size)
	ax.set_aspect('equal')
	#ax.invert_yaxis()  # (0,0) oben links

	# Vertikale Linien (Jumps in x)
	for y in jump_points[0]:
		ax.axhline(y, color='red', linestyle='--', linewidth=1)

	# Horizontale Linien (Jumps in y)
	for x in jump_points[1]:
		ax.axvline(y, color='blue', linestyle='--', linewidth=1)

	# Titel, Layout und Anzeige
	ax.set_title(f'Jump Points (seed={seed}, delta-range={deltas})')
	plt.tight_layout()
	plt.show()


def construction_of_two_ground_truths(seed1, seed2, grid_size, deltas, max_jumps, cut_off):
	_, ground_truth1, _ = calculate_target_function(grid_size, deltas, max_jumps, cut_off, seed=seed1, plot=False)
	_, ground_truth2, _ = calculate_target_function(grid_size, deltas, max_jumps, cut_off, seed=seed2, plot=False)

	vmin1 = min(np.min(ground_truth1), -1)  
	vmax1 = max(np.max(ground_truth1), 1)

	vmin2 = min(np.min(ground_truth2), -1)  
	vmax2 = max(np.max(ground_truth2), 1)

	fig, axes = plt.subplots(1, 2, figsize=(10, 5))

	im1 = axes[0].imshow(ground_truth1, cmap='bwr',  vmin=vmin1, vmax=vmax1)
	#axes[0].set_title(f"Ground Truth Seed {seed1}")
	axes[0].axis('off')
	fig.colorbar(im1, ax=axes[0])

	im2 = axes[1].imshow(ground_truth2, cmap='bwr',  vmin=vmin2, vmax=vmax2)
	#axes[1].set_title(f"Ground Truth Seed {seed2}")
	axes[1].axis('off')
	fig.colorbar(im2, ax=axes[1])

	plt.tight_layout()
	#plt.savefig("ground_truths_two_versions.png", dpi=300)  # Hier speichern
	plt.savefig(r"C:\Lena\Universität\Inhaltlich\Master\AMasterarbeit\Masterarbeit_Dokument\ground_truth_examples.png", dpi=300)
	
	plt.show()

def construct_single_ground_truth(seed, grid_size, deltas, max_jumps, cut_off):
	_, ground_truth, _ = calculate_target_function(grid_size, deltas, max_jumps, cut_off, seed=seed1, plot=False)

	vmin = min(np.min(ground_truth), -1)  
	vmax = max(np.max(ground_truth), 1)

	plt.figure()
	plt.imshow(ground_truth, cmap='bwr', vmin=vmin, vmax=vmax)
	plt.axis('off')
	plt.tight_layout()
	#plt.savefig(f"reconstruction_iter{iteration}_cutoff{cut_off}.png", dpi=300)
	plt.savefig(fr"C:\Lena\Universität\Inhaltlich\Master\AMasterarbeit\Masterarbeit_Dokument\ground_truth_seed{seed}.png", dpi=300)
	plt.close()
