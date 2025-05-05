import numpy as np
import matplotlib.pyplot as plt
import pickle

from Setup.target_function import calculate_target_function
from Setup.ground_truth import GroundTruth




def construct_jump_points(seed, grid_size, deltas, max_jumps, cut_off, u_filepaths, labels, loc, colors=None):
	
	original = GroundTruth(grid_size, max_jumps, seed)

	
	jump_points = original.get_jump_points_bin(deltas)[0]  

	# Visualisierung
	fig, ax = plt.subplots(figsize=(6, 6))
	ax.set_xlim(0, grid_size)
	ax.set_ylim(0, grid_size)
	ax.set_aspect('equal')
	ax.invert_yaxis()  # (0,0) oben links
	ax.axis('off')
	
	for y in jump_points[0]:
		ax.axhline(y, color='black', linestyle='-', linewidth=1.5)

	# Horizontale Linien (Jumps in y)
	for x in jump_points[1]:
		ax.axvline(x, color='black', linestyle='-', linewidth=1.5)

	# Titel, Layout und Anzeige
	if colors is None:
		cmap = plt.get_cmap("tab10")
		colors = [cmap(i) for i in range(len(u_filepaths))]

   
	for filepath, color, label in zip(u_filepaths, colors, labels):
		with open(filepath, "rb") as f:
			u = pickle.load(f)

		proxy = None
		for atom in u.atoms:
			y0, y1 = atom.support.x_min, atom.support.x_max
			x0, x1= atom.support.y_min, atom.support.y_max
			proxy = ax.plot([x0, x1], [y0, y0], color=color, linewidth=1)[0]
			ax.plot([x0, x1], [y0, y0], color=color, linewidth=1)
			ax.plot([x0, x1], [y1, y1], color=color, linewidth=1)
			ax.plot([x0, x0], [y0, y1], color=color, linewidth=1)
			ax.plot([x1, x1], [y0, y1], color=color, linewidth=1)
		if proxy:
			proxy.set_label(label)
	
	ax.legend(loc=loc, fontsize='medium', frameon=True)
	plt.tight_layout()
	plt.savefig(fr"C:\Lena\Universität\Inhaltlich\Master\AMasterarbeit\Masterarbeit_Dokument\ground_truth_jump_points_gradient_support_seed{seed}_cutoff{cut_off}.png", dpi=300)
	plt.show()




def construction_of_two_ground_truths(seed1, seed2, grid_size, deltas, max_jumps, cut_off, variance):
	_, ground_truth1, _ = calculate_target_function(grid_size, deltas, max_jumps, cut_off, variance, seed=seed1, plot=False)
	_, ground_truth2, _ = calculate_target_function(grid_size, deltas, max_jumps, cut_off, variance, seed=seed2, plot=False)

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

def construct_single_ground_truth(seed, grid_size, deltas, max_jumps, cut_off, variance):
	_, ground_truth, _ = calculate_target_function(grid_size, deltas, max_jumps, cut_off, variance, seed=seed, plot=False)

	vmin = min(np.min(ground_truth), -1)  
	vmax = max(np.max(ground_truth), 1)

	plt.figure()
	plt.imshow(ground_truth, cmap='bwr', vmin=vmin, vmax=vmax)
	plt.axis('off')
	plt.tight_layout()
	#plt.savefig(f"reconstruction_iter{iteration}_cutoff{cut_off}.png", dpi=300)
	plt.savefig(fr"C:\Lena\Universität\Inhaltlich\Master\AMasterarbeit\Masterarbeit_Dokument\ground_truth_seed{seed}.png", dpi=300)
	plt.close()
