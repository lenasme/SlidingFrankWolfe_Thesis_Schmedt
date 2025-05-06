import numpy as np
import matplotlib.pyplot as plt
import pickle
from Optimization_Functionals.simple_function import SimpleFunction
from Optimization_Utils.methods import compute_cheeger_set, fit_weights


def standard_optimization( ground_truth, target_function_f, grid_size, grid_size_coarse, cut_off, reg_param, seed, max_iter_primal_dual = 10000, plot=True):
	
	l1_errors = []


	atoms = []
	u = SimpleFunction(atoms, grid_size, cut_off)

	objective_whole_iteration = []
	
	iteration = 0
	max_iter = 21

	convergence = False

	objective_whole_iteration.append(u.compute_objective_sliding( target_function_f, reg_param))
	

	while not convergence and iteration < max_iter:   

		weights_in_eta = - u.compute_truncated_frequency_image_sf( plot = False) + target_function_f

		optimal_rectangle = compute_cheeger_set(weights_in_eta, grid_size, grid_size_coarse, cut_off, max_iter_primal_dual = 10000, plot=False)

		
		u.extend_support(optimal_rectangle)

		
		
		fit_weights(u,  target_function_f, reg_param)

		weights = []
		for atom in u.atoms:
			weights.append(atom.weight)

		objective_whole_iteration.append(u.compute_objective_sliding( target_function_f, reg_param))


		if plot == True:

			fig, ax = plt.subplots(1, 3, figsize=(18, 6))  

			data = u.construct_image_matrix_sf(plot=False) 
			vmin = min(np.min(data), np.min(ground_truth))
			vmax = max(np.max(data), np.max(ground_truth))

			im1 = ax[0].imshow(data, cmap="bwr", vmin=vmin,
							   vmax=vmax)  
			fig.colorbar(im1, ax=ax[0])
			ax[0].set_title("Current Function")

			im2 = ax[1].imshow(ground_truth, cmap = 'bwr', vmin=vmin, vmax=vmax)

			fig.colorbar(im2, ax = ax[1])
			ax[1].set_title("Ground Truth")

			diff = - data + ground_truth
			vmax_diff = np.max(np.abs(diff))
			im3 = ax[2].imshow(diff, cmap = 'bwr', vmin=-vmax_diff, vmax=vmax_diff)
			fig.colorbar(im3, ax = ax[2])
			ax[2].set_title("Difference")


			plt.tight_layout()

			plt.show()
		
		

		plt.figure()
		plt.plot(objective_whole_iteration)
		plt.title("Overall Objective Development")
		plt.show()

		convergence = ((objective_whole_iteration[-2] - objective_whole_iteration[-1] ) < 10 )

		data = u.construct_image_matrix_sf(plot=False)
		
		l1_error = np.sum(np.abs(-data + ground_truth))
		l1_error_normalized = l1_error / (grid_size * grid_size)

		
		l1_errors.append(l1_error_normalized)


		iteration += 1

	plt.figure()
	plt.plot(l1_errors)
	plt.title("L1 Error per Iteration")
	plt.xlabel("Iteration")
	plt.ylabel("L1 Error")
	plt.show()

	with open(f"fw_l1_errors_cutoff{cut_off}_seed{seed}.pkl", "wb") as f:
		pickle.dump(l1_errors, f)

	

	print("number of rectangles", u.num_atoms)
	number_of_atoms = u.num_atoms
	np.save(f"fw_number_of_rectangels_iteration20_cutoff{cut_off}_seed{seed}.npy", number_of_atoms)



	with open(f"simplefunction_fw_u_cutoff_{cut_off}_seed{seed}.pkl", "wb") as f:
		pickle.dump(u, f)

	with open(f"objective_fw_cutoff{cut_off}_seed{seed}.pkl", "wb") as f:
		pickle.dump(objective_whole_iteration, f)
