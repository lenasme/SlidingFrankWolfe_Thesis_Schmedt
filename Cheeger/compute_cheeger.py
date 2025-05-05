import numpy as np
#import cvxpy as cp
import shutil
import pickle

import copy
from scipy.optimize import minimize
import matplotlib.pyplot as plt




from .rectangular_set import RectangularSet, construct_rectangular_set_from01
from Setup.ground_truth import  construction_of_example_source
from .tools import run_primal_dual, extract_contour, write_latex_command
from .plot_utils import plot_primal_dual_results
from .fine_optimization_rectangle import run_fine_optimization

from SlidingFrankWolfe.simple_function import IndicatorFunction, SimpleFunction 



def calculate_target_function(grid_size, deltas, max_jumps, cut_off, variance, seed= None,  plot = True):
	
	
	ground_truth = construction_of_example_source(grid_size, deltas, max_jumps, seed = seed)

	operator_applied_on_ground_truth = np.fft.fft2(ground_truth)

	freqs_x= np.fft.fftfreq(grid_size, d=1 / grid_size)
	freqs_y = np.fft.fftfreq(grid_size, d=1 / grid_size)
	freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")
	

	
	mask = np.zeros((grid_size, grid_size))
	mask[(np.abs(freq_x) <= cut_off) & (np.abs(freq_y) <= cut_off)] = 1

	truncated_operator_applied_on_ground_truth = operator_applied_on_ground_truth * mask

	rng = np.random.default_rng(seed)
	noise =  (rng.standard_normal((grid_size, grid_size)) + 1j * rng.standard_normal((grid_size, grid_size)))
	current_norm = np.linalg.norm(noise)
	target_norm = np.sqrt(2 * variance)
	noise *= target_norm / current_norm


	target_function_f = truncated_operator_applied_on_ground_truth + noise

	if plot == True:
		

		plt.plot()
		plt.imshow(ground_truth, cmap = 'bwr')
		plt.colorbar()
		plt.title("Ground Truth")
		plt.show()

		plt.plot()
		plt.imshow(truncated_operator_applied_on_ground_truth.real, cmap= 'bwr')
		plt.colorbar()
		plt.title("Truncated Fourier Frequency Image")
		plt.show()

		
		plt.plot()
		plt.imshow(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, cmap = 'bwr')
		plt.colorbar()
		plt.title("Reconstruction of Truncated Fouried Applied on Ground Truth")
		plt.show()


	return truncated_operator_applied_on_ground_truth, ground_truth, target_function_f


def compute_cheeger_set(truncated_operator_applied_on_ground_truth, grid_size, grid_size_coarse, cut_off, max_iter_primal_dual = 10000, plot=True):
   
	h = grid_size / grid_size_coarse
	eta_bar = np.zeros((grid_size_coarse, grid_size_coarse))
	for i in range(grid_size_coarse):
		for j in range(grid_size_coarse):
			x_min = 0 + i * h
			x_max = (i+1) * h
			y_min = j * h
			y_max = (j+1) * h
			rectangle_coarse_grid = RectangularSet(x_min, x_max, y_min, y_max)

			eta_bar[i,j] = (rectangle_coarse_grid.compute_integral(cut_off, truncated_operator_applied_on_ground_truth, grid_size) / h**2).real


	u = run_primal_dual(grid_size_coarse, eta_bar, max_iter=max_iter_primal_dual, convergence_tol=None, plot=True)

	if plot == True:
		plot_primal_dual_results(u[1:-1, 1:-1], eta_bar)

	boundary_vertices = extract_contour(u)
	

	
	initial_rectangular_set = construct_rectangular_set_from01(boundary_vertices, grid_size)
	initial_coordinates = initial_rectangular_set.coordinates
	if plot == True:
		print("Rectangle created by outer boundary vertices of Primal-Dual result:")
		initial_rectangular_set.plot_rectangular_set(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, grid_size)

	x_min, x_max, y_min, y_max = initial_rectangular_set.coordinates[0], initial_rectangular_set.coordinates[1], initial_rectangular_set.coordinates[2], initial_rectangular_set.coordinates[3]

	
	weights = truncated_operator_applied_on_ground_truth
	
	optimal_rectangle,  objective_tab, gradient_tab , x_mins, x_maxs, y_mins, y_maxs =  run_fine_optimization(initial_rectangular_set, cut_off, weights, grid_size )

	if objective_tab[-1] < 0:
		return initial_rectangular_set

	if plot == True:
		print("Optimal Rectangle initialized by outer boundary vertices of Primal-Dual result:")
		optimal_rectangle.plot_rectangular_set(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, grid_size)
		print(f"initiale Koordinaten: {initial_coordinates}")
		print(f"optimale Koordinaten: {optimal_rectangle.coordinates}")
		print(f"Verschiebung: {optimal_rectangle.coordinates - initial_coordinates}")

	return optimal_rectangle


#def fourier_image_rectangle(rectangular_set, grid_size, cut_off):
	new_indicator_function = IndicatorFunction(rectangular_set, grid_size)
	

	fourier_image = new_indicator_function.compute_truncated_frequency_image(cut_off, plot = False)

	return fourier_image



def fit_weights(u, target_function_f, reg_param, plot=False):
	print("Vorheriges Objective:", u.compute_objective_sliding(target_function_f, reg_param))
	def objective_scipy(a_vec):
		# Update objective for scipy
		for i in range(u.num_atoms):
			u.atoms[i].weight = a_vec[i]
		return u.compute_objective_sliding(target_function_f, reg_param)
	
	def gradient_scipy(a_vec):
		# Update gradient for scipy
		for i in range(u.num_atoms):
			u.atoms[i].weight = a_vec[i]
		full_gradient = u.compute_gradient_sliding(target_function_f, reg_param) 
		return full_gradient[:,0]

	a_init = np.array([atom.weight for atom in u.atoms])
	bounds =[(-1, 1)] * u.num_atoms
	res = minimize(objective_scipy, a_init, jac=gradient_scipy, bounds = bounds, method='L-BFGS-B')
	

	if res.success:
		for i in range(u.num_atoms):
			u.atoms[i].weight = res.x[i]

	if plot == True:
		print("Current objective:", u.compute_objective_sliding(target_function_f, reg_param))

	u.remove_small_atoms(threshold = 0.015)



def sliding_step(u,  target_function_f, reg_param):
	a = np.zeros(u.num_atoms)
	x_mins = np.zeros(u.num_atoms)
	x_maxs = np.zeros(u.num_atoms)
	y_mins = np.zeros(u.num_atoms)
	y_maxs = np.zeros(u.num_atoms)

	for i in range(u.num_atoms):
		a[i] = u.atoms[i].weight
		x_mins[i] = u.atoms[i].support.coordinates[0]
		x_maxs[i] = u.atoms[i].support.coordinates[1]
		y_mins[i] = u.atoms[i].support.coordinates[2]
		y_maxs[i] = u.atoms[i].support.coordinates[3]

	initial_parameters = np.concatenate((a, x_mins, x_maxs, y_mins, y_maxs))
	bounds =[(-1, 1)] * u.num_atoms + [(0, u.grid_size)] * u.num_atoms + [(0, u.grid_size)] * u.num_atoms + [(0, u.grid_size)] * u.num_atoms + [(0, u.grid_size)] * u.num_atoms

	objective_development = []
	gradient_development = []
	x_min_values = []
	x_max_values = []
	y_min_values = []
	y_max_values = []
	

	def callback(params):
		objective_value = u.objective_wrapper_sliding(params, target_function_f, reg_param)
		gradient_value = u.gradient_wrapper_sliding(params, target_function_f, reg_param)

		x_min_value = u.atoms[0].support.x_min
		x_max_value = u.atoms[0].support.x_max
		y_min_value = u.atoms[0].support.y_min
		y_max_value = u.atoms[0].support.y_max

		gradient_norm = np.linalg.norm(gradient_value)
	
		objective_development.append(objective_value)
		gradient_development.append(gradient_norm)
		x_min_values.append(x_min_value)
		x_max_values.append(x_max_value)
		y_min_values.append(y_min_value)
		y_max_values.append(y_max_value)
		


	result = minimize( fun = u.objective_wrapper_sliding, x0 = initial_parameters, args =(target_function_f, reg_param),jac = u.gradient_wrapper_sliding, bounds =bounds, method='L-BFGS-B', options={'gtol': 1e-3,'maxiter': 100, 'disp': True }, callback = callback)


	if not result.success:
		print("Optimization did not converge:", result.message)

	new_parameters = result.x

	new_weights = new_parameters[:u.num_atoms]
	new_x_mins = new_parameters[u.num_atoms:2*u.num_atoms]
	new_x_maxs = new_parameters[2*u.num_atoms:3*u.num_atoms]
	new_y_mins = new_parameters[3*u.num_atoms:4*u.num_atoms]
	new_y_maxs = new_parameters[4*u.num_atoms:5*u.num_atoms]

	for i in range(u.num_atoms):
		u.atoms[i].weight = new_weights[i]
		u.atoms[i].support.coordinates = (new_x_mins[i], new_x_maxs[i], new_y_mins[i], new_y_maxs[i])

	u.remove_small_atoms(threshold = 1e-3)
	return u, objective_development, gradient_development, x_min_values, x_max_values, y_min_values, y_max_values




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

		weights_in_eta = - u.compute_truncated_frequency_image_sf( plot = True) + target_function_f

		optimal_rectangle = compute_cheeger_set(weights_in_eta, grid_size, grid_size_coarse, cut_off, max_iter_primal_dual = 10000, plot=True)

		
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



	vmin = min(np.min(ground_truth), -1) 
	vmax = max(np.max(ground_truth), 1)
	vmax_diff = np.max(np.abs(ground_truth))
	data = u.construct_image_matrix_sf(plot=False)
	diff = -data + ground_truth
			
			

	# save reconstructed image
	plt.figure()
	plt.imshow(data, cmap='bwr', vmin=vmin, vmax=vmax)
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(fr"C:\Lena\Universität\Inhaltlich\Master\AMasterarbeit\Masterarbeit_Dokument\fw_reconstruction_cutoff{cut_off}_seed{seed}.png", dpi=300)
	plt.close()

	# save image of differences
	plt.figure()
	plt.imshow(diff, cmap='bwr', vmin=-vmax_diff, vmax=vmax_diff)
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(fr"C:\Lena\Universität\Inhaltlich\Master\AMasterarbeit\Masterarbeit_Dokument\fw_difference_cutoff{cut_off}_seed{seed}.png", dpi=300)
	plt.close()


	with open(f"simplefunction_fw_u_cutoff_{cut_off}_seed{seed}.pkl", "wb") as f:
		pickle.dump(u, f)

	with open(f"objective_fw_cutoff{cut_off}_seed{seed}.pkl", "wb") as f:
		pickle.dump(objective_whole_iteration, f)




def optimization_with_sliding ( ground_truth, target_function_f, grid_size, grid_size_coarse, cut_off, reg_param, seed, max_iter_primal_dual = 10000, plot=True):
	
	l1_errors = []
	atoms = []
	u = SimpleFunction(atoms, grid_size, cut_off)

	objective_whole_iteration = []
	
	iteration = 0
	max_iter = 21

	convergence = False

	objective_whole_iteration.append(u.compute_objective_sliding( target_function_f, reg_param))

	while not convergence and iteration < max_iter:   
	
		weights_in_eta = - u.compute_truncated_frequency_image_sf( plot = True) + target_function_f

		optimal_rectangle = compute_cheeger_set(weights_in_eta, grid_size, grid_size_coarse, cut_off, max_iter_primal_dual = 10000, plot=True)

		u.extend_support(optimal_rectangle)
		
		
		fit_weights(u,  target_function_f, reg_param)


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

		
		
		v = copy.deepcopy(u)


		u, objective_development, gradient_development, x_min_values, x_max_values, y_min_values, y_max_values = sliding_step(u, target_function_f, reg_param)


		if plot == True:

			plt.figure()
			plt.plot(objective_development)
			plt.title("Objective")
			plt.show()

			plt.figure()
			plt.plot(gradient_development)
			plt.title("Gradient")
			plt.show()

			

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

			
			

			plt.plot()
			data = v.construct_image_matrix_sf(plot=False)  - u.construct_image_matrix_sf(plot=False) 
			plt.imshow(data, cmap = 'bwr')
			plt.colorbar()
			plt.title("Changes by Sliding Step")
			plt.show()

		fit_weights(u, target_function_f, reg_param)


		objective_whole_iteration.append(u.compute_objective_sliding( target_function_f, reg_param))
		

		plt.figure()
		plt.plot(objective_whole_iteration)
		plt.title("Objective development each iteration")
		plt.show()

		data = u.construct_image_matrix_sf(plot=False)
		
		l1_error = np.sum(np.abs(-data + ground_truth))
		l1_error_normalized = l1_error / (grid_size * grid_size)

		l1_errors.append(l1_error_normalized)

		convergence = ((objective_whole_iteration[-2] - objective_whole_iteration[-1] ) < 10 )


		iteration += 1

	plt.figure()
	plt.plot(l1_errors)
	plt.title("L1 Error per Iteration")
	plt.xlabel("Iteration")
	plt.ylabel("L1 Error")
	plt.show()

	with open(f"sfw_l1_errors_cutoff{cut_off}_seed{seed}.pkl", "wb") as f:
		pickle.dump(l1_errors, f)

	
	print("number of rectangles", u.num_atoms)
	number_of_atoms = u.num_atoms
	np.save(f"sfw_number_of_rectangels_iteration20_cutoff{cut_off}_seed{seed}.npy", number_of_atoms)


	vmin = min(np.min(ground_truth), -1)  
	vmax = max(np.max(ground_truth), 1)
	vmax_diff = np.max(np.abs(ground_truth))
	data = u.construct_image_matrix_sf(plot=False)
	diff = -data + ground_truth

	# save reconstructed image
	plt.figure()
	plt.imshow(data, cmap='bwr', vmin=vmin, vmax=vmax)
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(fr"C:\Lena\Universität\Inhaltlich\Master\AMasterarbeit\Masterarbeit_Dokument\sfw_reconstruction_cutoff{cut_off}_seed{seed}.png", dpi=300)
	plt.close()

	# save image of differences
	plt.figure()
	plt.imshow(diff, cmap='bwr', vmin=-vmax_diff, vmax=vmax_diff)
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(fr"C:\Lena\Universität\Inhaltlich\Master\AMasterarbeit\Masterarbeit_Dokument\sfw_difference_cutoff{cut_off}_seed{seed}.png", dpi=300)
	plt.close()


	with open(f"simplefunction_sfw_u_cutoff_{cut_off}_seed{seed}.pkl", "wb") as f:
		pickle.dump(u, f)

	with open(f"objective_sfw_cutoff{cut_off}_seed{seed}.pkl", "wb") as f:
		pickle.dump(objective_whole_iteration, f)