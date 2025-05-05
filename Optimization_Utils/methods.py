import numpy as np
from scipy.optimize import minimize

from .coarse_optimization import run_primal_dual, extract_contour, plot_primal_dual_results
from .fine_optimization import run_fine_optimization
from Optimization_Functionals.rectangular_set import RectangularSet, construct_rectangular_set_from01




def compute_cheeger_set(truncated_operator_applied_on_ground_truth, grid_size, grid_size_coarse, cut_off, max_iter_primal_dual = 10000, plot=False):
   
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

	
	print("Optimal Rectangle initialized by outer boundary vertices of Primal-Dual result:")
	optimal_rectangle.plot_rectangular_set(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, grid_size)
	if plot == True:
		print(f"initiale Koordinaten: {initial_coordinates}")
		print(f"optimale Koordinaten: {optimal_rectangle.coordinates}")
		print(f"Verschiebung: {optimal_rectangle.coordinates - initial_coordinates}")

	return optimal_rectangle



def fit_weights(u, target_function_f, reg_param, plot=False):
	
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
		


	result = minimize( fun = u.objective_wrapper_sliding, x0 = initial_parameters, args =(target_function_f, reg_param),jac = u.gradient_wrapper_sliding, bounds =bounds, method='L-BFGS-B', options={'gtol': 1e-3,'maxiter': 150, 'disp': True }, callback = callback)


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

	u.remove_small_atoms(threshold = 0.015)
	return u, objective_development, gradient_development, x_min_values, x_max_values, y_min_values, y_max_values