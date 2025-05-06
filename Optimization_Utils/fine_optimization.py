import numpy as np
from scipy.optimize import minimize

from Optimization_Functionals import RectangularSet


def run_fine_optimization(initial_rectangular_set, cut_off, weights, grid_size ):

	objective_development = []
	gradient_development = []
	x_min_development = []
	x_max_development = []
	y_min_development = []
	y_max_development = []

	def callback(coordinates):
		objective_value = initial_rectangular_set.compute_objective_wrapper(coordinates, cut_off, weights, grid_size)
		gradient_value = initial_rectangular_set.objective_gradient_wrapper(coordinates, cut_off, weights, grid_size)

		gradient_norm = np.sqrt(gradient_value[0]**2 + gradient_value[1]**2 + gradient_value[2]**2 + gradient_value[3]**2)

		objective_development.append(objective_value)
		gradient_development.append(gradient_norm)
		x_min_development.append(coordinates[0])
		x_max_development.append(coordinates[1])
		y_min_development.append(coordinates[2])
		y_max_development.append(coordinates[3])


	
	result = minimize(initial_rectangular_set.compute_objective_wrapper, initial_rectangular_set.coordinates, args=(cut_off, weights, grid_size), jac=initial_rectangular_set.objective_gradient_wrapper , bounds =[(0, grid_size),(0,grid_size), (0, grid_size), (0, grid_size)]  , options={'maxiter': 10000, 'disp': True, 'ftol': 1e-7, 'gtol': 1e-6}, callback=callback)
	

	optimal_coordinates_grad = result.x
	optimal_rectangle_grad = RectangularSet(optimal_coordinates_grad[0], optimal_coordinates_grad[1], optimal_coordinates_grad[2], optimal_coordinates_grad[3])


	return optimal_rectangle_grad,  objective_development, gradient_development, x_min_development, x_max_development, y_min_development, y_max_development
	

