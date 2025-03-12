import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.optimize import minimize

#from .tools import resample
#from .plot_utils import plot_rectangular_set

#from Setup.ground_truth import GroundTruth

from .rectangular_set import RectangularSet




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




	start = time.time()
		
	#result = minimize(initial_rectangular_set.compute_objective_wrapper, initial_rectangular_set.coordinates, args=(cut_off, weights, grid_size), jac=initial_rectangular_set.objective_gradient_wrapper , bounds =[(0,1),(0,1), (0,1), (0,1)]  , options={'maxiter': 10000, 'disp': True, 'ftol': 1e-7, 'gtol': 1e-6}, callback=callback)
	result = minimize(initial_rectangular_set.compute_objective_wrapper, initial_rectangular_set.coordinates, args=(cut_off, weights, grid_size) , bounds =[(0,grid_size),(0,grid_size), (0,grid_size), (0,grid_size)]  , options={'maxiter': 10000, 'disp': True, 'ftol': 1e-7, 'gtol': 1e-6}, callback=callback)
	end = time.time()

	optimal_coordinates = result.x
	optimal_rectangle = RectangularSet(optimal_coordinates[0], optimal_coordinates[1], optimal_coordinates[2], optimal_coordinates[3])

	return optimal_rectangle, objective_development, gradient_development



#class CheegerOptimizer:
	def __init__(self, step_size, max_iter, eps_stop, num_points, point_density, max_tri_area, num_iter_resampling,
				 alpha, beta):

		self.step_size = step_size
		self.max_iter = max_iter
		self.eps_stop = eps_stop
		self.num_points = num_points
		self.point_density = point_density
		self.max_tri_area = max_tri_area
		self.num_iter_resampling = num_iter_resampling
		self.alpha = alpha
		self.beta = beta

		self.state = None

	

	#def run_rectangular(self, f, initial_set):
		#convergence = False
		#obj_tab = []
		#grad_norm_tab = []

		#iteration = 0

		#self.state = CheegerOptimizerState(initial_set, f)
		x_values = [v[0] for v in initial_set.boundary_vertices]
		y_values = [v[1] for v in initial_set.boundary_vertices]                 

		x_min= np.clip(min(x_values), 0,1)
		x_max= np.clip(max(x_values), 0,1)
		y_min= np.clip(min(y_values), 0,1)
		y_max= np.clip(max(y_values), 0,1)

		outer_vertices= np.array([x_min, x_max, y_min, y_max])
		rectangle_boundary_vertices= np.array([[x_min,y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])
		rectangle_set = RectangularSet(rectangle_boundary_vertices)
		
		num_vertices = rectangle_set.boundary_vertices.shape[0]  # Anzahl der Vertices
		print("original objective wrapper:" , rectangle_set.objective_wrapper(outer_vertices,f))
		print("original objective:" , rectangle_set.compute_objective(f))
		start = time.time()
		
		result = minimize(rectangle_set.objective_wrapper, outer_vertices, args=(f,), bounds =[(0,1),(0,1), (0,1), (0,1)]  , options={'maxiter': 10000, 'disp': True, 'ftol': 1e-7, 'gtol': 1e-6})
					  
		end = time.time()

		optimal_rectangle = result.x
		optimal_objective = result.fun
		print("Erfolg:", result.success)
		print("Nachricht:", result.message)
		print("Optimales Rechteck:", optimal_rectangle)
		print("Optimales Objective:", optimal_objective)
		
		

		opt_rectangle_boundary_vertices= np.array([[optimal_rectangle[0], optimal_rectangle[2]], [optimal_rectangle[0], optimal_rectangle[3]], [optimal_rectangle[1], optimal_rectangle[3]], [optimal_rectangle[1], optimal_rectangle[2]]])
		opt_rect_set = RectangularSet(opt_rectangle_boundary_vertices)      
		print("hab optimales rechteck gefunden")
		plot_rectangular_set(opt_rect_set, eta=f, display_inner_mesh=False)
		print("Die Berechnung des Rechtecks hat ", end - start, " Sekunden gedauert." )
		print("Perimeter korrekt:", opt_rect_set.compute_anisotropic_perimeter())
		print("Perimeter Rechteck:", opt_rect_set.compute_anisotropic_perimeter_convex())
		print("Value integral :", opt_rect_set.compute_weighted_area_rec(f))
		print("Objective:", opt_rect_set.compute_objective(f))
		print("Unterschied in objective vom original zu optimal:", rectangle_set.compute_objective(f)-  opt_rect_set.compute_objective(f))		  
		#return simple_set, obj_tab, grad_norm_tab, opt_rect_set
		return  opt_rect_set
	

	#def run_rectangular(self, f, initial_set, verbose=True):
		convergence = False
		obj_tab = []
		grad_norm_tab = []

		iteration = 0

		x_min, x_max = np.min(initial_set.boundary_vertices[:, 0]), np.max(initial_set.boundary_vertices[:, 0])
		y_min, y_max = np.min(initial_set.boundary_vertices[:, 1]), np.max(initial_set.boundary_vertices[:, 1])

		# Erstelle ein Array für die Eckpunkte im Uhrzeigersinn (oder gegen den Uhrzeigersinn)
		rect_vertices = np.array([
			[x_min, y_min],  # untere linke Ecke
			[x_max, y_min],  # untere rechte Ecke
			[x_max, y_max],  # obere rechte Ecke
			[x_min, y_max]   # obere linke Ecke
		])

		# Falls deine Set-Klasse eine Methode hat, um Boundary-Vertices zu setzen:
		initial_set = RectangularSet(rect_vertices,  max_tri_area=self.max_tri_area)
		

		self.state = CheegerOptimizerState(initial_set, f)

		while not convergence and iteration < self.max_iter:
			gradient = self.state.compute_gradient_rectangular(f)


			#gradient_field = np.zeros((self.state.grid_size, self.state.grid_size))
			#print(gradient_field.shape)
			#grad_norm = np.linalg.norm(gradient, axis=1)
			#for i, (x, y) in enumerate(self.state.set.boundary_vertices):
			#	gradient_field[int(y*self.state.grid_size -1), int(x*self.state.grid_size -1 )] = grad_norm[i]

			#gradient_magnitude = np.linalg.norm(gradient, axis=-1)

			# Visualisierung
			#plt.figure(figsize=(8, 6))
			#plt.imshow(gradient_field, cmap = 'viridis', origin = 'lower')
			#plt.colorbar(label="Gradient Magnitude")
			#plt.title("Gradientenbetrag des Funktionals")
			#plt.xlabel("x")
			#plt.ylabel("y")
			#plt.show()

			#x, y = self.state.set.boundary_vertices[:, 0]*self.state.grid_size, self.state.set.boundary_vertices[:, 1]*self.state.grid_size
			x_min, x_max = np.min(self.state.set.boundary_vertices[:, 0]) * self.state.grid_size, np.max(self.state.set.boundary_vertices[:, 0])* self.state.grid_size
			y_min, y_max = np.min(self.state.set.boundary_vertices[:, 1])* self.state.grid_size, np.max(self.state.set.boundary_vertices[:, 1])* self.state.grid_size

			x = np.array([x_min, x_max, (x_min + x_max) / 2, (x_min + x_max) / 2]) 
			y = np.array([(y_min + y_max) / 2, (y_min + y_max) / 2, y_min, y_max]) 
			
			eta_grid = f.integrate_on_pixel_grid(self.state.grid_size)
		

		 	# Plot für den Perimeter-Gradienten
		
			gradient_array = np.array([
			[gradient[0,0], 0],  # x_min: Änderung nur in x-Richtung
			[gradient[1,0], 0],  # x_max
			[0, gradient[2,0]],  # y_min: Änderung nur in y-Richtung
			[0, gradient[3,0]]   # y_max
				])

			#fig, axes = plt.subplots(1, 1, figsize=(12, 6))
			plt.plot()
			 # Plot für den Perimeter-Gradienten
			#im1 = axes.imshow(eta_grid.T, cmap='bwr', origin='lower', extent=[0, self.state.grid_size, 0, self.state.grid_size])
			#sc1 = axes.quiver(x, y, gradient[:,0],gradient[:,1], cmap='viridis', color='k')
			#axes.set_title("Perimeter-Gradient ")
			#fig.colorbar(im1, ax=axes[0], label=r'$\eta$')
			#fig.colorbar(sc1, ax=axes[0], label="Gradient")
			plt.imshow(eta_grid.T, cmap='bwr', origin='lower', extent=[0, self.state.grid_size, 0, self.state.grid_size])
			plt.quiver(x, y, gradient_array[:,0],gradient_array[:,1], cmap='viridis', color='k')
			plt.title("Gesamt-Gradient ")
			#fig.colorbar(im1, ax=axes[0], label=r'$\eta$')
			#fig.colorbar(sc1, ax=axes[0], label="Gradient")
			plt.colorbar()
			plt.show()

			grad_norm_tab.append(np.sum(np.linalg.norm(gradient, axis=-1)))
			
			#grad_norm_tab.append(np.sum(np.linalg.norm(gradient, ord=1, axis=-1)))
			obj_tab.append(self.state.obj)
			
			#print(obj_tab[-1])

			n_iter_linesearch, max_displacement = self.perform_linesearch_rectangular(f, gradient)
			#print("weighted area:", self.state.weighted_area)
			#print("perimeter:", self.state.perimeter)
			iteration += 1
			convergence = (max_displacement < self.eps_stop)

			if verbose:
				print("iteration {}: {} linesearch steps".format(iteration, n_iter_linesearch))


				iterations = range(1, len(obj_tab) + 1)

				fig, ax1 = plt.subplots(figsize=(10, 5))

				# Plot für die Zielfunktion
				color = 'tab:blue'
				ax1.set_xlabel('Iteration')
				ax1.set_ylabel('Zielfunktion', color=color)
				ax1.plot(iterations, obj_tab, color=color, label='Zielfunktion')
				ax1.tick_params(axis='y', labelcolor=color)
				ax1.legend(loc='upper left')

				# Zweite Achse für Gradienten-Norm
				ax2 = ax1.twinx()
				color = 'tab:red'
				ax2.set_ylabel('Gradienten-Norm', color=color)
				ax2.plot(iterations, grad_norm_tab, color=color, linestyle='--', label='Gradienten-Norm')
				ax2.tick_params(axis='y', labelcolor=color)
				ax2.legend(loc='upper right')

				plt.title('Verlauf der Zielfunktion und Gradienten-Norm (iteration: {iteration})')
				plt.show()

			if self.num_iter_resampling is not None and iteration % self.num_iter_resampling == 0:
				new_boundary_vertices = resample(self.state.set.boundary_vertices, num_points=self.num_points,
												 point_density=self.point_density)
				new_set = RectangularSet(new_boundary_vertices, max_tri_area=self.max_tri_area)
				self.state.update_set(new_set, f)
		
		return self.state.set, obj_tab, grad_norm_tab