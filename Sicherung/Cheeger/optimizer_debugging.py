import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.optimize import minimize

from .tools import resample
from .plot_utils import plot_rectangular_set
#from .simple_set import SimpleSet
from .rectangular_set_debugging import RectangularSet


class CheegerOptimizerState:
	def __init__(self, initial_set, f):
		self.set = None
		self.weighted_area_tab = None
		self.weighted_area = None
		self.perimeter = None
		self.obj = None

		self.grid_size = None

		self.update_set(initial_set, f)

	def update_obj(self):
		self.weighted_area = np.sum(self.weighted_area_tab)
		self.perimeter = self.set.compute_anisotropic_perimeter()

		self.obj = (self.perimeter / np.abs(self.weighted_area))

	def update_boundary_vertices(self, new_boundary_vertices, f):
		self.set.boundary_vertices = new_boundary_vertices

		boundary_weighted_area_tab = self.set.compute_weighted_area_rec_tab(f, boundary_faces_only=True)
		self.weighted_area_tab[self.set.mesh_boundary_faces_indices] = boundary_weighted_area_tab

		self.update_obj()

	def update_set(self, new_set, f):
		self.set = new_set
		self.grid_size = f.grid_size

		weighted_area_tab = self.set.compute_weighted_area_rec_tab(f)
		self.weighted_area_tab = weighted_area_tab

		self.update_obj()

	#hier überall rec hinzu
	def compute_gradient(self, f):
		perimeter_gradient = self.set.compute_anisotropic_perimeter_gradient()
		#print("anisotropic perimeter gradients", perimeter_gradient)
		#left_perimeter_gradient, right_perimeter_gradient = self.set.compute_anisotropic_perimeter_gradient()
		#print("left anisotropic perimeter gradients", left_perimeter_gradient, "right anisotropic perimeter gradient:", right_perimeter_gradient)
		#perimeter_gradient = (left_perimeter_gradient + right_perimeter_gradient) / 2
		#print("mean perimeter gradient:", mean_perimeter_gradient)
		area_gradient = self.set.compute_weighted_area_rec_gradient(f)
		#gradient = (mean_perimeter_gradient * self.weighted_area - area_gradient * self.perimeter) / self.weighted_area ** 2
		gradient = (perimeter_gradient * self.weighted_area - area_gradient * self.perimeter) / self.weighted_area ** 2
		print("Fläche:", self.weighted_area)
		print("Perimeter:", self.perimeter)
		#print("Norm Perimetergradient:", np.linalg.norm(mean_perimeter_gradient))
		#print("Norm Flächengradient:", np.linalg.norm(area_gradient))
		
		
		# Extrahiere die Koordinaten der Boundary-Vertices
		x, y = self.set.boundary_vertices[:, 0]*self.grid_size, self.set.boundary_vertices[:, 1]*self.grid_size
		eta_grid = f.integrate_on_pixel_grid(self.grid_size)
		# Berechnung der Gradientennormen
		#grad_per = np.linalg.norm(mean_perimeter_gradient, axis=1)
		#grad_area = np.linalg.norm(area_gradient, axis=1)

		# Erstelle zwei Plots
		fig, axes = plt.subplots(1, 2, figsize=(12, 6))

		 # Plot für den Perimeter-Gradienten
		im1 = axes[0].imshow(eta_grid.T, cmap='bwr', origin='lower', extent=[0, self.grid_size, 0, self.grid_size])
		sc1 = axes[0].quiver(x, y, perimeter_gradient[:,0], perimeter_gradient[:,1], cmap='viridis', color='k')
		axes[0].set_title("Perimeter-Gradient ")
		fig.colorbar(im1, ax=axes[0], label=r'$\eta$')
		fig.colorbar(sc1, ax=axes[0], label="Gradient")

		# Plot für den Flächen-Gradienten
		im2 = axes[1].imshow(eta_grid.T, cmap='bwr', origin='lower', extent=[0, self.grid_size, 0, self.grid_size])
		sc2 = axes[1].quiver(x, y, area_gradient[:,0], area_gradient[:,1], cmap='viridis', color='k')
		axes[1].set_title("Flächen-Gradient ")
		fig.colorbar(im2, ax=axes[1], label=r'$\eta$')
		fig.colorbar(sc2, ax=axes[1], label="Gradient")

		plt.tight_layout()
		plt.show()

		
		
		return np.sign(self.weighted_area) * gradient
	


	def compute_gradient_rectangular(self, f):
		perimeter_gradient = self.set.compute_anisotropic_perimeter_gradient_rectangular()
		perimeter_gradient = perimeter_gradient[:, np.newaxis]
		area_gradient = self.set.compute_weighted_area_gradient_rectangular(f)
		
		gradient = (perimeter_gradient * self.weighted_area - area_gradient * self.perimeter) / self.weighted_area ** 2
		
		print("dimension perimeter gradient", perimeter_gradient.shape)
		print("dimension area ", area_gradient.shape)
		print("dimension gradient", gradient.shape)
		print("Fläche:", self.weighted_area)
		print("Perimeter:", self.perimeter)
		#print("Norm Perimetergradient:", np.linalg.norm(mean_perimeter_gradient))
		#print("Norm Flächengradient:", np.linalg.norm(area_gradient))
		
		
		# Extrahiere die Koordinaten der Boundary-Vertices
		#x, y = self.set.boundary_vertices[:, 0]*self.grid_size, self.set.boundary_vertices[:, 1]*self.grid_size
		x_min, x_max = np.min(self.set.boundary_vertices[:, 0]), np.max(self.set.boundary_vertices[:, 0])
		y_min, y_max = np.min(self.set.boundary_vertices[:, 1]), np.max(self.set.boundary_vertices[:, 1])

		x = np.array([x_min, x_max, (x_min + x_max) / 2, (x_min + x_max) / 2]) * self.grid_size
		y = np.array([(y_min + y_max) / 2, (y_min + y_max) / 2, y_min, y_max]) * self.grid_size
		
		eta_grid = f.integrate_on_pixel_grid(self.grid_size)
		# Berechnung der Gradientennormen
		#grad_per = np.linalg.norm(mean_perimeter_gradient, axis=1)
		#grad_area = np.linalg.norm(area_gradient, axis=1)

		print("Area Gradient (raw):", area_gradient)
		print("Perimeter Gradient (raw):", perimeter_gradient)

		perimeter_gradient = np.array([
		[perimeter_gradient[0,0], 0],  # x_min: Änderung nur in x-Richtung
		[perimeter_gradient[1,0], 0],  # x_max
		[0, perimeter_gradient[2,0]],  # y_min: Änderung nur in y-Richtung
		[0, perimeter_gradient[3,0]]   # y_max
			])
		
		area_gradient = np.array([
		[area_gradient[0,0], 0],  # x_min: Änderung nur in x-Richtung
		[area_gradient[1,0], 0],  # x_max
		[0, area_gradient[2,0]],  # y_min: Änderung nur in y-Richtung
		[0, area_gradient[3,0]]   # y_max
			])

		# Erstelle zwei Plots
		fig, axes = plt.subplots(1, 2, figsize=(12, 6))

		 # Plot für den Perimeter-Gradienten
		im1 = axes[0].imshow(eta_grid.T, cmap='bwr', origin='lower', extent=[0, self.grid_size, 0, self.grid_size])
		sc1 = axes[0].quiver(x, y, perimeter_gradient[:,0], perimeter_gradient[:,1], cmap='viridis', color='k')
		axes[0].set_title("Perimeter-Gradient ")
		fig.colorbar(im1, ax=axes[0], label=r'$\eta$')
		fig.colorbar(sc1, ax=axes[0], label="Gradient")

		# Plot für den Flächen-Gradienten
		im2 = axes[1].imshow(eta_grid.T, cmap='bwr', origin='lower', extent=[0, self.grid_size, 0, self.grid_size])
		sc2 = axes[1].quiver(x, y, area_gradient[:,0], area_gradient[:,1], cmap='viridis', color='k')
		axes[1].set_title("Flächen-Gradient ")
		fig.colorbar(im2, ax=axes[1], label=r'$\eta$')
		fig.colorbar(sc2, ax=axes[1], label="Gradient")

		plt.tight_layout()
		plt.show()

		print("gradient:", np.sign(self.weighted_area)*gradient)
		
		return np.sign(self.weighted_area) * gradient

class CheegerOptimizer:
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

	def perform_linesearch(self, f, gradient):
		t = self.step_size

		ag_condition = False

		former_obj = self.state.obj
		former_boundary_vertices = self.state.set.boundary_vertices
		
		
		#print("original boundary vertices:", former_boundary_vertices)

		iteration = 0
		plt.figure()
		plt.imshow(f.integrate_on_pixel_grid(80).T,  cmap = 'bwr')
		plt.scatter(former_boundary_vertices[:, 0]*80,former_boundary_vertices[:, 1]*80)
		plt.title("original boundaries")
		plt.show()

		#print("Gradient shape:", gradient.shape)
		#print("Gradient min/max:", np.min(gradient), np.max(gradient))
		#print("Gradient values (first 5 rows):", gradient[:5])

		#print("Former boundary vertices min/max x:", np.min(former_boundary_vertices[:,0]), np.max(former_boundary_vertices[:,0]))
		#print("Former boundary vertices min/max y:", np.min(former_boundary_vertices[:,1]), np.max(former_boundary_vertices[:,1]))
		#print("gradient for line search:", gradient)
		while not ag_condition:
			#new_boundary_vertices = np.clip(former_boundary_vertices - t * gradient, 0, 1)
			new_boundary_vertices = np.clip(former_boundary_vertices - t * gradient, 0, 1)
			
			
			plt.figure(figsize=(6,6))
			plt.imshow(f.integrate_on_pixel_grid(80).T,  cmap = 'bwr')
			plt.scatter(new_boundary_vertices[:, 0]*80, new_boundary_vertices[:, 1]*80)
			plt.title("new boundaries boundaries from iteration {}".format(iteration))
			plt.show()

			#print("new boundary vertices iteratiron", iteration, ":", new_boundary_vertices)

			self.state.update_boundary_vertices(new_boundary_vertices, f)
			new_obj = self.state.obj
			print("new objective:", new_obj)
			print("former objective:" , former_obj)
			print("perimeter calculation convex:", self.state.set.compute_anisotropic_perimeter_convex())
			print("perimeter calculation correct:", self.state.set.compute_anisotropic_perimeter())
			ag_condition = (new_obj <= former_obj - self.alpha * t * np.abs(gradient).sum())
			t = self.beta * t

			iteration += 1

		max_displacement = np.max(np.linalg.norm(new_boundary_vertices - former_boundary_vertices, axis=-1))
		print("AG condition satisfied:", ag_condition)
		print("max displacement", max_displacement)
		return iteration, max_displacement
	
	def perform_linesearch_rectangular(self, f, gradient):
		t = self.step_size

		ag_condition = False

		former_obj = self.state.obj
		former_boundary_vertices = self.state.set.boundary_vertices
		
		x_min, x_max = np.min(former_boundary_vertices[:, 0]), np.max(former_boundary_vertices[:, 0])
		y_min, y_max = np.min(former_boundary_vertices[:, 1]), np.max(former_boundary_vertices[:, 1])

		
		former_parameters = np.array([x_min, x_max, y_min, y_max])
		#print("original boundary vertices:", former_boundary_vertices)
		former_parameters = former_parameters[:, np.newaxis]
		iteration = 0
		plt.figure()
		plt.imshow(f.integrate_on_pixel_grid(80).T,  cmap = 'bwr')
		plt.scatter(former_boundary_vertices[:, 0]*80,former_boundary_vertices[:, 1]*80)
		plt.title("original boundaries")
		plt.show()

		#print("Gradient shape:", gradient.shape)
		#print("Gradient min/max:", np.min(gradient), np.max(gradient))
		#print("Gradient values (first 5 rows):", gradient[:5])

		#print("Former boundary vertices min/max x:", np.min(former_boundary_vertices[:,0]), np.max(former_boundary_vertices[:,0]))
		#print("Former boundary vertices min/max y:", np.min(former_boundary_vertices[:,1]), np.max(former_boundary_vertices[:,1]))
		#print("gradient for line search:", gradient)
		while not ag_condition:
			print("gradient:", gradient)
			print("shape gradient:" ,gradient.shape)
			print("former boundaries shape", former_boundary_vertices.shape)
			print("former parameters shape", former_parameters.shape)
			
			#new_boundary_vertices = np.clip(former_boundary_vertices - t * gradient, 0, 1)
			new_parameters = np.clip(former_parameters - t * gradient, 0, 1)
			new_boundary_vertices = np.array([[new_parameters[0,0],new_parameters[2,0]], [new_parameters[1,0],new_parameters[2,0]], [new_parameters[1,0],new_parameters[3,0]], [new_parameters[0,0],new_parameters[3,0]]])
			
			new_boundary_vertices = np.squeeze(new_boundary_vertices)
			print("new boundary vertices:",new_boundary_vertices)
			plt.figure(figsize=(6,6))
			plt.imshow(f.integrate_on_pixel_grid(80).T,  cmap = 'bwr')
			plt.scatter(new_boundary_vertices[:, 0]*80, new_boundary_vertices[:, 1]*80)
			plt.title("new boundaries boundaries from iteration {}".format(iteration))
			plt.show()

			#print("new boundary vertices iteratiron", iteration, ":", new_boundary_vertices)

			self.state.update_boundary_vertices(new_boundary_vertices, f)
			new_obj = self.state.obj
			print("new objective:", new_obj)
			print("former objective:" , former_obj)
			print("perimeter calculation convex:", self.state.set.compute_anisotropic_perimeter_convex())
			print("perimeter calculation correct:", self.state.set.compute_anisotropic_perimeter())
			ag_condition = (new_obj <= former_obj - self.alpha * t * np.abs(gradient).sum())
			t = self.beta * t

			iteration += 1

		max_displacement = np.max(np.linalg.norm(new_boundary_vertices - former_boundary_vertices, axis=-1))
		print("AG condition satisfied:", ag_condition)
		print("max displacement", max_displacement)
		return iteration, max_displacement

	def run(self, f, initial_set, verbose=True):
		convergence = False
		obj_tab = []
		grad_norm_tab = []

		iteration = 0

		self.state = CheegerOptimizerState(initial_set, f)

		while not convergence and iteration < self.max_iter:
			gradient = self.state.compute_gradient(f)


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
			x, y = self.state.set.boundary_vertices[:, 0]*self.state.grid_size, self.state.set.boundary_vertices[:, 1]*self.state.grid_size
			eta_grid = f.integrate_on_pixel_grid(self.state.grid_size)
		

		 	# Plot für den Perimeter-Gradienten
		

			#fig, axes = plt.subplots(1, 1, figsize=(12, 6))
			plt.plot()
			 # Plot für den Perimeter-Gradienten
			#im1 = axes.imshow(eta_grid.T, cmap='bwr', origin='lower', extent=[0, self.state.grid_size, 0, self.state.grid_size])
			#sc1 = axes.quiver(x, y, gradient[:,0],gradient[:,1], cmap='viridis', color='k')
			#axes.set_title("Perimeter-Gradient ")
			#fig.colorbar(im1, ax=axes[0], label=r'$\eta$')
			#fig.colorbar(sc1, ax=axes[0], label="Gradient")
			plt.imshow(eta_grid.T, cmap='bwr', origin='lower', extent=[0, self.state.grid_size, 0, self.state.grid_size])
			plt.quiver(x, y, gradient[:,0],gradient[:,1], cmap='viridis', color='k')
			plt.title("Gesamt-Gradient ")
			#fig.colorbar(im1, ax=axes[0], label=r'$\eta$')
			#fig.colorbar(sc1, ax=axes[0], label="Gradient")
			plt.colorbar()
			plt.show()

			grad_norm_tab.append(np.sum(np.linalg.norm(gradient, axis=-1)))
			
			#grad_norm_tab.append(np.sum(np.linalg.norm(gradient, ord=1, axis=-1)))
			obj_tab.append(self.state.obj)
			
			#print(obj_tab[-1])

			n_iter_linesearch, max_displacement = self.perform_linesearch(f, gradient)
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
	

	def run_rectangular(self, f, initial_set, verbose=True):
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