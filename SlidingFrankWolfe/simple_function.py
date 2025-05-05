import numpy as np
import matplotlib.pyplot as plt



from numba import jit


class IndicatorFunction:
	def __init__(self, rectangular_set, grid_size, weight = 1):
		self.support = rectangular_set
		self.grid_size = grid_size
		self.weight = weight
		self.area = (rectangular_set.x_max - rectangular_set.x_min) * (rectangular_set.y_max - rectangular_set.y_min)
		self.mean_value = self.area / (self.grid_size ** 2)

		self.inner_value = self.weight * (1 - self.mean_value)
		self.outer_value = self.weight * (0 - self.mean_value)


	def __call__(self, x):

		if self.support.x_min <= x[1] <= self.support.x_max and self.support.y_min <= x[0] <= self.support.y_max:
			return self.weight * (1- self.mean_value)
		else:
			return self.weight * (0 - self.mean_value)
		

	def construct_image_matrix(self, plot = True):
		x = np.linspace(0, self.grid_size, self.grid_size)
		y = np.linspace(0, self.grid_size, self.grid_size)

		X, Y = np.meshgrid(x,y)
		grid = np.stack([X,Y], axis = -1)

		image = np.array([[self(np.array([xi, yi])) for xi, yi in row] for row in grid])
		
		if plot == True:
			plt.imshow(image,  cmap= 'bwr')
			plt.colorbar()
			plt.show()
		
		return image

	def compute_truncated_frequency_image(self, cut_off, plot = True):
		image = self.construct_image_matrix(plot = plot)
		fourier_image = np.fft.fft2(image)
		freqs_x= np.fft.fftfreq(self.grid_size, d=1 / self.grid_size)
		freqs_y = np.fft.fftfreq(self.grid_size, d=1 / self.grid_size)
		freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")

		mask = np.zeros((self.grid_size, self.grid_size))
		mask[(np.abs(freq_x) <= cut_off) & (np.abs(freq_y) <= cut_off)] = 1

		truncated_fourier_image = fourier_image * mask

		if plot == True:
			plt.imshow(truncated_fourier_image.real, cmap = 'bwr')
			plt.colorbar()
			plt.show()
		
		return truncated_fourier_image


class SimpleFunction:
	def __init__(self, atoms, grid_size, cut_off):
		if not isinstance(atoms, list):
			atoms = [atoms]
		self.atoms = atoms
		self.grid_size = grid_size
		self.cut_off = cut_off
		


	def __call__(self, x):
		res = 0
		for f in self.atoms:
			res += f(x)
		return res
	
	@property
	def num_atoms(self):
		return len(self.atoms)


	def extend_support(self, rectangular_set):
		new_atom = IndicatorFunction(rectangular_set, self.grid_size)
		self.atoms.append(new_atom)


	def remove_small_atoms(self, threshold = 1e-3):
		remaining_atoms = []
		removed_count = 0
		

		for i in range(self.num_atoms):
			if np.abs(self.atoms[i].weight) > threshold:
				remaining_atoms.append(self.atoms[i])

			else:
				removed_count += 1
		print (f"{removed_count} Atome mit Gewicht < {threshold} entfernt.")
		self.atoms = remaining_atoms
		

		

	

	def construct_image_matrix_sf(self, plot = True):
		x = np.linspace(0, self.grid_size, self.grid_size)
		y = np.linspace(0, self.grid_size, self.grid_size)

		X, Y = np.meshgrid(x,y)
		grid = np.stack([X,Y], axis = -1)

		image = np.array([[self(np.array([xi, yi])) for xi, yi in row] for row in grid])
		
		if plot == True:
			plt.imshow(image,  cmap= 'bwr')
			plt.colorbar()
			plt.title("Current Simple Function")
			plt.show()
		
		return image



	
	
	def compute_truncated_frequency_image_sf(self, plot = False):
		image = self.construct_image_matrix_sf(plot = plot)
		fourier_image = np.fft.fft2(image)
		freqs_x= np.fft.fftfreq(self.grid_size, d=1 / self.grid_size)
		freqs_y = np.fft.fftfreq(self.grid_size, d=1 / self.grid_size)
		freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")

		mask = np.zeros((self.grid_size, self.grid_size))
		mask[(np.abs(freq_x) <= self.cut_off) & (np.abs(freq_y) <= self.cut_off)] = 1

		truncated_fourier_image = fourier_image * mask

		if plot == True:
			plt.imshow(truncated_fourier_image.real, cmap = 'bwr')
			plt.colorbar()
			plt.title("Current Truncated Fourier Frequency Image")
			plt.show()
		
		return truncated_fourier_image


		

	def compute_fourier_integral(self, k1, k2):
		weights = np.array([atom.weight for atom in self.atoms], dtype=complex)
		x_min = np.array([atom.support.x_min for atom in self.atoms], dtype=float)
		x_max = np.array([atom.support.x_max for atom in self.atoms], dtype=float)
		y_min = np.array([atom.support.y_min for atom in self.atoms], dtype=float)
		y_max = np.array([atom.support.y_max for atom in self.atoms], dtype=float)
		return self._compute_fourier_integral(k1, k2, weights, x_min, x_max, y_min, y_max, self.grid_size)

	@staticmethod
	@jit(nopython = True)
	def _compute_fourier_integral(k1, k2, weights, x_min, x_max, y_min, y_max, grid_size):
		if k1 == 0 and k2 == 0:
			return 0

		
	

		
		pi = np.pi
		j = 1j  # imaginäre Einheit
		factor = -2 * pi * j / grid_size

		if k1 == 0:
			y_factor = factor * k2
			exp_ymax = np.exp(y_factor * y_max)
			
			exp_ymin = np.exp(y_factor * y_min)
			
			term = (exp_ymin - exp_ymax)*(x_max - x_min)
			
			sum_result = np.sum(weights * (grid_size / (2 * pi * j * k2)) * term)
			return sum_result

		elif k2 == 0:
			x_factor = factor * k1
			exp_xmax = np.exp(x_factor * x_max)
			
			exp_xmin = np.exp(x_factor * x_min)
			
			term = (exp_xmin - exp_xmax)*(y_max - y_min)
			
			sum_result = np.sum(weights * (grid_size / (2 * pi * j * k1)) * term)
			return sum_result

		else:
			y_factor = factor * k2
			x_factor = factor * k1

			exp_xmax = np.exp(x_factor * x_max)
			exp_xmin = np.exp(x_factor * x_min)
			exp_ymax = np.exp(y_factor * y_max)
			exp_ymin = np.exp(y_factor * y_min)

			

			term = (exp_xmax - exp_xmin) * (exp_ymax - exp_ymin)
			
			sum_result = np.sum(weights * (grid_size ** 2) / ((2 * pi * j) ** 2 * k1 * k2) * term)
			
			return sum_result


	def compute_error_term(self, target_function_f):

		"""
		1/2 \sum__{k1, k2} | \sum_j a_j \int_{x_min_j}^{x_max_j} \int_{y_min_j}^{y_max_j} e^{-2 pi i (x*k1 / grid_size + y*k2/ grid_size)} dy dx - f(k1, k2)|^2
		"""
		image = np.zeros((self.grid_size, self.grid_size), dtype = complex)
		res = 0
		for k1 in range(- self.cut_off, self.cut_off +1):
			for k2 in range(-self.cut_off, self.cut_off +1):

				if k1==0 and k2==0:
					continue

				val = self.compute_fourier_integral(k1, k2)
				image[(k1+self.grid_size) % self.grid_size, (k2+self.grid_size) % self.grid_size] = val



				res += np.abs(val - target_function_f[(k1+self.grid_size) % self.grid_size, (k2+self.grid_size) % self.grid_size])**2


		return 0.5 * res

	


	def compute_objective_sliding(self, target_function_f, reg_param):

		"""
		J_\ alpha(a, E) = 1/2 \Vert \sum_i a_i K0(1_E_i) -f \Vert^2 + \ alpha \sum_i \ text{perimeter}(E_i) abs(a_i)
		= 1/2 * sum_k | \sum_j a_j \int_{x_min_j}^{x_max_j} \int_{y_min_j}^{y_max_j} e^{-2 pi i (x*k1 / grid_size + y*k2/ grid_size)} dy dx - f(k1, k2)|^2 + \ alpha \sum_i 2*(x_max,j - x_min,j)*(y_max,j - y_min,j) *|a_i|
		
		"""

		error_term = self.compute_error_term(target_function_f)
		regularization_term = 0
		for atom in self.atoms:
			regularization_term += 2*((atom.support.x_max - atom.support.x_min) +(atom.support.y_max - atom.support.y_min))* np.abs(atom.weight)
		regularization_term *= reg_param

		return error_term + regularization_term


	def compute_derivative_fourier_integral(self, k1, k2):
		weights = []
		x_mins = []
		x_maxs = []
		y_mins = []
		y_maxs = []

		for atom in self.atoms:
			weights.append(atom.weight)
			x_mins.append(atom.support.x_min)
			x_maxs.append(atom.support.x_max)
			y_mins.append(atom.support.y_min)
			y_maxs.append(atom.support.y_max)

		weights = np.array(weights)
		x_mins = np.array(x_mins)
		x_maxs = np.array(x_maxs)
		y_mins = np.array(y_mins)
		y_maxs = np.array(y_maxs)

		return self._compute_derivative_fourier_integral(k1, k2, weights, x_mins, x_maxs, y_mins, y_maxs, self.grid_size)




	
	@staticmethod
	@jit(nopython = True)
	def _compute_derivative_fourier_integral(k1, k2, weights, x_mins, x_maxs, y_mins, y_maxs, grid_size):
		gradient = np.zeros((len(weights), 5), dtype=np.complex128)

		for i in range(len(weights)):
			
		
			# Precompute values to avoid repeated calculations
			y_max_exp = np.exp(-2 * np.pi * 1j * y_maxs[i] * k2 / grid_size)
			y_min_exp = np.exp(-2 * np.pi * 1j * y_mins[i] * k2 / grid_size)
			x_max_exp = np.exp(-2 * np.pi * 1j * x_maxs[i] * k1 / grid_size)
			x_min_exp = np.exp(-2 * np.pi * 1j * x_mins[i] * k1 / grid_size)
		
			if k1 == 0 and k2 == 0:
				continue
			elif k1 == 0:  # Derivatives when k1 == 0
				common_factor = grid_size / (2 * np.pi * 1j * k2)
				gradient[i, 0] = common_factor * (-(y_max_exp - y_min_exp) * x_maxs[i] + (y_max_exp - y_min_exp) * x_mins[i])
				gradient[i, 1] = weights[i] * common_factor * (y_max_exp - y_min_exp)
				gradient[i, 2] = weights[i] * common_factor * (-y_max_exp + y_min_exp)
				gradient[i, 3] = weights[i] * (-y_min_exp * x_maxs[i] + y_min_exp * x_mins[i])
				gradient[i, 4] = weights[i] * (y_max_exp * x_maxs[i] - y_max_exp * x_mins[i])
			elif k2 == 0:  # Derivatives when k2 == 0
				common_factor = grid_size / (2 * np.pi * 1j * k1)
				gradient[i, 0] = common_factor * (-(x_max_exp - x_min_exp) * y_maxs[i] + (x_max_exp - x_min_exp) * y_mins[i])
				gradient[i, 1] = weights[i] * (-x_min_exp * y_maxs[i] + x_min_exp * y_mins[i])
				gradient[i, 2] = weights[i] * (x_max_exp * y_maxs[i] - x_max_exp * y_mins[i])
				gradient[i, 3] = weights[i] * common_factor * (x_max_exp - x_min_exp)
				gradient[i, 4] = weights[i] * common_factor * (-x_max_exp + x_min_exp)
			else:  # General case
				factor_a = grid_size ** 2 / (-(2 * np.pi) ** 2 * k1 * k2)
				factor_x = (grid_size * 1j)/(- 2 * np.pi *k2)
				factor_y = (grid_size * 1j)/(- 2 * np.pi *k1)
				gradient[i, 0] = factor_a * (x_max_exp * y_max_exp - x_max_exp * y_min_exp - x_min_exp * y_max_exp + x_min_exp * y_min_exp)
				gradient[i, 1] = weights[i] * factor_x * (x_min_exp * y_max_exp - x_min_exp * y_min_exp)
				gradient[i, 2] = weights[i] * factor_x  * (-x_max_exp * y_max_exp + x_max_exp * y_min_exp)
				gradient[i, 3] = weights[i] * factor_y  * (x_max_exp * y_min_exp - x_min_exp * y_min_exp)
				gradient[i, 4] = weights[i] * factor_y  * (-x_max_exp * y_max_exp + x_min_exp * y_max_exp)

		return gradient


	
	def compute_gradient_sliding(self, target_function_f, reg_param):
		gradient_error_term = np.zeros((len(self.atoms), 5))
	
		grid_range = range(-self.cut_off, self.cut_off + 1)
		for k1 in grid_range:
			for k2 in grid_range:
				if k1 == 0 and k2 == 0:
					continue
				gradient_integral = self.compute_derivative_fourier_integral(k1, k2)
				error_term_conjugate = np.conj(self.compute_fourier_integral(k1, k2) - target_function_f[(k1 + self.grid_size) % self.grid_size, (k2 + self.grid_size) % self.grid_size])
				gradient_error_term += np.real(error_term_conjugate * gradient_integral)

		gradient_regularization_term = reg_param * 2 * np.array([[(atom.support.x_max - atom.support.x_min + atom.support.y_max - atom.support.y_min)* np.sign(atom.weight),
															  -np.abs(atom.weight), np.abs(atom.weight),
															  -np.abs(atom.weight), np.abs(atom.weight)] 
															 for atom in self.atoms])
		return gradient_error_term + gradient_regularization_term
	

	


	def gradient_wrapper_sliding(self, x, target_function_f, reg_param):
		""""
		needed to integrate into scipy.optimize.minimize
		"""
		for i in range (len(self.atoms)):
			self.atoms[i].weight = x[i]
			self.atoms[i].support.x_min = x[len(self.atoms) + i]
			self.atoms[i].support.x_max = x[2* len(self.atoms) + i]
			self.atoms[i].support.y_min = x[3* len(self.atoms) + i]
			self.atoms[i].support.y_max = x[4 *len(self.atoms) + i]

		gradient = self.compute_gradient_sliding(target_function_f, reg_param)
		
		return gradient.T.flatten()
	

	def objective_wrapper_sliding(self, x, target_function_f, reg_param):
		""""
		nedded to integrate into scipy.optimize.minimize
		"""
		for i in range (len(self.atoms)):
			self.atoms[i].weight = x[i]
			self.atoms[i].support.x_min = x[len(self.atoms) + i]
			self.atoms[i].support.x_max = x[2* len(self.atoms) + i]
			self.atoms[i].support.y_min = x[3* len(self.atoms) + i]
			self.atoms[i].support.y_max = x[4 *len(self.atoms) + i]

		return self.compute_objective_sliding( target_function_f, reg_param)
	



	

		

	



  
 
	