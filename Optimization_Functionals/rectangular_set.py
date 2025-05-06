import numpy as np
import matplotlib.pyplot as plt


class RectangularSet:

	def __init__(self, x_min, x_max, y_min, y_max):
		
		self.x_min = x_min
		self.x_max = x_max
		self.y_min = y_min
		self.y_max = y_max

		self.coordinates = np.array([x_min, x_max, y_min, y_max])


	
	@property
	def coordinates(self):
		return  np.array([self.x_min, self.x_max, self.y_min, self.y_max])

	@coordinates.setter
	def coordinates(self, new_coordinates):
		self.x_min = new_coordinates[0]
		self.x_max = new_coordinates[1]
		self.y_min = new_coordinates[2]
		self.y_max = new_coordinates[3]



	def plot_rectangular_set(self, eta, grid_size):
		fig, ax = plt.subplots(figsize=(7, 7))
		
		x = np.linspace(0, grid_size, grid_size )
		y = np.linspace(0, grid_size, grid_size )
		
		x_grid, y_grid = np.meshgrid(x, y)
		z_grid = np.flipud(eta)


		v_abs_max = np.max(np.abs(z_grid))
		

		im = ax.contourf(x_grid, y_grid, z_grid, levels=30, cmap='bwr', vmin=-v_abs_max, vmax=v_abs_max)

		fig.colorbar(im, ax=ax)

		y_curve = grid_size -np.append([self.x_min, self.x_max, self.x_max, self.x_min], self.x_min)
		x_curve =  np.append([self.y_min, self.y_min,  self.y_max, self.y_max], self.y_min)

		ax.plot(x_curve, y_curve, color='black')

		ax.axis('equal')
		ax.axis('on') 
	

		ax.set_xlim(0, grid_size)
		ax.set_ylim(0, grid_size)
		plt.title("Rectangle displayed with target function")
	
		plt.show()


	
	""" Computation of components of min \ frac{Per(E)}{abs(\int_E \eta)}"""

	
	
	def compute_anisotropic_perimeter(self):
		"""
		Per(E)
		
		"""
		return 2 * ((self.x_max - self.x_min) + (self.y_max - self.y_min))
	
	
	def compute_anisotropic_perimeter_gradient(self):
		""" 
		(\ frac{d}{d x_min} Per(E), \ frac{d}{d x_max} Per(E), \ frac{d}{d y_min} Per(E), \ frac{d}{d y_max} Per(E) ) 
		
		"""
		gradient = np.array([-2, 2, -2, 2])
		return gradient



	

	def compute_integral(self, cut_off, weights, grid_size):
		"""
		  \int_{x_min} ^{x_max} \int_{y_min} ^{y_max} \sum_{k,j = -\Phi}^{\Phi} w_{k,j} e^{2 \pi i (k j)*(x y)} dy dx 


		"""

		weights[0,0] = 0 #ensures vanishing integral

		res = 0
		for k in range (- cut_off, cut_off +1 ):
			for l in range (- cut_off, cut_off + 1):

				if k == 0 and l == 0:
					res += weights[0,0]* 1/grid_size**2 * (self.y_max - self.y_min)*(self.x_max - self.x_min) 
				
				elif k == 0: 
					res += weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] / (grid_size*2*np.pi*1j *l)* (self.x_max * (np.exp(2*np.pi*1j*((l*self.y_max)/grid_size)) -  np.exp(2*np.pi*1j*((l*self.y_min)/grid_size)) )
																													   + self.x_min * (np.exp(2*np.pi*1j*((l*self.y_min)/grid_size)) -  np.exp(2*np.pi*1j*((l*self.y_max)/grid_size)) ))

				elif l == 0:
					res += weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] / (grid_size*2*np.pi*1j *k)* (self.y_max * (np.exp(2*np.pi*1j*((k*self.x_max)/grid_size)) -  np.exp(2*np.pi*1j*((k*self.x_min)/grid_size)) )
																													   + self.y_min * (np.exp(2*np.pi*1j*((k*self.x_min)/grid_size)) -  np.exp(2*np.pi*1j*((k*self.x_max)/grid_size)) ))

				else:
					res += weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] / (-(2*np.pi)**2 * k * l) *(np.exp(2*np.pi*1j * (k *self.x_max / grid_size + l* self.y_max / grid_size)) - np.exp( 2 * np.pi *1j * (k* self.x_max/grid_size + l* self.y_min/grid_size)) 
																												  - np.exp( 2 * np.pi *1j *(k* self.x_min/grid_size + l* self.y_max/grid_size)) + np.exp(2 * np.pi *1j*(k* self.x_min/grid_size + l* self.y_min/grid_size)))
		return res


	def compute_integral_gradient(self, cut_off, weights, grid_size):
		"""
		Saves the gradient of compute_integral in order of x_min, x_max, y_min, y_max

		"""
		weights[0,0] = 0
		gradient = np.zeros(4, dtype=np.complex128)	
		
		for k in range (- cut_off, cut_off +1 ):
			for l in range (- cut_off, cut_off + 1):

				if k == 0 and l == 0:
					gradient[0] += 0
					gradient[1] += 0
					gradient[2] += 0
					gradient[3] += 0 

				elif k == 0 and l != 0: 
					gradient[0] += weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] / (grid_size * 2*np.pi*1j*l ) *(np.exp( 2*np.pi* 1j * ( l * self.y_min)/ grid_size )  - np.exp( 2*np.pi* 1j * ( l * self.y_max)/ grid_size ) )

					gradient[1] += weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] / (grid_size * 2*np.pi*1j*l ) *(np.exp( 2*np.pi* 1j * ( l * self.y_max)/ grid_size )  - np.exp( 2*np.pi* 1j * ( l * self.y_min)/ grid_size ) )

					gradient[2] += weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] / (grid_size *grid_size ) *  (- np.exp( 2*np.pi* 1j * ( l * self.y_min)/ grid_size ) * self.x_max + np.exp( 2*np.pi* 1j * ( l * self.y_min)/ grid_size ) * self.x_min )

					gradient[3] += weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] / (grid_size *grid_size ) *  ( np.exp( 2*np.pi* 1j * ( l * self.y_max)/ grid_size ) * self.x_max - np.exp( 2*np.pi* 1j * ( l * self.y_max)/ grid_size ) * self.x_min )

				elif l == 0 and k != 0:
					gradient[0] += weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] / (grid_size *grid_size ) *  (- np.exp( 2*np.pi* 1j * ( k * self.x_min)/ grid_size ) * self.y_max + np.exp( 2*np.pi* 1j * ( k * self.x_min)/ grid_size ) * self.y_min )

					gradient[1] +=  weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] / (grid_size *grid_size ) *  ( np.exp( 2*np.pi* 1j * ( k * self.x_max)/ grid_size ) * self.y_max - np.exp( 2*np.pi* 1j * ( k * self.x_max)/ grid_size ) * self.y_min )

					gradient[2] += weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] / (grid_size * 2*np.pi*1j*k ) *(np.exp( 2*np.pi* 1j * ( k * self.x_min)/ grid_size)  - np.exp( 2*np.pi* 1j * ( k * self.x_max)/ grid_size ) )

					gradient[3] +=  weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] / (grid_size * 2*np.pi*1j*k ) *(np.exp( 2*np.pi* 1j * ( k * self.x_max)/ grid_size)  - np.exp( 2*np.pi* 1j * ( k * self.x_min)/ grid_size ) )
				
				else:
					
					
					
					gradient[0] += weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] * ( (1j) / ( - 2 * np.pi *l * grid_size) ) * ( - np.exp( 2 * np.pi * 1j * ((k * self.x_min) / (grid_size) + (l * self.y_max)/(grid_size))) 
																																			   + np.exp(  2 * np.pi * 1j * ((k * self.x_min) / (grid_size) + (l * self.y_min)/(grid_size))))  

					gradient[1] += weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] * ( (1j) / ( - 2 * np.pi *l * grid_size) ) * (  np.exp( 2 * np.pi * 1j * ((k * self.x_max) / (grid_size) + (l * self.y_max)/(grid_size))) 
																																			   - np.exp(  2 * np.pi * 1j * ((k * self.x_max) / (grid_size) + (l * self.y_min)/(grid_size))))  

					gradient[2] += weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] * ( (1j) / ( - 2 * np.pi * k * grid_size) ) * ( - np.exp( 2 * np.pi * 1j * ((k * self.x_max) / (grid_size) + (l * self.y_min)/(grid_size))) 
																																			   + np.exp(  2 * np.pi * 1j * ((k * self.x_min) / (grid_size) + (l * self.y_min)/(grid_size))))  

					gradient[3] += weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] * ( (1j) / ( - 2 * np.pi * k * grid_size) ) * (  np.exp( 2 * np.pi * 1j * ((k * self.x_max) / (grid_size) + (l * self.y_max)/(grid_size))) 
																																			   - np.exp(  2 * np.pi * 1j * ((k * self.x_min) / (grid_size) + (l * self.y_max)/(grid_size))))  
					
		return gradient



	def compute_objective(self, cut_off, weights, grid_size ):
		res = self.compute_anisotropic_perimeter() /  np.abs(self.compute_integral(cut_off, weights, grid_size) )
		return res
	
	def compute_objective_wrapper(self, x, cut_off, weights, grid_size):
		
		"""needed for scipy.optimize.minimize """
		
		self.x_min = x[0]
		self.x_max = x[1]
		self.y_min = x[2]
		self.y_max = x[3]
		
		
		return self.compute_objective(cut_off, weights, grid_size)
	
	def compute_objective_gradient(self, cut_off, weights, grid_size ):

		perimeter = self.compute_anisotropic_perimeter()
		perimeter_gradient = self.compute_anisotropic_perimeter_gradient()
		integral = self.compute_integral(cut_off, weights, grid_size)
		integral_gradient = self.compute_integral_gradient(cut_off, weights, grid_size)
		
		
		gradient = np.sign(integral) * (perimeter_gradient * integral - integral_gradient * perimeter) / integral ** 2

		return  gradient
	
	def objective_gradient_wrapper(self, x, cut_off, weights, grid_size):
		""""
		needed for scipy.optimize.minimize 
		"""
		self.x_min = x[0]
		self.x_max = x[1]
		self.y_min = x[2]
		self.y_max = x[3]
		gradient = np.real(self.compute_objective_gradient(cut_off, weights, grid_size))

		return np.ascontiguousarray(gradient, dtype=np.float64)

	
def construct_rectangular_set_from01(boundary_vertices, grid_size):
	x_values = [v[0] for v in boundary_vertices]
	y_values = [v[1] for v in boundary_vertices]                 

	x_min= grid_size * np.clip(min(x_values), 0, grid_size)
	x_max= grid_size * np.clip(max(x_values), 0,grid_size)
	y_min= grid_size * np.clip(min(y_values), 0, grid_size)
	y_max= grid_size * np.clip(max(y_values), 0, grid_size)

	rectangular_set = RectangularSet(x_min, x_max, y_min, y_max)

	return rectangular_set

def evaluate_inverse_fourier(x, cut_off, weights, grid_size ):
	
	res = 0
	for k in range (- cut_off, cut_off +1 ):
		for l in range (- cut_off, cut_off + 1):

			res += 1/ grid_size**2 * weights[(k+grid_size) % grid_size, (l+grid_size) % grid_size] * np.exp( 2* np.pi * 1j * (k*x[0] / grid_size + l*x[1] / grid_size))

	return res