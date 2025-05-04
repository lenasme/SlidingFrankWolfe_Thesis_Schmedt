import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from .tools import winding, triangulate, triangulate_combined

class RectangularSet:

	def __init__(self, boundary_vertices,max_tri_area=None, domain_vertices = np.array([[0,0], [0,1], [1,1], [1,0]])):
		self.num_boundary_vertices = len(boundary_vertices)

		# the curve is clockwise if and only if the sum over the edges of (x2-x1)(y2+y1) is positive
		rolled_boundary_vertices = np.roll(boundary_vertices, -1, axis=0)
		self.is_clockwise = (np.sum((rolled_boundary_vertices[:, 0] - boundary_vertices[:, 0]) *
									(rolled_boundary_vertices[:, 1] + boundary_vertices[:, 1])) > 0)
		self.mesh_vertices = None
		self.mesh_faces = None
		self.mesh_boundary_faces_indices = None
		# creation of the inner mesh
		
		#self.create_whole_mesh(boundary_vertices, domain_vertices, max_tri_area)
		self.create_mesh(boundary_vertices, max_tri_area)

		#self.plot_meshes(self.mesh_vertices, self.mesh_faces)

	@property
	def boundary_vertices_indices(self):
		return np.arange(self.num_boundary_vertices)

	# TODO: attention du coup self.boundary_vertices est vue comme une fonction, a voir si probleme avec numba
	@property
	def boundary_vertices(self):
		return self.mesh_vertices[self.boundary_vertices_indices]

	@property
	def mesh_boundary_faces(self):
		return self.mesh_faces[self.mesh_boundary_faces_indices]

	@property
	def minimal_x(self):
		x_values = [v[0] for v in self.boundary_vertices]
		min_x = min(x_values)
		return min_x

	@property
	def maximal_x(self):
		x_values = [v[0] for v in self.boundary_vertices]
		max_x = max(x_values)
		return max_x

	@property
	def minimal_y(self):
		y_values = [v[1] for v in self.boundary_vertices]
		min_y = min(y_values)
		return min_y

	@property
	def maximal_y(self):
		y_values = [v[1] for v in self.boundary_vertices]
		max_y = max(y_values)
		return max_y


	@boundary_vertices.setter
	def boundary_vertices(self, new_boundary_vertices):
		self.mesh_vertices[self.boundary_vertices_indices] = new_boundary_vertices

	
	def plot_meshes(self, vertices, faces):
		plt.figure(figsize=(8, 8))
	
		# Erstelle das Triangulationsobjekt
		triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], faces)
	
		# Zeichne das Mesh
		plt.triplot(triang, color='black', linewidth=0.5)
	
		# Optional: Markiere die Punkte
		plt.scatter(vertices[:, 0], vertices[:, 1], color='red', s=10, zorder=2)

		plt.xlabel("X")
		plt.ylabel("Y")
		plt.title("Trianguliertes Mesh")
		plt.axis("equal")  # Gleiche Skalierung der Achsen
		plt.show()


	def contains(self, x):
		"""
		Whether a given point x is inside the set or not

		Parameters
		----------
		x : array, shape (2,)
		The input point

		Returns
		------
		bool
		Whether x is in the set or not

		"""
		# The point is inside the set if and only if its winding number is non zero
		return winding(x, self.boundary_vertices) != 0


	def compute_area_rec(self):
		res = np.abs(self.maximal_x - self.minimal_x) * np.abs(self.maximal_y - self.minimal_y)
		return res
	

	def compute_perimeter_rec(self):
		"""
		Compute the perimeter of the set

		Returns
		-------
		float
		The perimeter

		"""
		x_values = [v[0] for v in self.boundary_vertices]
		y_values = [v[1] for v in self.boundary_vertices]

		min_x = min(x_values)
		max_x = max(x_values)
		min_y = min(y_values)
		max_y = max(y_values) 

		res = 2*(max_x - min_x) +2*(max_y - min_y)
		# hier umschreiben mit properties von oben self.minimal_x etc?
		
		return res
	
	def compute_anisotropic_perimeter_convex(self):
		x_min, y_min = np.min(self.boundary_vertices, axis=0)
		x_max, y_max = np.max(self.boundary_vertices, axis=0)
		return 2 * ((x_max - x_min) + (y_max - y_min))
	
	def compute_anisotropic_perimeter(self):
		"""
		Compute the anisotropic perimeter of the set.
	
		Returns
		-------
		float
			The anisotropic perimeter, computed as the sum of L1-norms of boundary edges.
		"""
		perimeter = 0.0

		for i in range(self.num_boundary_vertices):
			# Betrachte die Kante zwischen zwei aufeinanderfolgenden Randpunkte
			v1 = self.boundary_vertices[i]
			v2 = self.boundary_vertices[(i + 1) % self.num_boundary_vertices]

			# L1-Norm der Differenz = Summe der absoluten Differenzen in x- und y-Richtung
			perimeter += np.abs(v1[0] - v2[0]) + np.abs(v1[1] - v2[1])

		return perimeter
	
	
	def compute_perimeter_rec_gradient(self):
		"""
		Compute the "gradient" of the perimeter

		Returns
		-------
		array, shape (N, 2)
			Each row contains the two coordinates of the translation to apply at each boundary vertex

		Notes
		-----
		See [1]_ (first variation of the perimeter)

		References
		----------
		.. [1] Maggi, F. (2012). Sets of finite perimeter and geometric variational problems: an introduction to
			   Geometric Measure Theory (No. 135). Cambridge University Press.

		"""
		gradient = np.zeros_like(self.boundary_vertices)

		for i in range(self.num_boundary_vertices):
			e1 = self.boundary_vertices[(i-1) % self.num_boundary_vertices] - self.boundary_vertices[i]
			e2 = self.boundary_vertices[(i+1) % self.num_boundary_vertices] - self.boundary_vertices[i]

			# the i-th component of the gradient is -(ti_1 + ti_2) where ti_1 and ti_2 are the two tangent vectors
			# going away from the i-th vertex TODO: clarify
			gradient[i] = - (e1 / np.abs(e1).sum() + e2 / np.abs(e2).sum())

		return gradient
	

	def compute_anisotropic_perimeter_gradient(self): 
		"""
   		Compute the anisotropic perimeter gradient (L1-norm-based)

		Returns
		-------
		array, shape (N, 2)
			Each row contains the two coordinates of the translation to apply at each boundary vertex
		"""
		gradient = np.zeros_like(self.boundary_vertices)

		for i in range(self.num_boundary_vertices):
			e1 = self.boundary_vertices[(i-1) % self.num_boundary_vertices] - self.boundary_vertices[i]
			e2 = self.boundary_vertices[(i+1) % self.num_boundary_vertices] - self.boundary_vertices[i]

			# Gewichtung mit der L1-Norm statt der L2-Norm
			weight_e1 = np.abs(e1[0]) + np.abs(e1[1])
			weight_e2 = np.abs(e2[0]) + np.abs(e2[1])

			gradient[i] = -(e1 / weight_e1 + e2 / weight_e2)

		return gradient
	
	def compute_anisotropic_perimeter_gradient_rectangular(self):
		''' gradient für die 4 kanten des rechteck.. sortiert nach x_min, x_max, y_min, y_max'''
		gradient = np.array([-2, 2, -2, 2])
		return gradient



	#def compute_anisotropic_perimeter_gradient(self):
		"""
			Compute the gradient of the anisotropic perimeter.
		The anisotropic perimeter considers only horizontal and vertical projections.
	
		Returns
		-------
		array, shape (N, 2)
			Each row contains the two coordinates of the translation to apply at each boundary vertex.
		"""
		gradient = np.zeros_like(self.boundary_vertices)

		for i in range(self.num_boundary_vertices):
			# Nachbarn der aktuellen Ecke bestimmen
			prev_vertex = self.boundary_vertices[(i - 1) % self.num_boundary_vertices]
			next_vertex = self.boundary_vertices[(i + 1) % self.num_boundary_vertices]
			current_vertex = self.boundary_vertices[i]

			# Kantenrichtungen berechnen
			edge1 = prev_vertex - current_vertex
			edge2 = next_vertex - current_vertex


			
			# Gradient bestimmen (nur orthogonale Richtungen zählen)
			grad_x = 0
			grad_y = 0

			# Prüfe, ob die Kante horizontal oder vertikal ist
			if np.abs(edge1[0]) > np.abs(edge1[1]):  # Fast horizontale Kante
				grad_y += np.sign(edge1[1])
			else:  # Fast vertikale Kante
				grad_x += np.sign(edge1[0])

			if np.abs(edge2[0]) > np.abs(edge2[1]):  # Fast horizontale Kante
				grad_y += np.sign(edge2[1])
			else:  # Fast vertikale Kante
				grad_x += np.sign(edge2[0])

			# Setze den Gradient für diesen Punkt
			gradient[i] = np.array([grad_x, grad_y])

		return -gradient  # Minuszeichen, da wir minimieren wollen
	


	#def compute_anisotropic_perimeter_gradient(self):
		gradient_left = np.zeros_like(self.boundary_vertices)
		gradient_right = np.zeros_like(self.boundary_vertices)

		x_min, y_min = np.min(self.boundary_vertices, axis=0)
		x_max, y_max = np.max(self.boundary_vertices, axis=0)

		x_min_mask = self.boundary_vertices[:, 0] == x_min
		x_max_mask = self.boundary_vertices[:, 0] == x_max
		y_min_mask = self.boundary_vertices[:, 1] == y_min
		y_max_mask = self.boundary_vertices[:, 1] == y_max

		num_x_min = np.sum(x_min_mask)
		num_x_max = np.sum(x_max_mask)
		num_y_min = np.sum(y_min_mask)
		num_y_max = np.sum(y_max_mask)

		print("num minimal x:", num_x_min, "num maximal x:", num_x_max, "num minimal y:", num_y_min, "num maximal y:", num_y_max)

		if num_x_min == 1:
			gradient_left[x_min_mask, 0] = -1
			gradient_right[x_min_mask, 0] = -1
		else:
			gradient_left[x_min_mask, 0] = -1 # Nach links verschieben → Perimeter steigt
			gradient_right[x_min_mask, 0] = 0  # Nach rechts verschieben -> Perimeter unverändert


		if num_x_max == 1:
			gradient_left[x_max_mask, 0] = 1
			gradient_right[x_max_mask, 0] = 1
		else:
			gradient_left[x_max_mask, 0] = 0  # Nach links verschieben → Perimeter unverändert
			gradient_right[x_max_mask, 0] = 1 # Nach rechts verschieben -> Perimeter steigt


		if num_y_min == 1:
			gradient_left[y_min_mask, 1] = -1
			gradient_right[y_min_mask, 1] = -1
		else:
			gradient_left[y_min_mask, 1] = -1  # Nach unten verschieben → Perimeter steigt
			gradient_right[y_min_mask, 1] = 0  # Nach oben verschieben -> Perimeter unverändert
	

		if num_y_max == 1:
			gradient_left[y_max_mask, 1] = 1
			gradient_right[y_max_mask, 1] = 1
		else:
			gradient_left[y_max_mask, 1] = 0  # Nach unten verschieben → Perimeter unverändert
			gradient_right[y_max_mask, 1] = 1  # Nach oben verschieben -> Perimeter steigt

	
		return 2*gradient_left, 2*gradient_right
	
	#def compute_anisotropic_perimeter_gradient(self):
		x_min, y_min = np.min(self.boundary_vertices, axis=0)
		x_max, y_max = np.max(self.boundary_vertices, axis=0)
		
		gradient = np.zeros_like(self.boundary_vertices)
		
		x_min_mask = self.boundary_vertices[:, 0] == x_min
		x_max_mask = self.boundary_vertices[:, 0] == x_max
		y_min_mask = self.boundary_vertices[:, 1] == y_min
		y_max_mask = self.boundary_vertices[:, 1] == y_max

		count_x_min = np.sum(x_min_mask)
		count_x_max = np.sum(x_max_mask)
		count_y_min = np.sum(y_min_mask)
		count_y_max = np.sum(y_max_mask)

		# Set gradient in x-direction
		gradient[x_min_mask, 0] = -1 if count_x_min == 1 else 0
		gradient[x_max_mask, 0] = 1  # Immer 1, weil Änderung x_max beeinflusst

		# Set gradient in y-direction
		gradient[y_min_mask, 1] = -1 if count_y_min == 1 else 0
		gradient[y_max_mask, 1] = 1  # Immer 1, weil Änderung y_max beeinflusst

		return 2* gradient

	def compute_weighted_area_rec_tab(self, fourier, boundary_faces_only=False):
		"""
		Compute the integral of f on each face of the inner mesh

		Parameters
		----------
		f : function
			Function to be integrated. f must handle array inputs with shape (N, 2). It can be vector valued
		boundary_faces_only : bool
			Whether to compute weighted areas only on boundary faces, defaut False

		Returns
		-------
		array, shape (N,) or (N,D)
			Value computed for the integral of f on each of the N triangles (if f takes values in dimension D, the shape
			of the resulting array is (N, D))

		"""
		if boundary_faces_only:
			triangles = self.mesh_vertices[self.mesh_boundary_faces]
		else:
			triangles = self.mesh_vertices[self.mesh_faces]

		return fourier.integrate_on_triangles(triangles)
	



	def compute_weighted_area_rec(self, fourier):
		# TODO: decide whether output type instability should be dealt with or not
		"""
		Compute the integral of f over the set

		Parameters
		----------
		f : function
			Function to be integrated. f must handle array inputs with shape (N, 2). It can be vector valued

		Returns
		-------
		float or array of shape (D,)
			Value computed for the integral of f over the set (if f takes values in dimension D, the result will be an
			array of shape (D,))
		"""
		return np.sum(self.compute_weighted_area_rec_tab(fourier))


	def compute_weighted_area_rec_gradient(self, fourier, weights=None):
		"""
		Compute the "gradient" of the weighted area, for a given weight function

		Parameters
		----------
		f : function
			Function to be integrated. f must handle array inputs with shape (N, 2). It can be vector valued

		Returns
		-------
		array, shape

		Notes
		-----
		Vectorized computations are really nasty here, mainly because f can be vector valued.

		"""
		# rotation matrix used to compute outward normals
		rot = np.array([[0, -1], [1, 0]]) if self.is_clockwise else np.array([[0, 1], [-1, 0]])

		rolled_vertices1 = np.roll(self.boundary_vertices, 1, axis=0)
		rolled_vertices2 = np.roll(self.boundary_vertices, -1, axis=0)

		if weights is None: # das sollte besser nicht eintreten, da imtegrate_on_polygonal_curve aus der klasse GaussianPolynomial aus cheeger und nicht aus SampledGaussianFilter kommt. Müsste sonst nochmal anpassen
			weights = fourier.integrate_on_polygonal_curve(self.boundary_vertices)

		normals1 = np.dot(self.boundary_vertices - rolled_vertices1, rot.T)
		normals1 /= np.linalg.norm(normals1, axis=-1)[:, None]
		normals2 = np.dot(rolled_vertices2 - self.boundary_vertices, rot.T)
		normals2 /= np.linalg.norm(normals2, axis=-1)[:, None]

		if weights.ndim == 2:
			gradient = weights[:, 0, None] * normals1 + weights[:, 1, None] * normals2
		else:
			gradient = weights[:, :, 0, None] * normals1[:, None, :] + weights[:, :, 1, None] * normals2[:, None, :]

		
		return gradient
	
	def compute_weighted_area_gradient_rectangular(self, fourier):
		"""
		Berechnet den Gradienten der gewichteten Fläche für ein Rechteck.

		Returns:
		--------
		gradient: np.array mit Länge 4 (Gradient für x_min, x_max, y_min, y_max)
		"""
		x_min, x_max = np.min(self.boundary_vertices[:, 0]), np.max(self.boundary_vertices[:, 0])
		y_min, y_max = np.min(self.boundary_vertices[:, 1]), np.max(self.boundary_vertices[:, 1])

		# Integriere entlang der vertikalen Kanten (links x_min, rechts x_max)
		integral_x_min = fourier.integrate_on_single_line(np.array([ self.boundary_vertices[0], self.boundary_vertices[3]]))
		
		integral_x_max = fourier.integrate_on_single_line(np.array([ self.boundary_vertices[1], self.boundary_vertices[2]]))

		# Integriere entlang der horizontalen Kanten (unten y_min, oben y_max)
		integral_y_min = fourier.integrate_on_single_line(np.array([ self.boundary_vertices[0], self.boundary_vertices[1]]))
		integral_y_max = fourier.integrate_on_single_line(np.array([ self.boundary_vertices[3], self.boundary_vertices[2]]))

		# Setze die Gradienten mit den Vorzeichen aus der Ableitung
		return np.array([-integral_x_min, integral_x_max, -integral_y_min, integral_y_max])
	
	
	
	
	def numerical_gradient_check(self, fourier, epsilon = 1e-6):
		"""
		Vergleicht den analytischen Gradienten mit einer numerischen Approximation.

		Parameters
		----------
		instance : Deine Klasse mit boundary_vertices und compute_weighted_area_rec_gradient
		fourier : Fourier-Objekt zur Berechnung von Integralen
		epsilon : Kleine Verschiebung für die numerische Ableitung

		Returns
		------
		numerical_gradient : Numerischer Gradient an den Randpunkten
		analytical_gradient : Vom Code berechneter Gradient
		"""
		boundary_vertices = self.boundary_vertices.copy()
		analytical_gradient = self.compute_weighted_area_rec_gradient(fourier)

		numerical_gradient = np.zeros_like(boundary_vertices)

		for i in range(len(boundary_vertices)):
			for d in range(2):  # x- und y-Richtung
				# Verschiebung des Punktes in Normalenrichtung
				boundary_vertices[i, d] += epsilon
				self.boundary_vertices = boundary_vertices
				I_plus = fourier.integrate_on_polygonal_curve(self.boundary_vertices).sum()
			
				boundary_vertices[i, d] -= 2 * epsilon
				self.boundary_vertices = boundary_vertices
				I_minus = fourier.integrate_on_polygonal_curve(self.boundary_vertices).sum()

				# Rücksetzen des Punktes
				boundary_vertices[i, d] += epsilon

				# Finite Differenz
				numerical_gradient[i, d] = (I_plus - I_minus) / (2 * epsilon)

		# Setze die boundary_vertices zurück
		self.boundary_vertices = boundary_vertices.copy()

		return numerical_gradient, analytical_gradient

	def compute_mesh_faces_orientation(self):
		faces = self.mesh_vertices[self.mesh_faces]
		diff1 = faces[:, 1] - faces[:, 0]
		diff2 = faces[:, 2] - faces[:, 1]
		res = np.sign(np.cross(diff1, diff2)).astype('int')

		return res
	
	
	
	def create_mesh(self, boundary_vertices, max_tri_area):
		"""
		Create the inner mesh of the set

		Additional: also outer of the set (not only inner)
		
		Parameters
		----------
		boundary_vertices : array, shape (N, 2)
			Each row contains the two coordinates of a boundary vertex
		max_tri_area : float
			Maximum triangle area for the inner mesh

		"""
		mesh = triangulate(boundary_vertices, max_triangle_area=max_tri_area)

		self.mesh_vertices = mesh['vertices']
		self.mesh_faces = mesh['triangles']

		# TODO: comment
		orientations = self.compute_mesh_faces_orientation()
		indices = np.where(orientations < 0)[0]
		for i in range(len(indices)):
			index = indices[i]
			tmp_face = self.mesh_faces[index].copy()
			self.mesh_faces[index, 1] = tmp_face[index, 2]
			self.mesh_faces[index, 2] = tmp_face[index, 1]

		assert np.alltrue(orientations > 0)

		boundary_faces_indices = []

		for i in range(len(self.mesh_faces)):
			# find the faces which have at least one vertex among the boundary vertices (the indices of boundary
			# vertices in self.vertices are 0,1,...,self.num_boundary_vertices-1)
			if len(np.intersect1d(np.arange(self.num_boundary_vertices), self.mesh_faces[i])) > 0:
				boundary_faces_indices.append(i)

		self.mesh_boundary_faces_indices = np.array(boundary_faces_indices)

		


	def create_whole_mesh(self, boundary_vertices, domain_vertices, max_tri_area):
		"""
		Create the inner mesh of the set

		Additional: also outer of the set (not only inner)
		
		Parameters
		----------
		boundary_vertices : array, shape (N, 2)
			Each row contains the two coordinates of a boundary vertex
		max_tri_area : float
			Maximum triangle area for the inner mesh

		"""
		mesh = triangulate_combined(boundary_vertices, domain_vertices, max_triangle_area=max_tri_area)

		self.mesh_vertices = mesh['vertices']
		self.mesh_faces = mesh['triangles']

		# TODO: comment
		orientations = self.compute_mesh_faces_orientation()
		indices = np.where(orientations < 0)[0]
		for i in range(len(indices)):
			index = indices[i]
			tmp_face = self.mesh_faces[index].copy()
			self.mesh_faces[index, 1] = tmp_face[index, 2]
			self.mesh_faces[index, 2] = tmp_face[index, 1]

		assert np.alltrue(orientations > 0)

		boundary_faces_indices = []

		for i in range(len(self.mesh_faces)):
			# find the faces which have at least one vertex among the boundary vertices (the indices of boundary
			# vertices in self.vertices are 0,1,...,self.num_boundary_vertices-1)
			if len(np.intersect1d(np.arange(self.num_boundary_vertices), self.mesh_faces[i])) > 0:
				boundary_faces_indices.append(i)

		self.mesh_boundary_faces_indices = np.array(boundary_faces_indices)	

		

	def compute_objective(self, fourier ):
		res = self.compute_anisotropic_perimeter() /  np.abs(self.compute_weighted_area_rec(fourier) )
		return res
	
	def objective_wrapper(self, x, fourier):
		#x ist hier boundary vertices flatten
		"""Setzt boundary_vertices neu und berechnet das Objektiv"""
		
		x_min, y_min, x_max, y_max = x  # Extrahiere die Variablen
		
		
		
		
		#self.boundary_vertices = x.reshape((-1, 2))
		return self.compute_objective(fourier)