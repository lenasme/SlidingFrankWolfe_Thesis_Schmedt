import numpy as np
from scipy.spatial import Delaunay
from .tools import winding, triangulate

class RectangularSet:

	def __init__(self, boundary_vertices, domain_vertices = np.array([[0,0], [0,1], [1,1], [1,0]]), max_tri_area=None):
		self.num_boundary_vertices = len(boundary_vertices)

		# the curve is clockwise if and only if the sum over the edges of (x2-x1)(y2+y1) is positive
		rolled_boundary_vertices = np.roll(boundary_vertices, -1, axis=0)
		self.is_clockwise = (np.sum((rolled_boundary_vertices[:, 0] - boundary_vertices[:, 0]) *
									(rolled_boundary_vertices[:, 1] + boundary_vertices[:, 1])) > 0)
		self.mesh_vertices = None
		self.mesh_faces = None
		self.mesh_boundary_faces_indices = None
		# creation of the inner mesh
		
		self.create_whole_mesh(boundary_vertices, domain_vertices, max_tri_area)

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
		combined_vertices = np.vstack([boundary_vertices, domain_vertices])
		mesh = Delaunay(combined_vertices)

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