
import numpy as np
from .tools import winding, triangulate

class RectangularSet:

	def __init__(self, boundary_vertices, max_tri_area=None):
		self.num_boundary_vertices = len(boundary_vertices)

		# the curve is clockwise if and only if the sum over the edges of (x2-x1)(y2+y1) is positive
		rolled_boundary_vertices = np.roll(boundary_vertices, -1, axis=0)
		self.is_clockwise = (np.sum((rolled_boundary_vertices[:, 0] - boundary_vertices[:, 0]) *
									(rolled_boundary_vertices[:, 1] + boundary_vertices[:, 1])) > 0)
		self.mesh_vertices = None
		self.mesh_faces = None
		self.mesh_boundary_faces_indices = None
		# creation of the inner mesh
		self.create_mesh(boundary_vertices, max_tri_area)

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
	
		
		return res

	def compute_weighted_area_rec(self, f, num_points=100):
		x_values = [v[0] for v in self.boundary_vertices]
		y_values = [v[1] for v in self.boundary_vertices]

		min_x = 1-max(x_values)
		max_x = 1-min(x_values)

		#min_x = min(x_values)
		#max_x = max(x_values)
		min_y = min(y_values)
		max_y = max(y_values) 

		x = np.linspace(min_x, max_x, num_points)
		y = np.linspace(min_y, max_y, num_points)
		dx = (max_x - min_x) / (num_points - 1)
		dy = (max_y - min_y) / (num_points - 1)

		#X, Y = np.meshgrid(x, y)

		#Z = f(X, Y)
		Z = f(x[:, None], y[None, :])
		# Approximation des Integrals
		integral = np.sum(Z) * dx * dy
		return integral

	def compute_mesh_faces_orientation(self):
		faces = self.mesh_vertices[self.mesh_faces]
		diff1 = faces[:, 1] - faces[:, 0]
		diff2 = faces[:, 2] - faces[:, 1]
		res = np.sign(np.cross(diff1, diff2)).astype('int')

		return res
	
	
	
	def create_mesh(self, boundary_vertices, max_tri_area):
		"""
		Create the inner mesh of the set

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







	

	
