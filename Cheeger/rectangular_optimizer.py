import numpy as np

from .simple_set import SimpleSet

def objective(boundary_vertices, eta):
  x_min = boundary_vertices[0]
  x_max = boundary_vertices[1]
  y_min = boundary_vertices[2]
  y_max = boundary_vertices[3]
  if x_max <= x_min or y_max <= y_min:
    return np.inf
  rectangle_boundary_vertices= np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])
  rect_set = SimpleSet(rectangle_boundary_vertices)
  res = rect_set.compute_perimeter() /  np.abs(rect_set.compute_weighted_area(eta) )
  return res
