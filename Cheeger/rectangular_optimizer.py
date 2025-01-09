import numpy as np

from .simple_set import SimpleSet

def objective(boundary_vertices, eta):
  rect_set = SimpleSet(boundary_vertices)
  x_min = boundary_vertices[0][0]
  x_max = boundary_vertices[2][0]
  y_min = boundary_vertices[0][1]
  y_max = boundary_vertices[2][1]
  if x_max <= x_min or y_max <= y_min:
    return np.inf
  res = rect_set.compute_perimeter() /  np.abs(rect_set.compute_weighted_area(eta) )
  return res
