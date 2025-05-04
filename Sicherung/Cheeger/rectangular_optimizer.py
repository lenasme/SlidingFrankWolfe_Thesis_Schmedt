import numpy as np

from .rectangular_set import RectangularSet

def objective(boundary_vertices, eta):
  x_min = boundary_vertices[0]
  x_max = boundary_vertices[1]
  y_min = boundary_vertices[2]
  y_max = boundary_vertices[3]
  if x_max <= x_min or y_max <= y_min:
    #print("die reihenfolge der vertices passt nicht!")
    #return np.inf
    penalty = 1e6 + (x_min - x_max) ** 2 + (y_min - y_max) ** 2
    return penalty
  rectangle_boundary_vertices= np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])
  rect_set = RectangularSet(rectangle_boundary_vertices)
  res = rect_set.compute_perimeter_rec() /  np.abs(rect_set.compute_weighted_area_rec(eta) )
  return res
