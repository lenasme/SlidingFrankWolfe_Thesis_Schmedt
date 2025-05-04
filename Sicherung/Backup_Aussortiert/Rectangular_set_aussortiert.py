def compute_weighted_area_rec(self, f, num_points=100):
		x_values = [v[0] for v in self.boundary_vertices]
		y_values = [v[1] for v in self.boundary_vertices]

		#min_x = 1-max(x_values)
		#max_x = 1-min(x_values)

		min_x = min(x_values)
		max_x = max(x_values)
		min_y = min(y_values)
		max_y = max(y_values) 

		x = np.linspace(min_x, max_x, num_points)
		y = np.linspace(min_y, max_y, num_points)
		dx = (max_x - min_x) / (num_points - 1)
		dy = (max_y - min_y) / (num_points - 1)

		#X, Y = np.meshgrid(x, y)

		#Z = f(X, Y)
		Z = f(x[:, None], y[None, :])
		#print("Z shape:", Z.shape)
		#Z = np.array([[f(xi, yi) for yi in y] for xi in x])
		# Approximation des Integrals
		integral = np.sum(Z) * dx * dy
		return integral