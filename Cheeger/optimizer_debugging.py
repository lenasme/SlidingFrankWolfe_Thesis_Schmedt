import numpy as np
import matplotlib.pyplot as plt

from .tools import resample
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
        #perimeter_gradient = self.set.compute_anisotropic_perimeter_gradient()
        #print("anisotropic perimeter gradients", perimeter_gradient)
        left_perimeter_gradient, right_perimeter_gradient = self.set.compute_anisotropic_perimeter_gradient()
        #print("left anisotropic perimeter gradients", left_perimeter_gradient, "right anisotropic perimeter gradient:", right_perimeter_gradient)
        mean_perimeter_gradient = (left_perimeter_gradient + right_perimeter_gradient) / 2
        print("mean perimeter gradient:", mean_perimeter_gradient)
        area_gradient = self.set.compute_weighted_area_rec_gradient(f)
        gradient = (mean_perimeter_gradient * self.weighted_area - area_gradient * self.perimeter) / self.weighted_area ** 2

        print("Fläche:", self.weighted_area)
        print("Perimeter:", self.perimeter)
        print("Norm Perimetergradient:", np.linalg.norm(mean_perimeter_gradient))
        print("Norm Flächengradient:", np.linalg.norm(area_gradient))
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


            gradient_field = np.zeros((self.state.grid_size, self.state.grid_size))
            print(gradient_field.shape)
            grad_norm = np.linalg.norm(gradient, axis=1)
            for i, (x, y) in enumerate(self.state.set.boundary_vertices):
                gradient_field[int(y), int(x)] = grad_norm[i]

            #gradient_magnitude = np.linalg.norm(gradient, axis=-1)

            # Visualisierung
            plt.figure(figsize=(8, 6))
            plt.imshow(gradient_field, cmap = 'viridis', origin = 'lower')
            plt.colorbar(label="Gradient Magnitude")
            plt.title("Gradientenbetrag des Funktionals")
            plt.xlabel("x")
            plt.ylabel("y")
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
