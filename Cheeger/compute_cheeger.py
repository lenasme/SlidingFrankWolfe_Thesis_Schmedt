import numpy as np
import time

from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
#from .simple_set import SimpleSet
#from .optimizer_debugging import CheegerOptimizer
#from .tools import run_primal_dual, extract_contour, resample
#from .plot_utils import plot_primal_dual_results, plot_simple_set, plot_rectangular_set
#from .rectangular_optimizer import objective
from .rectangular_set import RectangularSet, construct_rectangular_set_from01, evaluate_inverse_fourier
from Setup.ground_truth import GroundTruth, construction_of_example_source
from .tools import run_primal_dual, extract_contour
from .plot_utils import plot_primal_dual_results
from .optimizer_debugging import run_fine_optimization





def compute_cheeger_set(grid_size, deltas, max_jumps, grid_size_coarse, cut_off, max_iter_primal_dual = 10000, plot=True):
    ground_truth = construction_of_example_source(grid_size, deltas, max_jumps)

   

    operator_applied_on_ground_truth = np.fft.fft2(ground_truth)

    freqs_x= np.fft.fftfreq(grid_size, d=1 / grid_size)
    freqs_y = np.fft.fftfreq(grid_size, d=1 / grid_size)
    freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")
    

    
    mask = np.zeros((grid_size, grid_size))
    mask[(np.abs(freq_x) <= cut_off) & (np.abs(freq_y) <= cut_off)] = 1

    truncated_operator_applied_on_ground_truth = operator_applied_on_ground_truth * mask

    if plot == True:
        

        plt.plot()
        plt.imshow(ground_truth, cmap = 'bwr')
        plt.colorbar()
        plt.title("Ground Truth")
        plt.show()

        plt.plot()
        plt.imshow(truncated_operator_applied_on_ground_truth.real, cmap= 'bwr')
        plt.colorbar()
        plt.title("Truncated Fourier Frequency Image")
        plt.show()

        
        plt.plot()
        plt.imshow(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, cmap = 'bwr')
        plt.colorbar()
        plt.title("Truncated Fouried Applied on Ground Truth")
        plt.show()

    h = grid_size / grid_size_coarse
    eta_bar = np.zeros((grid_size_coarse, grid_size_coarse))
    for i in range(grid_size_coarse):
        for j in range(grid_size_coarse):
            x_min = 0 + i * h
            x_max = (i+1) * h
            y_min = j * h
            y_max = (j+1) * h
            rectangle_coarse_grid = RectangularSet(x_min, x_max, y_min, y_max)

            eta_bar[i,j] = (rectangle_coarse_grid.compute_integral(cut_off, truncated_operator_applied_on_ground_truth, grid_size) / h**2).real


    u = run_primal_dual(grid_size_coarse, eta_bar, max_iter=max_iter_primal_dual, convergence_tol=None, plot=True)

    if plot == True:
        plot_primal_dual_results(u[1:-1, 1:-1], eta_bar)

    boundary_vertices = extract_contour(u)
    

    
    initial_rectangular_set = construct_rectangular_set_from01(boundary_vertices, grid_size)

    if plot == True:
        
        initial_rectangular_set.plot_rectangular_set(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, grid_size)

    x_min, x_max, y_min, y_max = initial_rectangular_set.coordinates[0], initial_rectangular_set.coordinates[1], initial_rectangular_set.coordinates[2], initial_rectangular_set.coordinates[3]

    
    weights = truncated_operator_applied_on_ground_truth
    

   

    

    optimal_rectangle,  objective_tab, gradient_tab , x_mins, x_maxs, y_mins, y_maxs =  run_fine_optimization(initial_rectangular_set, cut_off, weights, grid_size )

    if plot == True:
        optimal_rectangle.plot_rectangular_set(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, grid_size)
        print(f"boundary coordinates: {initial_rectangular_set.coordinates}")
        print(f"Optimales Rechteck: {optimal_rectangle.coordinates}")
        print(f"Verschiebung: {optimal_rectangle.coordinates - initial_rectangular_set.coordinates}")

        fig, ax = plt.subplots()
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        rect = patches.Rectangle((0, 0), 1, 1, edgecolor='r', facecolor='none', linewidth=2)
        ax.add_patch(rect)

        def update(frame):
            rect.set_xy((x_mins[frame], y_mins[frame]))
            rect.set_width(x_maxs[frame] - x_mins[frame])
            rect.set_height(y_maxs[frame] - y_mins[frame])

        ani = animation.FuncAnimation(fig, update, frames=len(x_mins), interval=200)
        ani.save("animation.gif", writer="pillow")
        plt.show()



        plt.plot(objective_tab)
        plt.title("Objective")
        plt.show()

        plt.plot(gradient_tab)
        plt.title("Gradient")
        plt.show()


