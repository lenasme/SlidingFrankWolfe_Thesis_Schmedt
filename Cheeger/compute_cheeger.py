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

print(animation.writers.list())



def compute_cheeger_set(grid_size, deltas, max_jumps, grid_size_coarse, cut_off, max_iter_primal_dual = 10000, plot=True):
    ground_truth = construction_of_example_source(grid_size, deltas, max_jumps)

    #ground_truth = np.zeros((grid_size, grid_size))
    #x_min, x_max = 0,10
    #y_min, y_max = 20,60

    #ground_truth[x_min:x_max, y_min:y_max] = 1

    operator_applied_on_ground_truth = np.fft.fft2(ground_truth)

    freqs_x= np.fft.fftfreq(grid_size, d=1 / grid_size)
    freqs_y = np.fft.fftfreq(grid_size, d=1 / grid_size)
    freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")
    

    
    mask = np.zeros((grid_size, grid_size))
    mask[(np.abs(freq_x) <= cut_off) & (np.abs(freq_y) <= cut_off)] = 1

    truncated_operator_applied_on_ground_truth = operator_applied_on_ground_truth * mask

    if plot == True:
        

        plt.plot()
        plt.imshow(truncated_operator_applied_on_ground_truth.real, cmap= 'bwr')
        plt.colorbar()
        plt.title("Truncated Fourier Image")
        plt.show()

        
        plt.subplot(1,2,1)
        plt.imshow(np.fft.ifft2(operator_applied_on_ground_truth).real, cmap = 'bwr')
        plt.subplot(1,2,2)
        plt.imshow(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, cmap = 'bwr')
        plt.colorbar()
        plt.title("Komntrolle, ob transformation passt")       
        plt.show()

        plt.subplot(1,2,1)
        plt.imshow(np.fft.ifft2(operator_applied_on_ground_truth).imag, cmap = 'bwr')
        plt.subplot(1,2,2)
        plt.imshow(np.fft.ifft2(truncated_operator_applied_on_ground_truth).imag, cmap = 'bwr')
        plt.colorbar()
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

            eta_bar[i,j] = rectangle_coarse_grid.compute_integral(cut_off, truncated_operator_applied_on_ground_truth, grid_size) / h**2


    if plot == True:
        plt.plot()
        plt.imshow(eta_bar.real, cmap= 'bwr')
        plt.colorbar()
        plt.show()

    u = run_primal_dual(grid_size_coarse, eta_bar, max_iter=max_iter_primal_dual, convergence_tol=None, plot=True)

    if plot == True:
        plot_primal_dual_results(u[1:-1, 1:-1], eta_bar)

    boundary_vertices = extract_contour(u)
    

    

    initial_rectangular_set = construct_rectangular_set_from01(boundary_vertices, grid_size)

    if plot == True:
        print(f"boundary coordinates: {initial_rectangular_set.coordinates}")
        initial_rectangular_set.plot_rectangular_set(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, grid_size)

    x_min, x_max, y_min, y_max = initial_rectangular_set.coordinates[0], initial_rectangular_set.coordinates[1], initial_rectangular_set.coordinates[2], initial_rectangular_set.coordinates[3]

    #modified_rectangle = RectangularSet(  x_min ,  x_max, y_min,  y_max)

    #if plot == True:
        #modified_rectangle.plot_rectangular_set(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, grid_size)

    weights = truncated_operator_applied_on_ground_truth
    #test_x = np.array(initial_rectangular_set.coordinates, dtype=np.float64, order='F')
    #print("objective:",initial_rectangular_set.compute_objective_wrapper(test_x, cut_off, weights, grid_size))
    #print("x_min",initial_rectangular_set.x_min)
    #print("x_max",initial_rectangular_set.x_max)
    #print("y_min",initial_rectangular_set.y_min)
    #print("y_max",initial_rectangular_set.y_max)

    #print("perimeter:", initial_rectangular_set.compute_anisotropic_perimeter())
    #print("integral:", initial_rectangular_set.compute_integral(cut_off, weights, grid_size))
    #print("gradient:", initial_rectangular_set.objective_gradient_wrapper(test_x, cut_off, weights, grid_size))

    #optimal_rectangle, objective_tab, gradient_tab =  run_fine_optimization(initial_rectangular_set, cut_off, weights, grid_size )


    if plot == True:

        plt.subplot(1, 2, 1)
        plt.imshow(np.log(1 + np.abs(weights)), cmap='viridis')
        plt.colorbar()
        plt.title("Fourier-Koeffizienten (Magnitude)")

        plt.subplot(1, 2, 2)
        plt.imshow(np.angle(weights), cmap='twilight')
        plt.colorbar()
        plt.title("Fourier-Koeffizienten (Phase)")

        plt.show()

        plt.imshow(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, cmap='bwr')
        plt.colorbar()
        plt.title("Ground Truth")
        plt.show()

        

                


   
    
    
  
    print("x_min",initial_rectangular_set.x_min)
    print("x_max",initial_rectangular_set.x_max)
    print("y_min",initial_rectangular_set.y_min)
    print("y_max",initial_rectangular_set.y_max)

    print("perimeter:", initial_rectangular_set.compute_anisotropic_perimeter())
    print("integral:", initial_rectangular_set.compute_integral(cut_off, weights, grid_size))
    #print(f"Numerisches Integral: {integral_numerisch}")
    
    #print("gradient:", modified_rectangle.objective_gradient_wrapper(test_x, cut_off, weights, grid_size))


    import pandas as pd

    def generate_fourier_evaluation_table(cut_off, weights, grid_size, num_points=20):
        x_vals = np.linspace(0, grid_size, num_points)  # Gitterpunkte in x-Richtung
        y_vals = np.linspace(0, grid_size, num_points)  # Gitterpunkte in y-Richtung

        data = []  # Liste für die Ergebnisse

        for i in range(num_points):
            for j in range(num_points):
                x, y = x_vals[i], y_vals[j]
                val = evaluate_inverse_fourier([x, y], cut_off, weights, grid_size)
                data.append([x, y, val.real])  # Speichere x, y und den Realteil

        # Erstelle eine Pandas-Tabelle
        df = pd.DataFrame(data, columns=["X", "Y", "Value"])
        return df

    # Beispiel-Aufruf:
    #df = generate_fourier_evaluation_table(cut_off, weights, grid_size)
    #print(df)
    #df.to_csv("fourier_values.csv", index=False)

    optimal_rectangle_grad,  objective_tab, gradient_tab , x_mins, x_maxs, y_mins, y_maxs =  run_fine_optimization(initial_rectangular_set, cut_off, weights, grid_size )

    if plot == True:
        optimal_rectangle_grad.plot_rectangular_set(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, grid_size)

        print(f"Optimales Rechteck: {optimal_rectangle_grad.coordinates}")

        #optimal_rectangle_without_grad.plot_rectangular_set(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, grid_size)

        #print(f"Optimales Rechteck ohne Gradienten: {optimal_rectangle_without_grad.coordinates}")

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


