import numpy as np
import cvxpy as cp
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

from SlidingFrankWolfe.simple_function import IndicatorFunction, SimpleFunction



def calculate_target_function(grid_size, deltas, max_jumps, cut_off, noise_level = 0.01, plot = True):
    ground_truth = construction_of_example_source(grid_size, deltas, max_jumps)

    operator_applied_on_ground_truth = np.fft.fft2(ground_truth)

    freqs_x= np.fft.fftfreq(grid_size, d=1 / grid_size)
    freqs_y = np.fft.fftfreq(grid_size, d=1 / grid_size)
    freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")
    

    
    mask = np.zeros((grid_size, grid_size))
    mask[(np.abs(freq_x) <= cut_off) & (np.abs(freq_y) <= cut_off)] = 1

    truncated_operator_applied_on_ground_truth = operator_applied_on_ground_truth * mask

      
    noise = noise_level * (np.random.randn(grid_size, grid_size) + 1j * np.random.randn(grid_size, grid_size))

    target_function_f = truncated_operator_applied_on_ground_truth + noise

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


    return truncated_operator_applied_on_ground_truth, ground_truth, target_function_f


def compute_cheeger_set(truncated_operator_applied_on_ground_truth, grid_size, grid_size_coarse, cut_off, max_iter_primal_dual = 10000, plot=True):
   
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
    initial_coordinates = initial_rectangular_set.coordinates
    if plot == True:
        
        initial_rectangular_set.plot_rectangular_set(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, grid_size)

    x_min, x_max, y_min, y_max = initial_rectangular_set.coordinates[0], initial_rectangular_set.coordinates[1], initial_rectangular_set.coordinates[2], initial_rectangular_set.coordinates[3]

    
    weights = truncated_operator_applied_on_ground_truth
    
    optimal_rectangle,  objective_tab, gradient_tab , x_mins, x_maxs, y_mins, y_maxs =  run_fine_optimization(initial_rectangular_set, cut_off, weights, grid_size )

    if plot == True:
        optimal_rectangle.plot_rectangular_set(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, grid_size)
        print(f"initiale Koordinaten: {initial_coordinates}")
        print(f"optimale Koordinaten: {optimal_rectangle.coordinates}")
        print(f"Verschiebung: {optimal_rectangle.coordinates - initial_coordinates}")

        fig, ax = plt.subplots()
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        rect = patches.Rectangle((0, 0), 1, 1, edgecolor='r', facecolor='none', linewidth=2)
        ax.add_patch(rect)

        def update(frame):
            rect.set_xy((y_mins[frame], x_mins[frame]))
            rect.set_height(x_maxs[frame] - x_mins[frame])
            rect.set_width(y_maxs[frame] - y_mins[frame])

        ani = animation.FuncAnimation(fig, update, frames=len(x_mins), interval=200)
        ani.save("animation.gif", writer="pillow")
        plt.show()



        plt.plot(objective_tab)
        plt.title("Objective")
        plt.show()

        plt.plot(gradient_tab)
        plt.title("Gradient")
        plt.show()

    return optimal_rectangle


def fourier_image_rectangle(rectangular_set, grid_size, cut_off):
    new_indicator_function = IndicatorFunction(rectangular_set, grid_size)
    #image = new_indicator_function.construct_image_matrix(plot=True)

    plt.plot()
    plt.imshow(new_indicator_function.construct_image_matrix(plot=True), cmap = 'bwr')
    plt.colorbar()
    plt.title("Indikatorfunktion")
    plt.show()

    fourier_image = new_indicator_function.compute_truncated_frequency_image(cut_off)

    return fourier_image




def optimization ( target_function_f, grid_size, grid_size_coarse, cut_off, reg_param, max_iter_primal_dual = 10000, plot=True):
    
    atoms = []
    u = SimpleFunction(atoms, grid_size, cut_off)

    #Ku-f:
    weights_in_eta = - u.compute_truncated_frequency_image_sf(cut_off, show = True) + target_function_f

    optimal_rectangle = compute_cheeger_set(weights_in_eta, grid_size, grid_size_coarse, cut_off, max_iter_primal_dual = 10000, plot=True)

    u.extend_support(optimal_rectangle)

    #k(\1_E)
    K_0 = np.zeros((u.num_atoms, grid_size, grid_size), dtype = complex)
    perimeters = np.zeros(u.num_atoms)
    for i in range(u.num_atoms):
        print(f"Koordinate des Atoms {i}: {u.atoms[i].support.coordinates}")
        K_0[i] = fourier_image_rectangle(u.atoms[i].support, grid_size, cut_off)
        perimeters[i] = u.atoms[i].support.compute_anisotropic_perimeter()

    alpha = reg_param 

    print("Perimeter:", perimeters)

    if u.num_atoms == 0:
        print("Fehler: Es wurden keine Atome gefunden!")
        return

    print(f"Anzahl der Atome: {u.num_atoms}")
    print(f"K_0.shape: {K_0.shape}, erwartet: ({u.num_atoms}, {grid_size}, {grid_size})")
    print(f"perimeters.shape: {perimeters.shape}, erwartet: ({u.num_atoms},)")
    print(f"target_function_f.shape: {target_function_f.shape}, erwartet: ({grid_size}, {grid_size})")

    a = cp.Variable(u.num_atoms)

    K0_sum = cp.sum(cp.multiply(a[:, None, None], K_0), axis=0)

    objective = (1/2) * cp.norm(K0_sum - target_function_f, "fro")**2 + alpha * cp.sum(cp.multiply(perimeters, cp.abs(a)))
    
    problem = cp.Problem(cp.Minimize(objective))

    print(f"Ist das Problem DCP-konform? {problem.is_dcp()} (Sollte True sein)")

    try:
        result = problem.solve(solver=cp.SCS, verbose=True)
        a_opt = a.value
        print("Optimale a Werte:", a_opt)
    except Exception as e:
        print("Fehler beim Lösen des Optimierungsproblems:", e)


    for i in range(u.num_atoms):
        u.atoms[i].weight = a_opt[i]

    fourier_image = u.compute_truncated_frequency_image_sf(cut_off, show = True)

    plt.plot()
    plt.imshow(np.fft.ifft2(fourier_image).real, cmap = 'bwr')
    plt.colorbar()
    plt.title("Rekonstruktion")
    plt.show()