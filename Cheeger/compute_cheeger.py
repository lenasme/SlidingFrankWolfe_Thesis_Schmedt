import numpy as np
import time

from scipy.optimize import minimize
import matplotlib.pyplot as plt
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
    #ground_truth = construction_of_example_source(grid_size, deltas, max_jumps)

    ground_truth = np.zeros((grid_size, grid_size))
    x_min, x_max = 0,10
    y_min, y_max = 20,60

    ground_truth[x_min:x_max, y_min:y_max] = 1

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

        

                


    #test_x = np.array(modified_rectangle.coordinates, dtype=np.float64, order='F')
    
    #x_vals = np.linspace(modified_rectangle.x_min, modified_rectangle.x_max, grid_size)
    #y_vals = np.linspace(modified_rectangle.y_min, modified_rectangle.y_max, grid_size)
   # X, Y = np.meshgrid(x_vals, y_vals)

    #integral_numerisch = np.sum(evaluate_inverse_fourier(np.array([X, Y]), cut_off, weights, grid_size)) * (x_vals[1] - x_vals[0]) * (y_vals[1] - y_vals[0])
    
    
    
    
    #print("objective:",modified_rectangle.compute_objective_wrapper(test_x, cut_off, weights, grid_size))
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

    optimal_rectangle_grad,  objective_tab, gradient_tab =  run_fine_optimization(initial_rectangular_set, cut_off, weights, grid_size )

    if plot == True:
        optimal_rectangle_grad.plot_rectangular_set(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, grid_size)

        print(f"Optimales Rechteck: {optimal_rectangle_grad.coordinates}")

        #optimal_rectangle_without_grad.plot_rectangular_set(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, grid_size)

        #print(f"Optimales Rechteck ohne Gradienten: {optimal_rectangle_without_grad.coordinates}")

        plt.plot(objective_tab)
        plt.title("Objective")
        plt.show()

        plt.plot(gradient_tab)
        plt.title("Gradient")
        plt.show()


#def compute_cheeger(eta, grid_size_fm, max_iter_fm=10000, convergence_tol_fm=None, plot_results_fm=False,
                    #num_boundary_vertices_ld=None, point_density_ld=None, max_tri_area_ld=5e-3, step_size_ld=1e-2,
                   # max_iter_ld=500, convergence_tol_ld=1e-4, num_iter_resampling_ld=None, plot_results_ld=False):
    """
    Compute the Cheeger set associated to the weight function eta

    Parameters
    ----------
    eta : function
        Function to be integrated. f must handle array inputs with shape (N, 2)
    max_tri_area_fm : float
        Fixed mesh step parameter. Maximum triangle area allowed for the domain mesh
    max_iter_fm : int
        Fixed mesh step parameter. Maximum number of iterations for the primal dual algorithm
    plot_results_fm : bool
        Fixed mesh step parameter. Whether to plot the results of the fixed mesh step or not
    num_boundary_vertices_ld : int
        Local descent step parameter. Number of boundary vertices used to represent the simple set
    max_tri_area_ld : float
        Local descent step parameter. Maximum triangle area allowed for the inner mesh of the simple set
    step_size_ld : float
        Local descent step parameter. Step size used in the local descent
    max_iter_ld : int
        Local descent step parameter. Maximum number of iterations allowed for the local descent
    convergence_tol_ld : float
        Local descent step parameter. Convergence tol for the local descent
    num_iter_resampling_ld : None or int
        Local descent step parameter. Number of iterations between two resampling of the boundary curve (None for no
        resampling)
    plot_results_ld : bool
        Local descent step parameter. Whether to plot the results of the local descent step or not

    Returns
    -------
    simplet_set : SimpleSet
        Cheeger set
    obj_tab : array, shape (n_iter_ld,)
        Values of the objective over the course of the local descent
    grad_norm_tab : array, shape (n_iter_ld,)
        Values of the objective gradient norm over the course of the local descent

    """
    #assert (num_boundary_vertices_ld is None) or (point_density_ld is None)

    # compute the integral of the weight function on each pixel of the grid
    #eta_bar = eta.integrate_on_pixel_grid(grid_size_fm)
    
    


    # perform the fixed mesh optimization step
   # u = run_primal_dual(grid_size_fm, eta_bar, max_iter=max_iter_fm, convergence_tol=convergence_tol_fm, plot=True)

    #if plot_results_fm:
    #   c

    #boundary_vertices = extract_contour(u)


    ### Kontrolle, ob erste Kontur passt
                      
    #x= boundary_vertices[:,0]
    #y= boundary_vertices [:,1]

    #plt.figure(figsize=(8, 6))
    #plt.plot(x, y, marker="o", linestyle="-", color="blue")
    #plt.title("Pfad der Punkte")
    #plt.xlabel("x-Koordinaten")
    #plt.ylabel("y-Koordinaten")
    #plt.grid(True)
    #plt.show()
    ###

                      
    # initial set for the local descent
    #boundary_vertices = resample(boundary_vertices, num_boundary_vertices_ld, point_density_ld)
   # simple_set = SimpleSet(boundary_vertices, max_tri_area_ld)

    ### test

    #plot_simple_set(simple_set, eta=eta, display_inner_mesh=False)
    #print(simple_set.boundary_vertices)

   # weighted_area = simple_set.compute_weighted_area(eta)      
   # perimeter = simple_set.compute_perimeter()
   # print("integral fixed mesh:", weighted_area)
   # print("perimeter fixed mesh:", perimeter)
   # print("objective fixed mesh:", perimeter / np.abs(weighted_area))
    ###
                      
    # perform the local descent step
#    optimizer = CheegerOptimizer(step_size_ld, max_iter_ld, convergence_tol_ld, num_boundary_vertices_ld,
 #                                point_density_ld, max_tri_area_ld, num_iter_resampling_ld, 0.1, 0.5)

  #  cheeger_set, obj_tab, grad_norm_tab = optimizer.run(eta, simple_set)

   # if plot_results_ld:
   #     plot_simple_set(cheeger_set, eta=eta, display_inner_mesh=False)

      ###
    #    weighted_area = cheeger_set.compute_weighted_area(eta)      
     #   perimeter = cheeger_set.compute_perimeter()
     #   print("integral local descent:", weighted_area)
     #   print("perimeter local descent:", perimeter)
     #   print("objective local descent:", perimeter / np.abs(weighted_area))
      ###


                      
    #x_values = [v[0] for v in cheeger_set.boundary_vertices]
    #y_values = [v[1] for v in cheeger_set.boundary_vertices]
    #x_values = [v[0] for v in simple_set.boundary_vertices]
   # y_values = [v[1] for v in simple_set.boundary_vertices]                 

   # x_min= np.clip(min(x_values), 0,1)
   # x_max= np.clip(max(x_values), 0,1)
   # y_min= np.clip(min(y_values), 0,1)
    #y_max= np.clip(max(y_values), 0,1)
   
   # print("x_min", x_min, "x_max", x_max, "y_min", y_min, "y_max", y_max )

    #outer_vertices= np.array([x_min, x_max, y_min, y_max])
   # rectangle_boundary_vertices= np.array([[x_min,y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])

    #outer_vertices= np.array([1-x_max, 1-x_min, y_min, y_max])
    #rectangle_boundary_vertices= np.array([[1-x_max,y_min], [1-x_max, y_max], [1-x_min, y_max], [1-x_min, y_min]])
   # rectangle_set = RectangularSet(rectangle_boundary_vertices)

   # print("Starting value objective rectangular:", rectangle_set.compute_perimeter_rec() /  np.abs(rectangle_set.compute_weighted_area_rec(eta)))
   # print("Starting perimeter:", rectangle_set.compute_perimeter_rec())
   # print("Starting area value integral:", np.abs(rectangle_set.compute_weighted_area_rec(eta)))

   # start = time.time()

  #  result = minimize(objective, outer_vertices, args=(eta,), bounds=[(0,1),(0,1), (0,1), (0,1)] , options={'maxiter': 10000, 'disp': True, 'ftol': 1e-7, 'gtol': 1e-6})
                      
   # end = time.time()

    #optimal_rectangle = result.x
    #optimal_objective = result.fun
    #print("Optimales Rechteck:", optimal_rectangle)
   # print("Optimales Objective:", optimal_objective)
                      
   # opt_rectangle_boundary_vertices= np.array([[optimal_rectangle[0], optimal_rectangle[2]], [optimal_rectangle[0], optimal_rectangle[3]], [optimal_rectangle[1], optimal_rectangle[3]], [optimal_rectangle[1], optimal_rectangle[2]]])
   # opt_rect_set = RectangularSet(opt_rectangle_boundary_vertices)      
    
   # plot_rectangular_set(opt_rect_set, eta=eta, display_inner_mesh=False)
  #  print("Die Berechnung des Rechtecks hat ", end - start, " Sekunden gedauert." )
   # print("Perimeter:", opt_rect_set.compute_perimeter_rec())
  #  print("Value integral :", opt_rect_set.compute_weighted_area_rec(eta))
  #  print("Objective:", opt_rect_set.compute_perimeter_rec()/ np.abs(opt_rect_set.compute_weighted_area_rec(eta)))
                      
    #return simple_set, obj_tab, grad_norm_tab, opt_rect_set
    #return simple_set, opt_rect_set


#def rectangular_cheeger 
