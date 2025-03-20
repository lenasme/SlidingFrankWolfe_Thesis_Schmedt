import numpy as np
import matplotlib.pyplot as plt

from celer import Lasso

from Cheeger.rectangular_set import RectangularSet




class IndicatorFunction:
    def __init__(self, rectangular_set, grid_size, weight = 1):
        self.support = rectangular_set
        self.grid_size = grid_size
        self.weight = weight
        self.area = (rectangular_set.x_max - rectangular_set.x_min) * (rectangular_set.y_max - rectangular_set.y_min)
        self.mean_value = self.area / (self.grid_size ** 2)


    def __call__(self, x):

        if self.support.x_min <= x[1] <= self.support.x_max and self.support.y_min <= x[0] <= self.support.y_max:
            return self.weight * (1 - self.mean_value)
        else:
            return self.weight * (0 - self.mean_value)
        

    def construct_image_matrix(self, plot = True):
        x = np.linspace(0, self.grid_size, self.grid_size)
        y = np.linspace(0, self.grid_size, self.grid_size)

        X, Y = np.meshgrid(x,y)
        grid = np.stack([X,Y], axis = -1)

        image = np.array([[self(np.array([xi, yi])) for xi, yi in row] for row in grid])
        
        if plot == True:
            plt.imshow(image,  cmap= 'bwr')
            plt.colorbar()
            plt.show()
        
        return image

    def compute_truncated_frequency_image(self, cut_off, plot = True):
        image = self.construct_image_matrix(plot = plot)
        fourier_image = np.fft.fft2(image)
        freqs_x= np.fft.fftfreq(self.grid_size, d=1 / self.grid_size)
        freqs_y = np.fft.fftfreq(self.grid_size, d=1 / self.grid_size)
        freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")

        mask = np.zeros((self.grid_size, self.grid_size))
        mask[(np.abs(freq_x) <= cut_off) & (np.abs(freq_y) <= cut_off)] = 1

        truncated_fourier_image = fourier_image * mask

        if plot == True:
            plt.imshow(truncated_fourier_image.real, cmap = 'bwr')
            plt.colorbar()
            plt.show()
        
        return truncated_fourier_image


class SimpleFunction:
    def __init__(self, atoms, grid_size, cut_off):
        if not isinstance(atoms, list):
            atoms = [atoms]
        self.atoms = atoms
        self.grid_size = grid_size
        self.cut_off = cut_off
        


    def __call__(self, x):
        res = 0
        for f in self.atoms:
            res += f(x)
        return res
    
    @property
    def num_atoms(self):
        return len(self.atoms)


    def extend_support(self, rectangular_set):
        new_atom = IndicatorFunction(rectangular_set, self.grid_size)
        self.atoms.append(new_atom)

    

    def construct_image_matrix_sf(self, plot = True):
        x = np.linspace(0, self.grid_size, self.grid_size)
        y = np.linspace(0, self.grid_size, self.grid_size)

        X, Y = np.meshgrid(x,y)
        grid = np.stack([X,Y], axis = -1)

        image = np.array([[self(np.array([xi, yi])) for xi, yi in row] for row in grid])
        
        if plot == True:
            plt.imshow(image,  cmap= 'bwr')
            plt.colorbar()
            plt.title("Current Simple Function")
            plt.show()
        
        return image
    
    
    def compute_truncated_frequency_image_sf(self, cut_off, plot = True):
        image = self.construct_image_matrix_sf(plot = plot)
        fourier_image = np.fft.fft2(image)
        freqs_x= np.fft.fftfreq(self.grid_size, d=1 / self.grid_size)
        freqs_y = np.fft.fftfreq(self.grid_size, d=1 / self.grid_size)
        freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")

        mask = np.zeros((self.grid_size, self.grid_size))
        mask[(np.abs(freq_x) <= cut_off) & (np.abs(freq_y) <= cut_off)] = 1

        truncated_fourier_image = fourier_image * mask

        if plot == True:
            plt.imshow(truncated_fourier_image.real, cmap = 'bwr')
            plt.colorbar()
            plt.title("Current Truncated Fourier Frequency Image")
            plt.show()
        
        return truncated_fourier_image



#Hilfsmittel, die in Gradienten gebraucht werden: Dirac-Delta Distribution, Heavisidefunktion

def dirac_delta_approx(x, epsilon = 1e-6):
    return (np.abs(x) < epsilon) / (2 * epsilon)

def heaviside(x):
    return np.heaviside(x, 1.0)



def compute_objective_sliding(a, x_mins, x_maxs, y_mins, y_maxs, target_function_f, reg_param, grid_size, cut_off):
    """
    J_\ alpha(a, E) = 1/2 \Vert \sum_i a_i K0(1_E_i) -f \Vert^2 + \ alpha \sum_i \ text{perimeter}(E_i) abs(a_i)
    
    """
    indicator_function_values = np.zeros((len(a) , grid_size, grid_size))
    K0 = np.zeros((len(a), grid_size, grid_size), dtype=complex)   
    perimeters = np.zeros(len(a))

    freqs_x= np.fft.fftfreq(grid_size, d=1 / grid_size)
    freqs_y = np.fft.fftfreq(grid_size, d=1 / grid_size)
    freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")
    
    mask = np.zeros((grid_size, grid_size))
    mask[(np.abs(freq_x) <= cut_off) & (np.abs(freq_y) <= cut_off)] = 1
    
    for i in range(len(a)):
        indicator_function_values[i] = IndicatorFunction( RectangularSet(x_mins[i], x_maxs[i], y_mins[i], y_maxs[i]), grid_size).construct_image_matrix(plot = False)
        K0[i] = np.fft.fft2(indicator_function_values[i]) * mask
        perimeters[i] = 2 * (x_maxs[i] - x_mins[i]) + 2 * (y_maxs[i] - y_mins[i])

    sum_K0_a = np.sum(a[:,None, None] * K0, axis = 0)
    error_term = np.linalg.norm(sum_K0_a - target_function_f)**2

    regularization_term = reg_param * np.sum(perimeters * np.abs(a))

    objective = 0.5 * error_term + regularization_term

    return objective.real


def compute_derivative_zero_mean_indicator( x_min, x_max, y_min, y_max, grid_size):

    grad_xmin = np.zeros((grid_size, grid_size))
    grad_xmax = np.zeros((grid_size, grid_size))
    grad_ymin = np.zeros((grid_size, grid_size))
    grad_ymax = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):

            xi = i
            yi = j
            
            H_xmin = heaviside(xi - x_min)
            H_xmax = heaviside(x_max - xi)
            H_ymin = heaviside(yi - y_min)
            H_ymax = heaviside(y_max - yi)

            delta_xmin = dirac_delta_approx(xi - x_min)
            delta_xmax = dirac_delta_approx(x_max - xi)
            delta_ymin = dirac_delta_approx(yi - y_min)
            delta_ymax = dirac_delta_approx(y_max - yi)

            grad_xmin[i,j] = - delta_xmin * H_xmax * H_ymin * H_ymax + (y_max - y_min)/(grid_size**2)
            grad_xmax[i,j] = delta_xmax * H_xmin * H_ymin * H_ymax - (y_max - y_min)/(grid_size**2)
            grad_ymin[i,j] = - delta_ymin * H_xmin * H_xmax * H_ymax + (x_max - x_min)/(grid_size**2)
            grad_ymax[i,j] = delta_ymax * H_xmin * H_xmax * H_ymin - (x_max - x_min)/(grid_size**2)

    return grad_xmin, grad_xmax, grad_ymin, grad_ymax





def compute_gradient_sliding( a, x_mins, x_maxs, y_mins, y_maxs, target_function_f, reg_param, grid_size, cut_off):

    freqs_x= np.fft.fftfreq(grid_size, d=1 / grid_size)
    freqs_y = np.fft.fftfreq(grid_size, d=1 / grid_size)
    freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")
    
    mask = np.zeros((grid_size, grid_size))
    mask[(np.abs(freq_x) <= cut_off) & (np.abs(freq_y) <= cut_off)] = 1

    indicator_function_values = np.zeros((len(a) , grid_size, grid_size))
    indicator_gradients = np.zeros((len(a), 4, grid_size, grid_size))

    K0_indicators = np.zeros((len(a), grid_size, grid_size), dtype=complex)   
    K0_gradient_indicators = np.zeros((len(a), 4, grid_size, grid_size), dtype=complex)

    perimeters = np.zeros(len(a))
    perimeter_gradients = np.zeros((len(a), 4))

    final_gradient = np.zeros((len(a), 5 ), dtype = complex)

    for i in range(len(a)):
        indicator_function_values[i] = IndicatorFunction( RectangularSet(x_mins[i], x_maxs[i], y_mins[i], y_maxs[i]), grid_size).construct_image_matrix(plot = False)
        
        #K0_indicators[i] = np.fft.fft2(indicator_function_values[i]) * mask
        K0_indicators[i] = np.fft.fft2(indicator_function_values[i]) * mask/(grid_size**2)

        
        perimeters[i] = 2 * (x_maxs[i] - x_mins[i]) + 2 * (y_maxs[i] - y_mins[i])

        perimeter_gradients[i, 0]= -2
        perimeter_gradients[i, 1]= 2
        perimeter_gradients[i, 2]= -2
        perimeter_gradients[i, 3]= 2

        indicator_gradient_i_x_min, indicator_gradient_i_x_max, indicator_gradient_i_y_min, indicator_gradient_i_y_max = compute_derivative_zero_mean_indicator( x_mins[i], x_maxs[i], y_mins[i], y_maxs[i], grid_size)

        indicator_gradients[i,0]= indicator_gradient_i_x_min
        indicator_gradients[i,1]= indicator_gradient_i_x_max
        indicator_gradients[i,2]= indicator_gradient_i_y_min
        indicator_gradients[i,3]= indicator_gradient_i_y_max

        #K0_gradient_indicators[i, 0] = np.fft.fft2(indicator_gradients[i,0]) * mask
        #K0_gradient_indicators[i, 1] = np.fft.fft2(indicator_gradients[i,1]) * mask
        #K0_gradient_indicators[i, 2] = np.fft.fft2(indicator_gradients[i,2]) * mask
        #K0_gradient_indicators[i, 3] = np.fft.fft2(indicator_gradients[i,3]) * mask

        K0_gradient_indicators[i, 0] = np.fft.fft2(indicator_gradients[i,0]) * mask / (grid_size)**2
        K0_gradient_indicators[i, 1] = np.fft.fft2(indicator_gradients[i,1]) * mask / (grid_size)**2
        K0_gradient_indicators[i, 2] = np.fft.fft2(indicator_gradients[i,2]) * mask / (grid_size)**2
        K0_gradient_indicators[i, 3] = np.fft.fft2(indicator_gradients[i,3]) * mask / (grid_size)**2



    sum_K0_a = np.sum(a[:, None, None] * K0_indicators, axis = 0)
    residual = sum_K0_a - target_function_f

    print("Residual max abs (real):", np.max(np.abs(np.real(residual))))
    print("Residual max abs (imag):", np.max(np.abs(np.imag(residual))))

    for i in range(len(a)):
        final_gradient[i, 1]= a[i] * np.sum( np.conj(residual) * K0_gradient_indicators[i,0]) + reg_param * np.abs(a[i]) * perimeter_gradients[i, 0]
        final_gradient[i, 2]= a[i] * np.sum( np.conj(residual) * K0_gradient_indicators[i,1]) + reg_param * np.abs(a[i]) * perimeter_gradients[i, 1]
        final_gradient[i, 3]= a[i] * np.sum( np.conj(residual) * K0_gradient_indicators[i,2]) + reg_param * np.abs(a[i]) * perimeter_gradients[i, 2]
        final_gradient[i, 4]= a[i] * np.sum( np.conj(residual) * K0_gradient_indicators[i,3]) + reg_param * np.abs(a[i]) * perimeter_gradients[i, 3]

        final_gradient[i,0] = np.sum( np.conj(residual) * K0_indicators[i]) +  reg_param * perimeters[i] * np.sign(a[i])

    
    return np.real(final_gradient)



def objective_wrapper_sliding(params, target_function_f, reg_param, grid_size, cut_off):
    N = len(params) // 5
    a = params[:N]
    x_mins = params[N:2*N]
    x_maxs = params[2*N:3*N]
    y_mins = params[3*N:4*N]
    y_maxs = params[4*N:5*N]

    return compute_objective_sliding(a, x_mins, x_maxs, y_mins, y_maxs, target_function_f, reg_param, grid_size, cut_off)


def gradient_wrapper_sliding(params, target_function_f, reg_param, grid_size, cut_off):
    N = len(params) // 5
    a = params[:N]
    x_mins = params[N:2*N]
    x_maxs = params[2*N:3*N]
    y_mins = params[3*N:4*N]
    y_maxs = params[4*N:5*N]

    grad = compute_gradient_sliding(a, x_mins, x_maxs, y_mins, y_maxs, target_function_f, reg_param, grid_size, cut_off)

    return grad.flatten().real

    

        

    



  
 
    