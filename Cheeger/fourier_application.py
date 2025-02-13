import numpy as np
from numpy import exp
from numba import jit, prange
import matplotlib.pyplot as plt
from scipy.ndimage import zoom




def generate_eval_aux(grid, weights, cut_off):
    
    #freqs_x= np.fft.fftfreq(grid.shape[0], d=1 / grid.shape[0])
    #freqs_y = np.fft.fftfreq(grid.shape[1], d=1 / grid.shape[1])
    #freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")
    #freq_norms = np.abs(freq_x) + np.abs(freq_y)

    
    #mask = np.zeros((grid.shape[0], grid.shape[1]))
    #mask[freq_norms <= cut_off] = 1

    @jit(nopython=False, parallel=True)
    def aux(x, res):
        frequency_image = weights.reshape(grid.shape[0], grid.shape[1]) 
        reconstructed_image_not_vanish = np.fft.ifft2(frequency_image).real 
        reconstructed_image = reconstructed_image_not_vanish - (np.sum(reconstructed_image_not_vanish)/(grid.shape[0]*grid.shape[1]))

        plt.plot()
        plt.imshow(reconstructed_image, cmap = 'bwr')
        plt.colorbar()
        plt.show()

        print("Summe",np.sum(reconstructed_image))

        for i in prange(x.shape[0]):
            res[i]= reconstructed_image[int(x[i, 0]), int(x[i, 1])]
            #for j in range(weights.size):
                #squared_norm = (x[i, 0] - grid[j, 0]) ** 2 + (x[i, 1] - grid[j, 1]) ** 2
                #res[i] += weights[j] * exp(scale * squared_norm)

    return aux


def downsample_image(image, target_shape):
    # Berechne die Skalierungsfaktoren für jede Dimension
    scale_x = target_shape[0] / image.shape[0]
    scale_y = target_shape[1] / image.shape[1]
    
    # Wende die Reskalierung an (bilineare Interpolation)
    downsampled_image = zoom(image, (scale_x, scale_y))
    return downsampled_image


def generate_square_aux(grid, weights, cut_off):
    

    @jit(nopython=False, parallel=True)
    def aux(grid_size, res):
        frequency_image = weights.reshape(grid.shape[0], grid.shape[1]) 
        
        frequency_image_grid_size = downsample_image(frequency_image, (grid_size, grid_size))        
       
        reconstructed_image_grid_size_not_vanish = np.fft.ifft2((frequency_image_grid_size)).real  
        res[:] = reconstructed_image_grid_size_not_vanish - (np.sum(reconstructed_image_grid_size_not_vanish)/(grid_size*grid_size))     
        
    return aux


       





class FourierApplication:
    def __init__(self, grid, weights, cut_off, normalization = False):
        self.grid = grid
        self.weights = weights
        self.cut_off = cut_off
        self.normalization = normalization

        self._eval_aux = generate_eval_aux(self.grid, self.weights, self.cut_off)
        self._square_aux = generate_square_aux(self.grid, self.weights, self.cut_off)
        #self._triangle_aux = generate_triangle_aux(self.grid, self.weights, self.cut_off)
        #self._line_aux = generate_line_aux(self.grid, self.weights, self.cut_off)


    @property
    def grid_size(self):
        return len(self.grid)

    def __call__(self, x):
        if x.ndim == 1:
            tmp = np.zeros(1)
            self._eval_aux(np.reshape(x, (1, 2)), tmp)
            res = tmp[0]
        else:
            res = np.zeros(x.shape[0])
            self._eval_aux(x, res)
        #if self.normalization:
            #res = res / (2*np.pi * self.std**2)
        return res

    def integrate_on_pixel_grid(self, grid_size):
        res = np.zeros((grid_size, grid_size))
        self._square_aux(grid_size, res)
        return res

    #def integrate_on_triangles(self, triangles):
        res = np.zeros(len(triangles))
        self._triangle_aux(triangles, res)
        return res

    #def integrate_on_polygonal_curve(self, vertices):
        res = np.zeros((len(vertices), 2))
        self._line_aux(vertices, res)
        return res