import numpy as np
from numpy import exp
from numba import jit, prange
import quadpy
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

def generate_square_aux(grid, weights, cut_off):
    scheme = quadpy.c2.get_good_scheme(3)
    scheme_weights = scheme.weights
    scheme_points = (1 + scheme.points.T) / 2

    frequency_image = weights.reshape(grid.shape[0], grid.shape[1]) 
        
       
    reconstructed_not_vanish = np.fft.ifft2((frequency_image)).real  
    reconstructed_vanish = reconstructed_not_vanish - (np.sum(reconstructed_not_vanish)/(grid.shape[0]*grid.shape[1]))    

    @jit(nopython=False, parallel=True)
    def aux(grid_size, res):

       
        
        scale = grid.size / grid_size 
        #h = 2 / grid_size
        for i in prange(grid_size):
            for j in prange(grid_size):

                x_min = i * scale
                x_max = (i + 1) * scale
                y_min = j * scale
                y_max = (j + 1) * scale

                def integrand(x, y):
                # Bildwert an der Stelle (x, y), wobei x und y im Bereich [0,1] liegen
                    x_img = x_min + (x * scale)
                    y_img = y_min + (y * scale)
                    return reconstructed_vanish[int(x_img), int(y_img)]
            
                integral_value = 0
                for k in range(scheme_weights.size):

                    x, y = scheme_points[k]
                    integral_value += scheme_weights[k] * integrand(x, y)
                   
                res[i, j] = integral_value

                #res[i, j] *= h ** 2

    return aux

#def downsample_image(image, target_shape):
    # Berechne die Skalierungsfaktoren für jede Dimension
    scale_x = target_shape[0] / image.shape[0]
    scale_y = target_shape[1] / image.shape[1]
    
    # Wende die Reskalierung an (bilineare Interpolation)
    downsampled_image = zoom(image, (scale_x, scale_y))
    return downsampled_image


#def generate_square_aux(grid, weights, cut_off):
    

    @jit(nopython=False, parallel=True)
    def aux(grid_size, res):
        h = 1/ grid_size
        frequency_image = weights.reshape(grid.shape[0], grid.shape[1]) 
        
        frequency_image_grid_size = downsample_image(frequency_image, (grid_size, grid_size))        
       
        reconstructed_image_grid_size_not_vanish = np.fft.ifft2((frequency_image_grid_size)).real  
        res[:] = reconstructed_image_grid_size_not_vanish - (np.sum(reconstructed_image_grid_size_not_vanish)/(grid_size*grid_size))     
        
        res[:] *= h**2 #WARUM muss ich hier skalieren??

    return aux


       
#def generate_triangle_aux(grid, weights, cut_off):
    

    @jit(nopython=False, parallel=True)
    def aux(triangles, res):
        for i in prange(len(triangles)):
            a = np.sqrt((triangles[i, 1, 0] - triangles[i, 0, 0]) ** 2 + (triangles[i, 1, 1] - triangles[i, 0, 1]) ** 2)
            b = np.sqrt((triangles[i, 2, 0] - triangles[i, 1, 0]) ** 2 + (triangles[i, 2, 1] - triangles[i, 1, 1]) ** 2)
            c = np.sqrt((triangles[i, 2, 0] - triangles[i, 0, 0]) ** 2 + (triangles[i, 2, 1] - triangles[i, 0, 1]) ** 2)
            p = (a + b + c) / 2
            area = np.sqrt(p * (p - a) * (p - b) * (p - c))

            for k in range(scheme_weights.size):
                x = scheme_points[k, 0] * triangles[i, 0, 0] + \
                    scheme_points[k, 1] * triangles[i, 1, 0] + \
                    scheme_points[k, 2] * triangles[i, 2, 0]
                y = scheme_points[k, 0] * triangles[i, 0, 1] + \
                    scheme_points[k, 1] * triangles[i, 1, 1] + \
                    scheme_points[k, 2] * triangles[i, 2, 1]
                for j in range(grid.shape[0]):
                    squared_norm = (x - grid[j, 0]) ** 2 + (y - grid[j, 1]) ** 2
                    res[i] += scheme_weights[k] * weights[j] * exp(scale * squared_norm)

            res[i] *= area

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