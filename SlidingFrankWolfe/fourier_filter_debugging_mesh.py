import numpy as np
import quadpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from numpy import exp
from numba import jit, prange
from scipy.fft import ifftshift, fft2, ifft2, fftshift 
from shapely.geometry import Point, Polygon


def generate_square_aux(grid, cut_off, normalization):
    scheme = quadpy.c2.get_good_scheme(3)
    scheme_weights = scheme.weights
    scheme_points = (1 + scheme.points.T) / 2
    

    @jit(nopython=True, parallel=True)
    def aux(grid_size, res):
        
        freqs = np.fft.fftfreq(grid.shape[0], d=1 / grid.shape[0])
        freq_x, freq_y = np.meshgrid(freqs, freqs, indexing="ij")
        freq_norms = np.abs(freq_x) + np.abs(freq_y)
        # Frequenzmaske erstellen
        freq_mask = freq_norms <= cut_off
       

        h = 2 / grid_size
        for i in prange(grid_size):
            for j in prange(grid_size):
                for n in range(scheme_weights.size):
                    x = -1 + h * (scheme_points[n, 0] + i)
                    y = -1 + h * (scheme_points[n, 1] + j)

                    # Fourier-Transformation für den Punkt (x, y)
                    fft_image = np.exp(-2j * np.pi * (freq_x * x + freq_y * y))

                    # Anwenden der Frequenzmaske
                    fft_filtered = fft_image * freq_mask

                    for k in range(grid.shape[0]):
                        res[i, j, k] += scheme_weights[n] * np.sum(fft_filtered).real

                res[i, j] *= h ** 2

        if normalization:
            res /= np.sum(freq_mask)

    return aux



def generate_triangle_aux(grid, cut_off,  normalization):
    #scheme = quadpy.t2.get_good_scheme(5)
    #scheme_weights = scheme.weights
    #scheme_points = scheme.points.T
    #print("scheme_weights:", scheme_weights)
    # Frequenzgitter erstellen
    freqs_x= np.fft.fftfreq(grid.shape[0], d=1 / grid.shape[0])
    freqs_y = np.fft.fftfreq(grid.shape[1], d=1 / grid.shape[1])
    freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")
    freq_norms = np.abs(freq_x) + np.abs(freq_y)

    
    mask = np.zeros((grid.shape[0], grid.shape[1]))
    mask[freq_norms <= cut_off] = 1
   
    # Frequenzmaske erstellen
    #mask = (freq_norms <= cut_off)
    #mask_expanded = np.expand_dims(mask, axis=-1) 
    mask = np.fft.fftshift(mask)
    plt.plot()
    plt.imshow(mask)
    plt.colorbar()
    plt.show()

    #@jit(nopython=True, parallel=True)
    def aux(meshes, function, res):
        #print("Meshes:", meshes[:5])
        for i in range(len(meshes)):
            print(len(meshes))
            print(function.atoms[i].support.boundary_vertices)
            #print(type(meshes[i]))  # Gibt den Typ des Elements aus
            #print(meshes[i])
            

            plt.figure(figsize=(8, 8))

            for j in range(len(meshes[i])):
                triangle = meshes[i][j]  # Holt sich die Vertices des Dreiecks
                plt.plot([triangle[0, 0], triangle[1, 0], triangle[2, 0], triangle[0, 0]], 
                    [triangle[0, 1], triangle[1, 1], triangle[2, 1], triangle[0, 1]], 'k-')
            plt.gca().set_aspect('equal', adjustable='box')  # Gleiche Skalierung für x- und y-Achse
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Mesh Visualization')
            plt.show()


            rectangle_vertices = function.atoms[i].support.boundary_vertices  
            rectangle_polygon = Polygon(rectangle_vertices)
            maske = np.zeros((grid.shape[0], grid.shape[1]), dtype=bool)
            for j in range(len(meshes[i])):

                function_grid = np.zeros((grid.shape[0], grid.shape[1]))

                triangle_vertices = np.array([[meshes[i,j,0,0], meshes[i,j,0,1]],[meshes[i,j,1,0], meshes[i,j,1,1]],[meshes[i,j,2,0], meshes[i,j,2,1]]])
                triangle_polygon = Polygon(triangle_vertices)

                #fig, ax = plt.subplots()

                # Plotte das Polygon
                #x, y = triangle_polygon.exterior.xy
                #ax.fill(x, y, alpha=0.5, color='blue')  # Das Polygon füllen (blau)
                #ax.plot(x, y, color='red')  # Das Polygonumriss (rot)

                #ax.set_xlim(0, 1)
                #ax.set_ylim(0, 1)

                # Achsen einstellen
                #ax.set_aspect('equal')
                #ax.set_xlabel('X')
                #ax.set_ylabel('Y')

                #plt.title('Randpolygon')
                #plt.show()

                for x in range(function_grid.shape[0]):
                    for y in range(function_grid.shape[1]):
                    
            
                        norm_x = x / function_grid.shape[0]
                        norm_y = y / function_grid.shape[1]

                        point = Point(norm_x, norm_y)

        
                        if not (triangle_polygon.contains(point) or triangle_polygon.boundary.contains(point)):
                            function_grid[x, y] = 0
                            
                        elif  (triangle_polygon.contains(point) or triangle_polygon.boundary.contains(point) ) and(rectangle_polygon.contains(point) or rectangle_polygon.boundary.contains(point)) :
                            if not mask[x,y]:
                                function_grid[x, y] = function.atoms[i].inner_value
                                maske[x,y] = True

                        else:   
                            if not mask[x,y]:
                                function_grid[x, y] = function.atoms[i].outer_value
                                maske[x,y] = True
                    
                    #if function.atoms[i].support.contains((norm_x, norm_y)):
                        #function_grid[x, y] = function.atoms[i].inner_value
                    #else:
                        #function_grid[x, y] = function.atoms[i].outer_value

                    



             
                plt.plot()
                plt.imshow(function_grid)
                plt.show()
                
                print("function_grid min/max:", np.min(function_grid), np.max(function_grid))
                
                
                fft_image = ((np.fft.fft2(function_grid)))
                shifted_fft_image = np.fft.fftshift(fft_image) * mask
            

                plt.plot()
                plt.imshow(shifted_fft_image.real)
                plt.show()

                #ifft_image = np.fft.ifft2(np.fft.ifftshift(fft_filtered)).real
                ifft_image = np.fft.ifft2(shifted_fft_image)
                plt.plot()
                plt.imshow(np.abs(ifft_image), cmap = 'bwr')
                plt.show()

                print("res shape:", res.shape)
                print("i, j:", i, j)
                print("shifted_fft_image shape:", shifted_fft_image.shape)
                res[i, j, :] = shifted_fft_image.flatten()   

                #res[i, j, m] += scheme_weights[n] * np.sum(fft_filtered).real

                #res[i, j, m] *= area

        if normalization:
            res /= np.sum(mask)

    return aux




def generate_line_aux(grid, cut_off, normalization):
    scheme = quadpy.c1.gauss_patterson(3)
    scheme_weights = scheme.weights
    scheme_points = (scheme.points + 1) / 2

    # Frequenzgitter erstellen
    freqs = np.fft.fftfreq(grid.shape[0], d=1 / grid.shape[0])
    freq_x, freq_y = np.meshgrid(freqs, freqs, indexing="ij")
    freq_norms = np.abs(freq_x) + np.abs(freq_y)

    # Frequenzmaske erstellen
    mask = freq_norms <= cut_off

    @jit(nopython=True, parallel=True)
    def aux(curves, mask_vertices, res):
        for i in range(len(curves)):
            num_vertices_i = np.sum(mask_vertices[i])
            for j in prange(num_vertices_i):
                edge_length = np.sqrt((curves[i, (j + 1) % num_vertices_i, 0] - curves[i, j, 0]) ** 2 +
                                      (curves[i, (j + 1) % num_vertices_i, 1] - curves[i, j, 1]) ** 2)

                # Integration entlang des Segments (Vorwärtsrichtung)
                for n in range(scheme_weights.size):
                    x = scheme_points[n] * curves[i, j] + (1 - scheme_points[n]) * curves[i, (j + 1) % num_vertices_i]

                    # Fourier-Transformation für den Punkt x
                    fft_image = np.exp(-2j * np.pi * (freq_x * x[0] + freq_y * x[1]))

                    # Anwenden der Frequenzmaske
                    fft_filtered = fft_image * mask

                    for m in range(grid.shape[0]):
                        res[i, j, m, 0] += scheme_weights[n] * np.sum(fft_filtered).real

                res[i, j, :, 0] *= edge_length / 2

                # Integration entlang des Segments (Rückwärtsrichtung)
                edge_length = np.sqrt((curves[i, j, 0] - curves[i, j - 1, 0]) ** 2 +
                                      (curves[i, j, 1] - curves[i, j - 1, 1]) ** 2)

                for n in range(scheme_weights.size):
                    x = scheme_points[n] * curves[i, j] + (1 - scheme_points[n]) * curves[i, j - 1]

                    # Fourier-Transformation für den Punkt x
                    fft_image = np.exp(-2j * np.pi * (freq_x * x[0] + freq_y * x[1]))

                    # Anwenden der Frequenzmaske
                    fft_filtered = fft_image * mask

                    for m in range(grid.shape[0]):
                        res[i, j, m, 1] += scheme_weights[n] * np.sum(fft_filtered).real

                res[i, j, :, 1] *= edge_length / 2

        if normalization:
            res /= np.sum(mask)

    return aux

# überall fft_filtered.real ergänzt


class TruncatedFourierTransform:
    def __init__(self, grid, cut_off, normalization=False):
        self.grid = grid
        self.cut_off = cut_off  # Cut-off frequency (1-Norm)
        self.normalization = normalization


        self._square_aux = generate_square_aux(self.grid, self.cut_off, self.normalization)
        self._triangle_aux = generate_triangle_aux(self.grid, self.cut_off, self.normalization)
        self._line_aux = generate_line_aux(self.grid, self.cut_off, self.normalization)


    @property
    def grid_size(self):
        return len(self.grid)
    
    def integrate_on_pixel_grid(self, grid_size):
        res = np.zeros((grid_size, grid_size, self.grid_size))
        self._square_aux(grid_size, res)
        return res

    def integrate_on_meshes(self, meshes):
        res = np.zeros((len(meshes), len(meshes[0]), self.grid_size))
        self._triangle_aux(meshes, res)
        return res

    def integrate_on_curves(self, curves):
        max_num_vertices = max([len(vertices) for vertices in curves])
        res = np.zeros((len(curves), max_num_vertices, self.grid_size, 2))
        curves_array = np.zeros((len(curves), max_num_vertices, 2))
        mask = np.zeros((len(curves), max_num_vertices), dtype='bool')

        for i in range(len(curves)):
            curves_array[i, :len(curves[i]), :] = curves[i]
            mask[i, :len(curves[i])] = True

        self._line_aux(curves_array, mask, res)
        return [res[i, :len(curves[i])] for i in range(len(curves))]

