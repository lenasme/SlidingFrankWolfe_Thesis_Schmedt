import numpy as np
import quadpy
import matplotlib.pyplot as plt

#from numpy import exp
from numba import jit, prange
from scipy.fft import ifftshift


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



def generate_triangle_aux(grid, cut_off, normalization):
    scheme = quadpy.t2.get_good_scheme(5)
    scheme_weights = scheme.weights
    scheme_points = scheme.points.T

    # Frequenzgitter erstellen
    freqs = np.fft.fftfreq(grid.shape[0], d=1 / grid.shape[0])
    freq_x, freq_y = np.meshgrid(freqs, freqs, indexing="ij")
    freq_norms = np.abs(freq_x) + np.abs(freq_y)

    print("Min freq_x:", np.min(freq_x), "Max freq_x:", np.max(freq_x))
    print("Min freq_y:", np.min(freq_y), "Max freq_y:", np.max(freq_y))

    # Frequenzmaske erstellen
    mask = (freq_norms <= cut_off)
    mask = np.fft.ifftshift(mask)
    plt.imshow(mask)
    plt.colorbar()
    plt.show()

    #@jit(nopython=True, parallel=True)
    def aux(meshes, res):
        print("Meshes:", meshes[:5])
        for i in range(len(meshes)):
            for j in range(len(meshes[i])):
                # Berechnung der Dreiecksfläche
                a = np.sqrt((meshes[i, j, 1, 0] - meshes[i, j, 0, 0]) ** 2 + (meshes[i, j, 1, 1] - meshes[i, j, 0, 1]) ** 2)
                b = np.sqrt((meshes[i, j, 2, 0] - meshes[i, j, 1, 0]) ** 2 + (meshes[i, j, 2, 1] - meshes[i, j, 1, 1]) ** 2)
                c = np.sqrt((meshes[i, j, 2, 0] - meshes[i, j, 0, 0]) ** 2 + (meshes[i, j, 2, 1] - meshes[i, j, 0, 1]) ** 2)
                p = (a + b + c) / 2
                area = np.sqrt(p * (p - a) * (p - b) * (p - c))

                for m in range(grid.shape[0]):
                    for n in range(scheme_weights.size):
                        x = scheme_points[n, 0] * meshes[i, j, 0, 0] + \
                            scheme_points[n, 1] * meshes[i, j, 1, 0] + \
                            scheme_points[n, 2] * meshes[i, j, 2, 0]

                        y = scheme_points[n, 0] * meshes[i, j, 0, 1] + \
                            scheme_points[n, 1] * meshes[i, j, 1, 1] + \
                            scheme_points[n, 2] * meshes[i, j, 2, 1]

                        print("x:", x[:10])  # Zeigt die ersten 10 x-Werte
                        print("y:", y[:10])  # Zeigt die ersten 10 y-Werte

                        # Fourier-Transformation für den Punkt (x, y)
                        fft_image = np.exp(-2j * np.pi * (freq_x * x + freq_y * y))
                        print("Shape von fft_image:", fft_image.shape)
                        print("Shape von np.sum(fft_image, axis=0):", np.sum(fft_image, axis=0).shape)
                        #print("Min:", np.min(np.abs(fft_image)))
                        #print("Max:", np.max(np.abs(fft_image)))
                        #print("Mean:", np.mean(np.abs(fft_image)))
                        #print("Std Dev:", np.std(np.abs(fft_image)))
                        #plt.imshow(np.abs(fft_image), origin="lower", cmap="inferno")
                        #plt.colorbar()
                        #plt.title("FFT-Bild vor Maskierung")
                        #plt.show()

                        # Anwenden der Frequenzmaske
                        fft_filtered = fft_image * mask

                        

                        res[i, j, m] += scheme_weights[n] * np.sum(fft_filtered).real

                    res[i, j, m] *= area

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
