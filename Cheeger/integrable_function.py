import numpy as np
import quadpy
from scipy.interpolate import RegularGridInterpolator
from numba import jit, prange
from math import exp

@jit(nopython=True, parallel=True)
def _jit_integrate(triangles, res, scheme_weights, interpolated_values):
    for i in prange(len(triangles)):
        # Dreiecksgeometrie
        a = np.sqrt((triangles[i, 1, 0] - triangles[i, 0, 0]) ** 2 + (triangles[i, 1, 1] - triangles[i, 0, 1]) ** 2)
        b = np.sqrt((triangles[i, 2, 0] - triangles[i, 1, 0]) ** 2 + (triangles[i, 2, 1] - triangles[i, 1, 1]) ** 2)
        c = np.sqrt((triangles[i, 2, 0] - triangles[i, 0, 0]) ** 2 + (triangles[i, 2, 1] - triangles[i, 0, 1]) ** 2)
        p = (a + b + c) / 2
        area = np.sqrt(p * (p - a) * (p - b) * (p - c))

        # Integration
        for k in range(scheme_weights.size):
            res[i] += scheme_weights[k] * interpolated_values[i, k]

        res[i] *= area


@jit(nopython=True, parallel=True)
def _jit_integrate_polygon(vertices, res, scheme_weights, interpolated_values_forward, interpolated_values_backward):
    """
    JIT-kompilierte Integration entlang der polygonalen Kurve.
    """
    for i in prange(len(vertices)):
        # Vorwärtskante: Punkt i -> i+1
        edge_length = np.sqrt(
            (vertices[(i + 1) % len(vertices), 0] - vertices[i, 0]) ** 2
            + (vertices[(i + 1) % len(vertices), 1] - vertices[i, 1]) ** 2
        )
        for k in range(len(scheme_weights)):
            res[i, 0] += scheme_weights[k] * interpolated_values_forward[i, k].item()
        res[i, 0] *= edge_length / 2

        # Rückwärtskante: Punkt i -> i-1
        edge_length = np.sqrt(
            (vertices[i, 0] - vertices[i - 1, 0]) ** 2
            + (vertices[i, 1] - vertices[i - 1, 1]) ** 2
        )
        for k in range(len(scheme_weights)):
            res[i, 1] += scheme_weights[k] * interpolated_values_backward[i, k].item()
        res[i, 1] *= edge_length / 2






class IntegrableFunction:
    def __init__(self, eta):
        """
        Initialisiere mit einem Raster (eta).
        """
        self.eta = eta
        self.grid_size = eta.shape
        x = np.linspace(0, 1, self.grid_size[0])
        y = np.linspace(0, 1, self.grid_size[1])
        self.interpolator = RegularGridInterpolator((x, y), eta, bounds_error = False, fill_value = 0.0)

    def __call__(self, x, y=None):
        """
        Interpoliert die Werte für die Eingabe `x` unter Verwendung des Interpolators.
        """
        if y is None:
            if x.ndim == 1:
                # Falls `x` ein einzelner Punkt ist
                return self.interpolator((x[0], x[1]))
            else:
                # Falls `x` mehrere Punkte sind
                points = np.array(x)
                #return np.array([self.interpolator((x_i[0], x_i[1])) for x_i in x])
                return np.array([self.interpolator((p[0], p[1])) for p in points])
        else: 
            print("bei call in y neq None gelandet")
            #if x.shape != y.shape:
             #   raise ValueError("x and y must have the same shape")
            # Falls `x` und `y` als separate Arrays (z. B. von np.meshgrid) übergeben werden

            if x.ndim == 2 and y.ndim == 2 and x.shape[1] == 1 and y.shape[0] == 1:
                x_mesh, y_mesh = np.meshgrid(x[:, 0], y[0, :], indexing='ij')
            elif x.shape == y.shape:
                x_mesh, y_mesh = x, y
            else:
                raise ValueError("x and y must have compatible shapes")


            points = np.stack((x_mesh.ravel(), y_mesh.ravel()), axis=-1)  # Kombiniere X und Y zu Punkten
            #try:
            values = self.interpolator(points)
            #except TypeError:
                #values = np.array([self.interpolator((p[0], p[1])) for p in points])  # Werte interpolieren
            return values.reshape(x_mesh.shape)  # Zurück zur ursprünglichen Form bringen




    def eval_aux(self, x):
        """
        Berechnet den interpolierten Wert für gegebene Punkte x.
        """
        return self.interpolator(x)

    def integrate_on_triangles(self, triangles):
        """
        Interpoliert die Werte von `eta` und integriert über die Dreiecke.
        """
        res = np.zeros(len(triangles))
        self._triangle_aux(triangles, res)
        return res

   



    def _triangle_aux(self, triangles, res):
        """
        Führt die Integration über Dreiecke durch, ohne den Interpolator direkt in der JIT-Funktion zu verwenden.
        """
        scheme = quadpy.t2.get_good_scheme(5)
        scheme_weights = scheme.weights
        scheme_points = scheme.points.T

        # Vorbereitung: Interpolation der Werte
        interpolated_values = []
        for triangle in triangles:
            triangle_values = []
            for k in range(scheme_points.shape[0]):
                # Berechnung der Punkte für Interpolation
                x = scheme_points[k, 0] * triangle[0, 0] + \
                    scheme_points[k, 1] * triangle[1, 0] + \
                    scheme_points[k, 2] * triangle[2, 0]
                y = scheme_points[k, 0] * triangle[0, 1] + \
                    scheme_points[k, 1] * triangle[1, 1] + \
                    scheme_points[k, 2] * triangle[2, 1]

                # Interpolationswert hinzufügen
                triangle_values.append(self.interpolator((x, y)))
            interpolated_values.append(triangle_values)

        interpolated_values = np.array(interpolated_values)

        # Aufruf der JIT-Funktion mit vorberechneten interpolierten Werten

        _jit_integrate(triangles, res, scheme_weights, interpolated_values)



    def integrate_on_pixel_grid(self, grid_size):
        """
        Skaliert das Array 'eta' auf die angegebene Gittergröße.
        """
        from scipy.ndimage import zoom

    

        scaling_factor = grid_size / self.eta.shape[0]
        scaled_array = zoom(self.eta, scaling_factor, order=1)  # Bilineare Interpolation
        #

        #min_val = np.min(scaled_array)
        #max_val = np.max(scaled_array)
        #scaled_array = (scaled_array - min_val) / (max_val - min_val)
        
        for i in range (grid_size):
            for j in range (grid_size):
                scaled_array[i,j] *= (1/ grid_size)**2
                
        ### hier stand vorher scaled_array[i,j]*= (2/grid_size)**2, dadurch wird aber das Gebiet [-1,1]^2 impliziert
        #
        return scaled_array


 


    def integrate_on_polygonal_curve(self, vertices):
        """
        Integriert die Werte von `eta` entlang einer polygonalen Kurve.
        """
        # Initialisiere das Ergebnisarray
        res = np.zeros((len(vertices), 2))

        # Numerisches Integrationsschema (Quadpy)
        scheme = quadpy.c1.gauss_patterson(3)
        scheme_weights = scheme.weights
        scheme_points = (1 + scheme.points) / 2

        # Vorbereitung: Interpolation der Werte entlang der Kanten
        interpolated_values_forward = []
        interpolated_values_backward = []

        for i in range(len(vertices)):
            # Vorwärtskante: Punkt i -> i+1
            forward_values = []
            for k in range(len(scheme_points)):
                point = (
                    scheme_points[k] * vertices[i]
                    + (1 - scheme_points[k]) * vertices[(i + 1) % len(vertices)]
                )
                forward_values.append(self.interpolator(point))
            interpolated_values_forward.append(forward_values)

            # Rückwärtskante: Punkt i -> i-1
            backward_values = []
            for k in range(len(scheme_points)):
                point = (
                    scheme_points[k] * vertices[i]
                    + (1 - scheme_points[k]) * vertices[i - 1]
                )
                backward_values.append(self.interpolator(point))
            interpolated_values_backward.append(backward_values)

        # Konvertiere in Numpy-Arrays
        interpolated_values_forward = np.array(interpolated_values_forward)
        interpolated_values_backward = np.array(interpolated_values_backward)

        # JIT-kompilierte Funktion aufrufen
        _jit_integrate_polygon(vertices, res, scheme_weights, interpolated_values_forward,       interpolated_values_backward)

        return res

        

        
