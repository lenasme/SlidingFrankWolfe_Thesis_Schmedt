import numpy as np
import matplotlib.pyplot as plt

from celer import Lasso




class IndicatorFunction:
    def __init__(self, rectangular_set, grid_size, weight = 0.5):
        self.support = rectangular_set
        self.grid_size = grid_size
        self.weight = weight
        self.area = (rectangular_set.x_max - rectangular_set.x_min) * (rectangular_set.y_max - rectangular_set.y_min)
        self.mean_value = self.area / (self.grid_size ** 2)


    def __call__(self, x):

        if self.support.x_min <= x[0] <= self.support.x_max and self.support.y_min <= x[1] <= self.support.y_max:
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
            plt.imshow(image.T,  cmap= 'bwr')
            plt.colorbar()
            plt.show()
        
        return image

    def compute_truncated_frequency_image(self, cut_off, show = True):
        image = self.construct_image_matrix(plot = True)
        fourier_image = np.fft.fft2(image)
        freqs_x= np.fft.fftfreq(self.grid_size, d=1 / self.grid_size)
        freqs_y = np.fft.fftfreq(self.grid_size, d=1 / self.grid_size)
        freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")

        mask = np.zeros((self.grid_size, self.grid_size))
        mask[(np.abs(freq_x) <= cut_off) & (np.abs(freq_y) <= cut_off)] = 1

        truncated_fourier_image = fourier_image * mask

        if show == True:
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
            plt.imshow(image.T,  cmap= 'bwr')
            plt.colorbar()
            plt.show()
        
        return image
    
    
    def compute_truncated_frequency_image_sf(self, cut_off, show = True):
        image = self.construct_image_matrix_sf(plot = True)
        fourier_image = np.fft.fft2(image)
        freqs_x= np.fft.fftfreq(self.grid_size, d=1 / self.grid_size)
        freqs_y = np.fft.fftfreq(self.grid_size, d=1 / self.grid_size)
        freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")

        mask = np.zeros((self.grid_size, self.grid_size))
        mask[(np.abs(freq_x) <= cut_off) & (np.abs(freq_y) <= cut_off)] = 1

        truncated_fourier_image = fourier_image * mask

        if show == True:
            plt.imshow(truncated_fourier_image.real, cmap = 'bwr')
            plt.colorbar()
            plt.show()
        
        return truncated_fourier_image







    



    #@property
    #def num_atoms(self):
    #    return len(self.atoms)

   # @property
   # def weights(self):
   #     return np.array([atom.weight for atom in self.atoms])

   # @property
   # def supports(self):
    #    return [atom.support for atom in self.atoms]


   # @property
   # def support_boundary_vertices(self):
   #     return [atom.support.boundary_vertices for atom in self.atoms]


    
   # def transform_into_image(self, grid_size):
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)

        X, Y = np.meshgrid(x,y)
        grid = np.stack([X,Y], axis = -1)

        image = np.array([[self((xi, yi)) for xi, yi in row] for row in grid])
        return image


   # def compute_obs(self, fourier, version=0):
        if self.num_atoms == 0:
            return np.zeros(fourier.grid_size)

        max_num_triangles = max(len(atom.support.mesh_faces) for atom in self.atoms)
        meshes = np.zeros((self.num_atoms, max_num_triangles, 3, 2))
        obs = np.zeros((self.num_atoms, max_num_triangles, fourier.grid_size))

        for i in range(self.num_atoms):
            support_i = self.atoms[i].support
            meshes[i, :len(support_i.mesh_faces)] = support_i.mesh_vertices[support_i.mesh_faces]

        fourier._triangle_aux(meshes, obs) 

        

        if version == 1:
            res = [obs[i, :len(self.atoms[i].support.mesh_faces), :] for i in range(self.num_atoms)]
        else:
            res = np.zeros(fourier.grid_size)
            for i in range(self.num_atoms):
                res += self.atoms[i].weight * np.sum(obs[i], axis=0)

        return res




    #def extend_support(self, rectangular_set, weight = 0.5):
     
     #   new_atom = WeightedIndicatorFunction(weight, rectangular_set)
        #if not isinstance(self.atoms, list):
         #   self.atoms = []
        #self.atoms.append(new_atom)

    #def extend_support(self, rectangular_set):
        ### zero
        new_atom = ZeroWeightedIndicatorFunction(rectangular_set)
        #new_atom = WeightedIndicatorFunction(rectangular_set)
        #if not isinstance(self.atoms, list):
         #   self.atoms = []
        self.atoms.append(new_atom)


   # def linear_fit_weights(self, gamma, M, f):
        scaled_atoms = [
            #ZeroWeightedIndicatorFunction(atom.support, atom.weight * (1- gamma))
            WeightedIndicatorFunction(atom.support, atom.weight * (1- gamma))
            for atom in self.atoms[:-1]]
        new_weight = - self.atoms[-1].weight * (gamma * M * np.sign(self.atoms[-1].support.compute_weighted_area_rec(f))/ self.atoms[-1].support.compute_perimeter_rec())
        print("self.atoms perimeter... größer als 4???", self.atoms[-1].support.compute_perimeter_rec())
        print("self.atoms.rect_area in weight berechnung:", self.atoms[-1].support.compute_weighted_area_rec(f))
        new_atom = ZeroWeightedIndicatorFunction(self.atoms[-1].support, new_weight)
        print(new_weight)
        
        scaled_atoms.append(new_atom)
        self.atoms = scaled_atoms
        print("die gewichte:", self.weights)
        #return SimpleFunction([scaled_atoms, new_atom])

    #def fit_weights4(self, y, cut_f, grid_size, reg_param, tol_factor=1e-4):
        obs = self.compute_phi_E(cut_f)

        
        mat = obs
        y= y.real

        
        print("mat shape:", mat.shape)
        print("y shape:", y.shape)

        tol = tol_factor * np.linalg.norm(y)**2 / y.size
       # perimeters = np.array([self.atoms[i].support.compute_perimeter() for i in range(self.num_atoms)])
        perimeters = np.array([self.atoms[i].support.compute_perimeter_rec() for i in range(self.num_atoms)])
        print("Perimeter:", perimeters)
        
        lasso = Lasso(alpha=reg_param/(y.size ), fit_intercept=False, tol=tol, weights=perimeters)
        #lasso = Lasso(alpha=reg_param, fit_intercept=False, tol=tol, weights=perimeters)
        lasso.fit(mat, y.reshape(-1))

        new_weights = lasso.coef_
        print("current weights:", new_weights)
        
        self.atoms = [ZeroWeightedIndicatorFunction( self.atoms[i].support, new_weights[i])
                      for i in range(self.num_atoms) if np.abs(new_weights[i]) > 1e-2]
        # TODO: clean zero weight condition

    #def fit_weights(self, y, phi, reg_param, tol_factor=1e-4):
    #def fit_weights(self, y, cut_f, grid_size, reg_param, tol_factor=1e-4):
        obs = self.compute_obs(cut_f, grid_size, version=1)

        #print(obs)
        
        #mat = np.array([np.sum(obs[i], axis=0) for i in range(self.num_atoms)])
        mat = np.array([obs[i].reshape(-1) for i in range(self.num_atoms)])
        #mat = mat.T
        mat = mat.reshape((self.num_atoms, -1)).T
        mat = mat.reshape((-1, mat.shape[-1]))
        #mat = mat.reshape((grid_size**2, self.num_atoms)).T  # oder (N, 1)

        mat = mat.real
        y= y.real

        
        print("mat shape:", mat.shape)
        print("y shape:", y.shape)

        tol = tol_factor * np.linalg.norm(y)**2 / y.size
       # perimeters = np.array([self.atoms[i].support.compute_perimeter() for i in range(self.num_atoms)])
        perimeters = np.array([self.atoms[i].support.compute_perimeter_rec() for i in range(self.num_atoms)])
        print("Perimeter:", perimeters)
        
        lasso = Lasso(alpha=reg_param/(y.size ), fit_intercept=False, tol=tol, weights=perimeters)
        #lasso = Lasso(alpha=reg_param, fit_intercept=False, tol=tol, weights=perimeters)
        lasso.fit(mat, y.reshape(-1))

        new_weights = lasso.coef_
        print("current weights:", new_weights)
        ### zero
        #self.atoms = [ZeroWeightedIndicatorFunction( self.atoms[i].support, new_weights[i])
         #             for i in range(self.num_atoms) if np.abs(new_weights[i]) > 1e-2]
        self.atoms = [WeightedIndicatorFunction( self.atoms[i].support,new_weights[i])
                      for i in range(self.num_atoms) if np.abs(new_weights[i]) > 1e-2]
        # TODO: clean zero weight condition

 
    