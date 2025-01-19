import numpy as np

from celer import Lasso


# erstellt die einzelnen Indikatorfunktionen
class WeightedIndicatorFunction:
    def __init__(self, weight, simple_set):
    # simple_set oder rectangular_set?
        self.weight = weight
        self.support = simple_set

    def __call__(self, x):
        #x_values = [v[0] for v in self.boundary_vertices]
        #y_values = [v[1] for v in self.boundary_vertices]

        #min_x = min(x_values)
        #max_x = max(x_values)
        #min_y = min(y_values)
        #max_y = max(y_values)

        #area = np.abs(x_max - x_min) * np.abs(y_max - y_min)
        
        if self.support.contains(x):
            return self.weight #- area
        else:
            return 0 #- area

class ZeroWeightedIndicatorFunction:
    def __init__(self, simple_set, weight = 1.0):
        self.support = simple_set
        self.weight = weight

    def __call__(self, x):
        if self.support.contains(x):
            return self.weight * (1 - self.support.compute_area_rec() / 1 )
            #return self.weight # * (1 - self.support.compute_area_rec() / 1 )
        else:
            return self.weight * (0 - self.support.compute_area_rec() / 1 )
            #return 0

# fasst die verschiedenen Indikatorfunktionen zu einer simple function mit mehreren Atomen zusammen
# atoms werden instanzen von WeightedIndicatorFunction sein
class SimpleFunction:
    def __init__(self, atoms, imgsz = 120):
        ### zero
        if isinstance(atoms, ZeroWeightedIndicatorFunction):
            atoms = [atoms]
        self.atoms = atoms
        self.imgsz = imgsz

    def __call__(self, x):
    # addiert die indikatorfunktionen an der jeweiligen Stelle x
        res = 0
        for f in self.atoms:
            res += f(x)
        return res


    #def __mul__(self, scalar):
     #   """
      #  Skaliert die SimpleFunction mit einem Skalar.
       # """
        #if not isinstance(scalar, (int, float)):
         #   raise ValueError("SimpleFunction can only be multiplied by a scalar.")
        #scaled_atoms = [
         #   ZeroWeightedIndicatorFunction(atom.support, atom.weight * scalar)
          #  for atom in self.atoms
        #]
        #return SimpleFunction(scaled_atoms)

    @property
    def num_atoms(self):
        return len(self.atoms)

    @property
    def weights(self):
        return np.array([atom.weight for atom in self.atoms])

    @property
    def supports(self):
        return [atom.support for atom in self.atoms]


    @property
    def support_boundary_vertices(self):
        return [atom.support.boundary_vertices for atom in self.atoms]


    
    def transform_into_image(self, grid_size):
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)

        X, Y = np.meshgrid(x,y)
        grid = np.stack([X,Y], axis = -1)

        image = np.array([[self((xi, yi)) for xi, yi in row] for row in grid])
        return image


    #obs: observation
    # f: u?
    #def compute_obs(self, f, version=0):
    #    if self.num_atoms == 0:
    #        return np.zeros(f.grid_size)

        #max_num_triangles = max(len(atom.support.mesh_faces) for atom in self.atoms)
        #meshes = np.zeros((self.num_atoms, max_num_triangles, 3, 2))
        #obs = np.zeros((self.num_atoms, max_num_triangles, f.grid_size))
    #        for i in range(self.num_atoms):
     #       support_i = self.atoms[i].support
      #      meshes[i, :len(support_i.mesh_faces)] = support_i.mesh_vertices[support_i.mesh_faces]

       # f._triangle_aux(meshes, obs)
#
 #       if version == 1:
  #          res = [obs[i, :len(self.atoms[i].support.mesh_faces), :] for i in range(self.num_atoms)]
   #     else:
    #        res = np.zeros(f.grid_size)
     #       for i in range(self.num_atoms):
      #          res += self.atoms[i].weight * np.sum(obs[i], axis=0)

       # return res


    def compute_obs(self, cut_f, grid_size, version=0):
        from Setup.ground_truth import EtaObservation

        if self.num_atoms == 0:
            print("atoms Liste ist leer")
            return np.zeros((grid_size , grid_size))

        if version == 0:
            combined_image = np.zeros((grid_size, grid_size))
            fourier = EtaObservation(cut_f)
            for atom in self.atoms:
                atom_simple_function = SimpleFunction(atom, imgsz= grid_size)
                atom_image = atom_simple_function.transform_into_image(grid_size)
                combined_image += atom.weight * atom_image
            truncated_transform = fourier.trunc_fourier(combined_image)
            return np.real(truncated_transform)

        elif version == 1:
            observations = []
            fourier = EtaObservation(cut_f)
            for atom in self.atoms:
                atom_simple_function = SimpleFunction(atom, imgsz = grid_size)
                atom_image = atom_simple_function.transform_into_image(grid_size)
                truncated_transform = fourier.trunc_fourier(atom.weight * atom_image)
                observations.append(np.real(truncated_transform))
            return np.array(observations)
        else:
            raise ValueError("Invalid version specified. Use version=0 or version=1.")


    #def extend_support(self, rectangular_set, weight = 0.5):
     
     #   new_atom = WeightedIndicatorFunction(weight, rectangular_set)
        #if not isinstance(self.atoms, list):
         #   self.atoms = []
        #self.atoms.append(new_atom)

    def extend_support(self, rectangular_set):
        ### zero
        new_atom = ZeroWeightedIndicatorFunction(rectangular_set)
        #if not isinstance(self.atoms, list):
         #   self.atoms = []
        self.atoms.append(new_atom)


    def linear_fit_weights(self, gamma, M, f):
        scaled_atoms = [
            ZeroWeightedIndicatorFunction(atom.support, atom.weight * (1- gamma))
            for atom in self.atoms[:-1]]
        new_atom = ZeroWeightedIndicatorFunction(self.atoms[-1].support, self.atoms[-1].weight * (gamma * M * np.sign(self.atoms[-1].support.compute_weighted_area_rec(f))/ self.atoms[-1].support.compute_perimeter_rec() ))
        return SimpleFunction([scaled_atoms, new_atom])



    #def fit_weights(self, y, phi, reg_param, tol_factor=1e-4):
    def fit_weights(self, y, cut_f, grid_size, reg_param, tol_factor=1e-4):
        obs = self.compute_obs(cut_f, grid_size, version=1)

        print(obs)
        
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
        
        lasso = Lasso(alpha=reg_param/y.size, fit_intercept=False, tol=tol, weights=perimeters)
        #lasso = Lasso(alpha=reg_param, fit_intercept=False, tol=tol, weights=perimeters)
        lasso.fit(mat, y.reshape(-1))

        new_weights = lasso.coef_
        print("new weights:", new_weights)
        ### zero
        self.atoms = [ZeroWeightedIndicatorFunction(new_weights[i], self.atoms[i].support)
                      for i in range(self.num_atoms) if np.abs(new_weights[i]) > 1e-2]
        #self.atoms = [WeightedIndicatorFunction(new_weights[i], self.atoms[i].support)
        #              for i in range(self.num_atoms) if np.abs(new_weights[i]) > 1e-2]
        # TODO: clean zero weight condition

 

