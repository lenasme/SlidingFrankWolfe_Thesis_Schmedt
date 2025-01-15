import numpy as np

from celer import Lasso
from Setup.ground_truth import Trunc_Fourier

# erstellt die einzelnen Indikatorfunktionen
class WeightedIndicatorFunction:
    def __init__(self, weight, simple_set):
    # simple_set oder rectangular_set?
        self.weight = weight
        self.support = simple_set

    def __call__(self, x):
        if self.support.contains(x):
            return self.weight
        else:
            return 0

# fasst die verschiedenen Indikatorfunktionen zu einer simple function mit mehreren Atomen zusammen
# atoms werden instanzen von WeightedIndicatorFunction sein
class SimpleFunction:
    def __init__(self, atoms):
        self.atoms = atoms

    def __call__(self, x):
    # addiert die indikatorfunktionen an der jeweiligen Stelle x
        res = 0
        for f in self.atoms:
            res += f(x)
        return res

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
        if self.num_atoms == 0:
            return np.zeros((grid_size , grid_size))

        if version == 0:
            comined_image = np.zeros((grid_size, grid_size))
            for atom in self.atoms:
                atom_image = atom.transform_into_image(grid_size)
                combined_image += atom.weight * atom_image
            truncated_transform = Trunc_Fourier(combined_image, cut_f)
            return np.real(truncated_transform)

        elif version == 1:
            observation = []
            for atom in self.atoms:
                atom_image = atom.transform_into_image(grid_size)
                truncated_transform = Trunc_Fourier(atom.weight * atom_image, cut_f)
                observations.append(np.real(truncated_transform))
            return np.array(observations)
        else:
            raise ValueError("Invalid version specified. Use version=0 or version=1.")


    def extend_support(self, simple_set):
        new_atom = WeightedIndicatorFunction(0, simple_set)
        self.atoms.append(new_atom)

    #def fit_weights(self, y, phi, reg_param, tol_factor=1e-4):
    def fit_weights(self, y, cut_f, grid_size, tol_factor=1e-4):
        obs = self.compute_obs(cut_f, grid_size, version=1)
        
        mat = np.array([np.sum(obs[i], axis=0) for i in range(self.num_atoms)])
        mat = mat.reshape((self.num_atoms, -1)).T

        tol = tol_factor * np.linalg.norm(y)**2 / y.size
        perimeters = np.array([self.atoms[i].support.compute_perimeter() for i in range(self.num_atoms)])

        lasso = Lasso(alpha=reg_param/y.size, fit_intercept=False, tol=tol, weights=perimeters)
        lasso.fit(mat, y.reshape(-1))

        new_weights = lasso.coef_
        self.atoms = [WeightedIndicatorFunction(new_weights[i], self.atoms[i].support)
                      for i in range(self.num_atoms) if np.abs(new_weights[i]) > 1e-2]
        # TODO: clean zero weight condition

