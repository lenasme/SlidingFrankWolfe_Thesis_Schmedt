import numpy as np
from celer import Lasso

class ZeroWeightedIndicatorFunction:
    def __init__(self, simple_set, weight=1.0):
        self.support = simple_set
        self.weight = weight

    def __call__(self, x):
        if self.support.contains(x):
            return self.weight * (1 - self.support.compute_area_rec() / 1)
        else:
            return self.weight * (0 - self.support.compute_area_rec() / 1)

def generate_fourier_aux_rect(grid, cut_f):
    k_vals = np.array([[k1, k2] for k1 in range(-cut_f, cut_f + 1) for k2 in range(-cut_f, cut_f + 1)])
    num_freqs = len(k_vals)
    fourier_matrix = np.zeros((len(grid), num_freqs), dtype=complex)

    for i, x in enumerate(grid):
        for j, k in enumerate(k_vals):
            fourier_matrix[i, j] = np.exp(-2j * np.pi * np.dot(k, x))

    return fourier_matrix

class SimpleFunction:
    def __init__(self, atoms=None):
        self.atoms = atoms if atoms is not None else []

    @property
    def num_atoms(self):
        return len(self.atoms)

    def weights(self):
        return np.array([atom.weight for atom in self.atoms])

    @property
    def supports(self):
        return [atom.support for atom in self.atoms]

    @property
    def support_boundary_vertices(self):
        return [atom.support.boundary_vertices for atom in self.atoms]

    def compute_obs(self, cut_f, grid_size, version=0):
        if self.num_atoms == 0:
            print("atoms Liste ist leer")
            return np.zeros((grid_size, grid_size))

        grid = np.array([[x, y] for x in range(grid_size) for y in range(grid_size)])

        if version == 0:
            combined_image = np.zeros((grid_size, grid_size))
            for atom in self.atoms:
                atom_image = atom.support.transform_into_image(grid_size)
                combined_image += atom.weight * atom_image
                

            combined_image_flat = combined_image.reshape(-1, 2)
            truncated_transform = generate_fourier_aux_rect(combined_image_flat, cut_f)
            return truncated_transform
        elif version == 1:
            obs = np.zeros((self.num_atoms, grid_size, grid_size), dtype=complex)
            for i, atom in enumerate(self.atoms):
                atom_image = atom.support.transform_into_image(grid_size)
                obs[i] = generate_fourier_aux_rect(grid, cut_f)
            return obs

    def extend_support(self, simple_set):
        new_atom = ZeroWeightedIndicatorFunction(simple_set)
        self.atoms.append(new_atom)

    def fit_weights(self, y, cut_f, grid_size, reg_param, tol_factor=1e-4):
        obs = self.compute_obs(cut_f, grid_size, version=1)
        mat = np.array([np.sum(obs[i], axis=0) for i in range(self.num_atoms)])
        mat = mat.reshape((self.num_atoms, -1)).T

        tol = tol_factor * np.linalg.norm(y)**2 / y.size
        perimeters = np.array([self.atoms[i].support.compute_perimeter_rec() for i in range(self.num_atoms)])

        lasso = Lasso(alpha=reg_param/y.size, fit_intercept=False, tol=tol, weights = perimeters)
        lasso.fit(mat, y.reshape(-1))

        new_weights = lasso.coef_
        self.atoms = [ZeroWeightedIndicatorFunction(self.atoms[i].support, weight=new_weights[i])
                      for i in range(self.num_atoms) if np.abs(new_weights[i]) > 1e-2]
