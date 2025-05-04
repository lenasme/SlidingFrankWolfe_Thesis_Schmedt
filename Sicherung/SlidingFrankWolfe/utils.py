import matplotlib.pyplot as plt
import numpy as np
import itertools

from scipy import interpolate
from shapely.geometry import Polygon
from .simple_function_debugging_mesh import SimpleFunction, ZeroWeightedIndicatorFunction

import matplotlib.ticker as tick
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})
rc('text', usetex=True)


def simple_function_diff(u,v):
    atoms = []

    for atom in u.atoms:
        atoms.append(atom)

    for atom in v.atoms:
        atoms.append(ZeroWeightedIndicatorFunction(atom.support, -atom.weight ))

    return SimpleFunction(atoms)


def plot_simple_function_aux(f, ax, m):
    offset_value = -sum(atom.weight * atom.support.compute_area_rec() for atom in f.atoms)
    ax.fill([0,1,1,0], [0,0,1,1], color=m.to_rgba(offset_value))

    n=f.num_atoms
    idx_set = set(list(range(n)))

    for k in range (1, n+1):
        for indices in itertools.combinations(idx_set, k):
            p= Polygon([(0,0),(1,0),(1,1),(0,1)])
            weight = offset_value
            for i in indices:
                points_i = f.atoms[i].support.boundary_vertices
                #
                if np.max(points_i) > 1 and f.imgsz:
                    points_i = points_i / f.imgsz
                #
                p_i = Polygon([tuple(points_i[k]) for k in range(len(points_i))])
                p=p.intersection(p_i)
                weight += f.atoms[i].weight

            if not p.is_empty:
                if p.geom_type == 'MultiPolygon':
                    for q in list(p):
                    #for q in p.geoms:
                        coords = list(q.boundary.coords)
                        x= np.array([coords[i][0] for i in range(len(coords))])
                        y= np.array([coords[i][1] for i in range(len(coords))])
                        #x = [coord[0] for coord in coords]
                        #y = [coord[1] for coord in coords]

                        ax.fill(x,y, color = m.to_rgba(weight))

                elif p.geom_type == 'Polygon':
                    coords = list(p.boundary.coords)
                    #coords = list(p.exterior.coords)
                    x= np.array([coords[i][0] for i in range(len(coords))])
                    y= np.array([coords[i][1] for i in range(len(coords))])
                    #x = [coord[0] for coord in coords]
                    #y = [coord[1] for coord in coords]

                    ax.fill(x,y, color = m.to_rgba(weight))

                else:
                    #print("p.geom_type ist weder polygon noch multipolygon sondern ein", p.geom_type)
                    continue


def plot_simple_function(f, m, save_path = None):
    fig, ax = plt.subplots(figsize = (7,7))
    ax.set_aspect('equal')

    plot_simple_function_aux(f, ax, m)

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis('off')

    cbar = fig.colorbar(m, ax=ax, fraction= 0.046, pad= 0.04, format=tick.FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=30)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()


def plot_obs(y, cmap, v_abs_max=None, save_path=None):
    if np.iscomplexobj(y):
        y = np.abs(y)

    if v_abs_max is None:
        v_abs_max = np.max(y)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')

    n = np.int(np.sqrt(y.size))

    im = ax.imshow(y.reshape((n, n)), origin='lower', cmap=cmap, vmin=-v_abs_max, vmax=v_abs_max)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format=tick.FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=30)

    ax.axis('off')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()