import numpy as np
import matplotlib.pyplot as plt

from numba import jit, prange



@jit(nopython=True)
def find_threshold(y):
    """
    Compute the value of the Lagrange multiplier involved in the projection of x into the unit l1 ball

    Parameters
    ----------
    x : array, shape (N,)
        Vector to be projected

    Returns
    -------
    res : float
        Value of the Lagrange multiplier

    Notes
    -----
    Sorting based algorithm. See [1]_ for a detailed explanation of the computations and alternative algorithms.

    References
    ----------
    .. [1] L. Condat, *Fast Projection onto the Simplex and the l1 Ball*, Mathematical Programming,
           Series A, Springer, 2016, 158 (1), pp.575-585.

    """

    j = len(y)
    stop = False

    partial_sum = np.sum(y)

    while j >= 1 and not stop:
        j = j - 1
        stop = (y[j] - (partial_sum - 1) / (j + 1) > 0)

        if not stop:
            partial_sum -= y[j]

    res = (partial_sum - 1) / (j + 1)

    return res


@jit(nopython=True, parallel=True)
def proj_one_one_unit_ball_aux(x, norm_row_x, thresh, res):
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            if norm_row_x[i, j] > thresh:
                res[i, j] = (norm_row_x[i, j] - thresh) * x[i, j] / norm_row_x[i, j]


def proj_one_one_unit_ball(x):
    """
    Projection onto the (1,1) unit ball

    Parameters
    ----------
    x : array, shape (N, M, 2)
        Should be seen as a (N*M, 2) matrix to be projected

    Returns
    -------
    res : array, shape (N, M, 2)
        Projection

    Notes
    -----
    See [1]_ for a detailed explanation of the computations and alternative algorithms.

    References
    ----------
    .. [1] L. Condat, *Fast Projection onto the Simplex and the l1 Ball*, Mathematical Programming,
           Series A, Springer, 2016, 158 (1), pp.575-585.

    """
    norm_row_x = np.linalg.norm(x, ord=1, axis=-1)
    norm_x = np.sum(norm_row_x)

    if norm_x > 1:
        res = np.zeros_like(x)
        y = np.sort(norm_row_x.ravel())[::-1]
        thresh = find_threshold(y)
        proj_one_one_unit_ball_aux(x, norm_row_x, thresh, res)
    else:
        res = x

    return res


def prox_inf_inf_norm(x, tau):
    """
    Proximal map of the (2, infinity) norm

    Parameters
    ----------
    x : array, shape (N,)
    tau : float


    Returns
    -------
    array, shape (N,)

    Notes
    -----
    .. math:: prox_{\\tau \\, ||.||_{2,\\infty}}(x) = x - \\tau ~ \\text{proj}_{\\{||.||_{2,1}\\leq 1\\}}(x / \\tau)

    """
    return x - tau * proj_one_one_unit_ball(x / tau)


def prox_dot_prod(x, tau, a):
    """
    Proximal map of the inner product between x and a

    Parameters
    ----------
    x : array, shape (N,)
    tau : float
    a : array, shape (N,)

    Returns
    -------
    array, shape (N,)

    """
    return x - tau * a


def postprocess_indicator(x):
    """
    Post process a piecewise constant function on a mesh to get an indicator function of a union of cells

    Parameters
    ----------
    x : array, shape (N + 2, N + 2)
        Values describing the piecewise constant function to be processed

    Returns
    -------
    array, shape (N, N)
        Values of the indicator function on each pixel of the grid

    """
    res = np.zeros_like(x)

    _, bins = np.histogram(x, bins=2)

    i1 = np.where(x < bins[1])
    i2 = np.where(x > bins[1])

    mean1 = np.mean(x[i1])
    mean2 = np.mean(x[i2])

    if abs(mean1) < abs(mean2):
        res[i1] = 0
        res[i2] = mean2
    else:
        res[i2] = 0
        res[i1] = mean1
  
    res /= np.sum(np.abs(grad(res)).sum(axis=-1))
    return res


@jit(nopython=True, parallel=True)
def update_grad(u, res):
    n = u.shape[0] - 2

    for i in prange(n + 1):
        for j in prange(n + 1):
            res[i, j, 0] = u[i + 1, j] - u[i, j]
            res[i, j, 1] = u[i, j + 1] - u[i, j]


def grad(u):
    n = u.shape[0] - 2
    res = np.zeros((n + 1, n + 1, 2))
    update_grad(u, res)
    return res


@jit(nopython=True, parallel=True)
def update_adj_grad(phi, res):
    n = phi.shape[0] - 1

    for i in prange(1, n + 1):
        for j in prange(1, n + 1):
            res[i, j] = -(phi[i, j, 0] + phi[i, j, 1] - phi[i - 1, j, 0] - phi[i, j - 1, 1])


def adj_grad(phi):
    n = phi.shape[0] - 1
    res = np.zeros((n + 2, n + 2))
    update_adj_grad(phi, res)
    return res


def power_method(grid_size, n_iter=100):
    x = np.random.random((grid_size + 2, grid_size + 2))
    x[0, :] = 0
    x[grid_size + 1, :] = 0
    x[:, 0] = 0
    x[:, grid_size + 1] = 0

    for i in range(n_iter):
        x = adj_grad(grad(x))
        x = x / np.linalg.norm(x)

    return np.sqrt(np.sum(x * (adj_grad(grad(x)))) / np.linalg.norm(x))


def run_primal_dual(grid_size, eta_bar, max_iter=10000, convergence_tol=None, verbose=False, plot=False):
    """
    Solves the "fixed mesh weighted Cheeger problem" by running a primal dual algorithm

    Parameters
    ----------
    grid_size
    eta_bar : array, shape (N, 2)
        Integral of the weight function on each triangle
    max_iter : integer
        Maximum number of iterations (for now, exact number of iterations, since no convergence criterion is
        implemented yet)
    verbose : bool, defaut False
        Whether to print some information at the end of the algorithm or not
    plot : bool, defaut False
        Whether to regularly plot the image given by the primal variable or not

    Returns
    -------
    array, shape (N, 2)
        Values describing a piecewise constant function on the mesh, which solves the fixed mesh weighted Cheeger problem

    """
    grad_op_norm = power_method(grid_size)

    grad_buffer = np.zeros((grid_size + 1, grid_size + 1, 2))
    adj_grad_buffer = np.zeros((grid_size + 2, grid_size + 2))

    sigma = 0.99 / grad_op_norm
    tau = 0.99 / grad_op_norm

    phi = np.zeros((grid_size + 1, grid_size + 1, 2))  # dual variable
    u = np.zeros((grid_size + 2, grid_size + 2))  # primal variable
    former_u = u

    eta_bar_pad = np.zeros((grid_size + 2, grid_size + 2))
    eta_bar_pad[1:grid_size+1, 1:grid_size+1] = eta_bar

    convergence = False
    iter = 0

    while not convergence:
        update_grad(2 * u - former_u, grad_buffer)
        phi = prox_inf_inf_norm(phi + sigma * grad_buffer, sigma)

        former_u = np.copy(u)  
        update_adj_grad(phi, adj_grad_buffer)
        u = prox_dot_prod(u - tau * adj_grad_buffer, tau, eta_bar_pad)
        iter += 1

        if convergence_tol is None:
            convergence = iter > max_iter
        else:
            convergence = np.linalg.norm(u - former_u) / np.linalg.norm(u) < convergence_tol

    if verbose:
        print(np.linalg.norm(u - former_u) / np.linalg.norm(u))

    return u


def extract_contour(u):
    v = postprocess_indicator(u)

    n = v.shape[0] - 2
   
    
    h = 1 / n
    grad_v = grad(v)
    edges = []

    for i in range(n+1):
        for j in range(n+1):
            if np.abs(grad_v[i, j, 0]) > 0:
                
                x, y =  i * h,  (j - 1) * h
                
                
                edges.append([[x, y], [x, y + h]])
            if np.abs(grad_v[i, j, 1]) > 0:

                x, y =  (i - 1) * h,  j * h

                edges.append([[x, y], [x + h, y]])

    edges = np.array(edges)

    path_vertices = [edges[0][0], edges[0][1]]

    mask = np.ones(len(edges), dtype=bool)
    mask[0] = False

    done = False

    while not done:
        prev_vertex = path_vertices[-1]
        where_next = np.where(np.isclose(edges[mask], prev_vertex[None, None, :]).all(2))

        if where_next[0].size == 0:
            done = True

        else:
            i, j = where_next[0][0], where_next[1][0]

            next_vertex = edges[mask][i, 1 - j]
            path_vertices.append(next_vertex)

            count = 0
            k = 0
            while count < i + 1:
                if mask[k]:
                    count += 1
                k += 1
            mask[k-1] = False

    return np.array(path_vertices[:-1])


def plot_primal_dual_results(u, eta_bar):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 14))

    grid_size = u.shape[0]
    h = 1 / grid_size


    
    eta_avg = eta_bar  / h ** 2

    v_abs_max = np.max(np.abs(eta_avg))

    im = axs[0].imshow(eta_avg, cmap='bwr', vmin=-v_abs_max, vmax=v_abs_max)
    axs[0].axis('equal')
    axs[0].axis('on') # vorher off
    fig.colorbar(im, ax=axs[0])

    v_abs_max = np.max(np.abs(u))

    im = axs[1].imshow(u, cmap='bwr', vmin=-v_abs_max, vmax=v_abs_max)
    axs[1].axis('equal')
    axs[1].axis('on') # vorher off
    fig.colorbar(im, ax=axs[1])

    plt.title("Primal-Dual Results")

    plt.show()