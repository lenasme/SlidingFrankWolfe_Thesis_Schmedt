import numpy as np
import matplotlib.pyplot as plt




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



