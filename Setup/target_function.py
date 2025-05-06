import numpy as np
import matplotlib.pyplot as plt

from .ground_truth import construction_of_example_source


def calculate_target_function(grid_size, deltas, max_jumps, cut_off, variance, seed= None,  plot = True):
	
	
	ground_truth = construction_of_example_source(grid_size, deltas, max_jumps, seed = seed)

	operator_applied_on_ground_truth = np.fft.fft2(ground_truth)

	freqs_x= np.fft.fftfreq(grid_size, d=1 / grid_size)
	freqs_y = np.fft.fftfreq(grid_size, d=1 / grid_size)
	freq_x, freq_y = np.meshgrid(freqs_x, freqs_y, indexing="ij")
	

	
	mask = np.zeros((grid_size, grid_size))
	mask[(np.abs(freq_x) <= cut_off) & (np.abs(freq_y) <= cut_off)] = 1

	truncated_operator_applied_on_ground_truth = operator_applied_on_ground_truth * mask

	rng = np.random.default_rng(seed)
	noise =  (rng.standard_normal((grid_size, grid_size)) + 1j * rng.standard_normal((grid_size, grid_size)))
	current_norm = np.linalg.norm(noise)
	target_norm = np.sqrt(2 * variance)
	noise *= target_norm / current_norm


	target_function_f = truncated_operator_applied_on_ground_truth + noise

	if plot == True:
		

		plt.plot()
		plt.imshow(ground_truth, cmap = 'bwr')
		plt.colorbar()
		plt.title("Ground Truth")
		plt.show()

		plt.plot()
		plt.imshow(truncated_operator_applied_on_ground_truth.real, cmap= 'bwr')
		plt.colorbar()
		plt.title("Truncated Fourier Frequency Image")
		plt.show()

		
		plt.plot()
		plt.imshow(np.fft.ifft2(truncated_operator_applied_on_ground_truth).real, cmap = 'bwr')
		plt.colorbar()
		plt.title("Reconstruction of Truncated Fouried Applied on Ground Truth")
		plt.show()


	return truncated_operator_applied_on_ground_truth, ground_truth, target_function_f