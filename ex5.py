import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from tqdm import tqdm

EXP_RANGE = range(8, 1025)
NUM_OF_EXP = 10

def create_random_input_signal(N: int):
	return np.random.random(N) + np.random.random(N) *1j

def create_N_by_N_DFT_matrix(N: int):
	i, j = np.meshgrid(np.arange(N), np.arange(N))
	omega = np.exp(-2 * np.pi * 1j / N)
	W = np.power(omega, i*j) / np.sqrt(N)
	return W


def DFT(x):
	"""This DFT uses the matrix-vector multiplication X = Wx to find the DFT of the input signal x.

	Args:
		x: Input signal, a vector.

	Return:
		X: The DFT of the signal.
	"""
	N = len(x)
	W = create_N_by_N_DFT_matrix(N)
	X = np.matmul(W, x)
	return X

def DFT_optimized(x):
	"""This DFT uses the the second equations given in the homework assignment to find the DFT of the input signal x.

	Args:
		x: Input singal, a vector.

	Returns:
		X: The DFT of the signal.
	"""
	N = len(x)
	X = np.zeros(N, dtype=np.cdouble)

	omega = np.exp(-2 * np.pi * 1j/ N)
	for k in range(N):
		y = np.power(np.power(omega, k), np.arange(N))
		X[k] = np.matmul(x, y) / np.sqrt(N)

	return X

def DFT_numpy(x):
	return np.fft.fft(x, norm="ortho")

def run_experiment(dft_func):
	avg_times = []
	for N in tqdm(EXP_RANGE, desc=f"Running {dft_func.__name__}", ncols=100):
		times = []

		for _ in range(NUM_OF_EXP):
			x = create_random_input_signal(N)
			start_time = timer()
			dft_func(x)
			times.append(timer() - start_time)

		avg_times.append(np.average(times))

	return avg_times


if __name__ == "__main__":
	time_spents = run_experiment(DFT)
	plt.plot(EXP_RANGE, time_spents, label="DFT")

	time_spents = run_experiment(DFT_optimized)
	plt.plot(EXP_RANGE, time_spents, label="DFT Optimized")

	plt.title("DFT Execution time")
	plt.xlabel("Input size")
	plt.ylabel("Average time spent (s)")
	plt.legend()
	plt.show()