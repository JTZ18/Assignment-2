from typing import Tuple, List
from array import array
import numpy as np
from numexpr import evaluate
from timeit import default_timer as timer
from tqdm import tqdm
import matplotlib.pyplot as plt

MATRIX_SIZES = (1, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200, 250, 375, 500)
NUM_OF_EXP = 10

def init_list_matrices(size: int):
	a = [np.random.rand(size) for _ in range(size)]
	b = [np.random.rand(size) for _ in range(size)]
	c = [np.random.rand(size) for _ in range(size)]
	return a, b, c

def init_array_matrices(size: int):
	"""	It is not possible to create a 2D array using the array module.
		Thus, a 1D array with size size*size is made instead."""
	a = array("d", np.random.rand(size*size))
	b = array("d", np.random.rand(size*size))
	c = array("d", np.random.rand(size*size))
	return a, b, c

def init_numpy_matrices(size: int):
	a = np.random.rand(size, size).astype(np.double)
	b = np.random.rand(size, size).astype(np.double)
	c = np.random.rand(size, size).astype(np.double)
	return a, b, c

def dgemm_list(a, b, c, size):
	for i in range(size):
		for j in range(size):
			for k in range(size):
				c[i][j] += a[i][k] * b[k][j]
	return c

def dgemm_array(a, b, c, size):
	for i in range(size):
		for j in range(size):
			for k in range(size):
				c[i*size+j] += a[i*size+k] * b[k*size+j]
	return c

def dgemm_numpy(a, b, c, size):
	c += np.matmul(a, b)
	return c

def dgemm_numexpr(a, b, c, size):
	temp = np.matmul(a, b)
	evaluate("temp + c", out=c)
	return c

def run_experiment(initMatrixFunc, dgemmFunc) -> Tuple[list]:
	avg_times = []
	std_times = []
	min_times = []
	max_times = []

	for matrix_size in MATRIX_SIZES:
		time_spent = []
		for _ in tqdm(range(NUM_OF_EXP), desc=f"Running {dgemmFunc.__name__} - {matrix_size}*{matrix_size} matrix", ncols=100):
			a, b, c = initMatrixFunc(matrix_size)
			start_time = timer()
			dgemmFunc(a, b, c, matrix_size)
			time_spent.append(timer() - start_time)

		avg_times.append(np.average(time_spent))
		std_times.append(np.std(time_spent))
		min_times.append(min(time_spent))
		max_times.append(max(time_spent))

	return avg_times, std_times, min_times, max_times

def output_stats(stats):
	print("Size\t"+"\t\t".join([f"{i}*{i}" for i in MATRIX_SIZES]))
	print("Avg\t"+"\t".join([f"{i:.4E}" for i in stats[0]]))
	print("Std\t"+"\t".join([f"{i:.4E}" for i in stats[1]]))
	print("Min\t"+"\t".join([f"{i:.4E}" for i in stats[2]]))
	print("Max\t"+"\t".join([f"{i:.4E}" for i in stats[3]]))

if __name__ == "__main__":
	experiment_results = run_experiment(init_list_matrices, dgemm_list)
	output_stats(experiment_results)
	plt.plot(MATRIX_SIZES, experiment_results[0], label="Python Lists")

	experiment_results = run_experiment(init_array_matrices, dgemm_array)
	output_stats(experiment_results)
	plt.plot(MATRIX_SIZES, experiment_results[0], label="Array Module")

	experiment_results = run_experiment(init_numpy_matrices, dgemm_numpy)
	output_stats(experiment_results)
	plt.plot(MATRIX_SIZES, experiment_results[0], label="NumPy Array")

	experiment_results = run_experiment(init_numpy_matrices, dgemm_numexpr)
	output_stats(experiment_results)
	plt.plot(MATRIX_SIZES, experiment_results[0], label="NumPy Array w/ numexpr")

	plt.title("Execution time")
	plt.xlabel("Matrix size N*N")
	plt.ylabel("Average time spent (s)")
	plt.legend()
	plt.show()