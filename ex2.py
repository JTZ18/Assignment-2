from typing import Tuple, List
from array import array
import numpy as np
from timeit import default_timer as timer
import sys
import matplotlib.pyplot as plt

STREAM_ARRAY_SIZES = (1, 5, 10, 50, 100, 500, 1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000)
INIT_A_VAL = 1.0
INIT_B_VAL = 2.0
INIT_C_VAL = 0.0
SCALAR = 2.0

def init_python_lists(size: int) -> Tuple[list]:
	a = [INIT_A_VAL] * size
	b = [INIT_B_VAL] * size
	c = [INIT_C_VAL] * size
	return a, b, c

def init_array_module_arrays(size: int) -> Tuple[array]:
	pla, plb, plc = init_python_lists(size)
	a = array('d', pla)
	b = array('d', plb)
	c = array('d', plc)
	return a, b, c

def init_numpy_arrays(size: int) -> Tuple[np.ndarray]:
	a = np.full(size, INIT_A_VAL, dtype=np.double)
	b = np.full(size, INIT_B_VAL, dtype=np.double)
	c = np.full(size, INIT_C_VAL, dtype=np.double)
	return a, b, c

def time_the_operations(a, b, c, size: int) -> List[float]:
	times = []

	# Copy
	start_time = timer()
	for j in range(size):
		c[j] = a[j]
	times.append(timer() - start_time)

	# Scale
	start_time = timer()
	for j in range(size):
		b[j] = SCALAR * c[j]
	times.append(timer() - start_time)

	# Sum
	start_time = timer()
	for j in range(size):
		c[j] = a[j] + b[j]
	times.append(timer() - start_time)

	# Triad
	start_time = timer()
	for j in range(size):
		a[j] = b[j] + SCALAR * c[j]
	times.append(timer() - start_time)
	return times

def calc_memory_bandwidth(array_type, array_size: int, times: Tuple[float]) -> List[float]:
	mem_bandwiths = []

	# Copy
	mem_bandwiths.append((2 * sys.getsizeof(array_type) * array_size / 2**20) / times[0])

	# Add
	mem_bandwiths.append((2 * sys.getsizeof(array_type) * array_size / 2**20) / times[1])

	# Scale
	mem_bandwiths.append((3 * sys.getsizeof(array_type) * array_size / 2**20) / times[2])

	# Triad
	mem_bandwiths.append((3 * sys.getsizeof(array_type) * array_size / 2**20) / times[3])

	return mem_bandwiths

def stream_benchmark(size: int, initArrFunc) -> List[float]:
	a, b, c = initArrFunc(size)
	times = time_the_operations(a, b, c, size)
	memoryBandwidths = calc_memory_bandwidth(type(a), size, times)
	return memoryBandwidths

def report_results(data_type_init_func, label: str):
  performances = []

  for stream_size in STREAM_ARRAY_SIZES:
    performances.append(stream_benchmark(stream_size, data_type_init_func))

  for idx, operation in enumerate(("copy", "add", "scale", "triad")):
    plt.plot(STREAM_ARRAY_SIZES, [performance[idx] for performance in performances], label=label + " - " + operation)

  plt.title("STREAM Benchmark")
  plt.xlabel("Stream Array Sizes")
  plt.ylabel("Memory Bandwidth (MB/seconds)")
  plt.legend()
  plt.show()

if __name__ == "__main__":
  report_results(init_python_lists, "Python Lists")
  report_results(init_array_module_arrays, "Array Module Arrays")
  report_results(init_numpy_arrays, "NumPy Arrays")