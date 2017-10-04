import numpy as np

N = 250
f = np.arange(N*N, dtype=np.int).reshape((N,N))
g = np.arange(81, dtype=np.int).reshape((9, 9))

%timeit -n3 py_naive_convolve(f, g)

%timeit cy_naive_convolve(f, g)
