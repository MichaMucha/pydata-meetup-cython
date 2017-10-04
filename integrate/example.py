import numpy as np
import pandas as pd

df = pd.DataFrame({'a': np.random.randn(1000),
                   'b': np.random.randn(1000),
                   'N': np.random.randint(100, 1000, (1000)),
                   'x': 'x'})

def f(x):
    return x * (x - 1)

def integrate_f(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f(a + i * dx)
    return s * dx

%timeit df.apply(lambda x: integrate_f(x['a'], x['b'], x['N']), axis=1)


%load_ext Cython
%%cython
cimport numpy as np
cdef double f_typed(double x) except? -2:
    return x * (x - 1)
cpdef double integrate_f_typed(double a, double b, int N):
    cdef int i
    cdef double s, dx
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_typed(a + i * dx)
    return s * dx

%timeit df.apply(lambda x: integrate_f_typed(x['a'], x['b'], x['N']), axis=1)

%%cython
cimport numpy as np
cdef double f_typed(double x) except? -2:
    return x * (x - 1)
cpdef double integrate_f_typed(double a, double b, int N):
    cdef int i
    cdef double s, dx
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_typed(a + i * dx)
    return s * dx
cpdef np.ndarray[double] apply_integrate_f(np.ndarray col_a, np.ndarray col_b, np.ndarray col_N):
    assert (col_a.dtype == np.float and col_b.dtype == np.float and col_N.dtype == np.int)
    cdef Py_ssize_t i, n = len(col_N)
    assert (len(col_a) == len(col_b) == n)
    cdef np.ndarray[double] res = np.empty(n)
    for i in range(len(col_a)):
        res[i] = integrate_f_typed(col_a[i], col_b[i], col_N[i])
    return res


%timeit apply_integrate_f(df['a'].values, df['b'].values, df['N'].values)
