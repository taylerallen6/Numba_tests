from numba import njit, prange, float64, int64
import numpy as np
import time


x = np.arange(2500000)


def no_numba_func(x):
  n = len(x)
  acc = 0.0
  for i in range(n):
    acc += np.sqrt(x[i])
  return acc

@njit(parallel=True, fastmath=True)
def numba_func(x):
  n = len(x)
  acc = 0.0
  for i in prange(n):
    acc += np.sqrt(x[i])
  return acc

@njit(float64(int64[:]), parallel=True, fastmath=True)
def numba_precompile_func(x):
  n = len(x)
  acc = 0.0
  for i in prange(n):
    acc += np.sqrt(x[i])
  return acc


# PYTHON BY ITSELF, NO NUMBA
start = time.time()
no_numba_func(x)
end = time.time()
print("Elapsed (python, no numba) = %s" % (end - start))

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
numba_func(x)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
numba_func(x)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))

# NUMBA PRECOMPILE
start = time.time()
numba_precompile_func(x)
end = time.time()
print("Elapsed (numba precompile) = %s" % (end - start))