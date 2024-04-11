from numba import njit, prange
import numpy as np
import time


# x = np.arange(1000000)
x = np.arange(2500000)
x = np.arange(100000000)

@njit(parallel=True, fastmath=True)
def ident_np(x):
  return np.cos(x) ** 2 + np.sin(x) ** 2

@njit(fastmath=True)
def ident_loops(x):
  r = np.empty_like(x)
  n = len(x)
  for i in range(n):
    r[i] = np.cos(x[i]) ** 2 + np.sin(x[i]) ** 2
  return r

@njit(parallel=True, fastmath=True, cache=True)
def do_sum_parallel(x):
  n = len(x)
  acc = 0.0
  for i in prange(n):
    acc += np.sqrt(x[i])
  return acc


# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
# ident_loops(x)
do_sum_parallel(x)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
# ident_loops(x)
do_sum_parallel(x)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))