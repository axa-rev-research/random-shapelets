import numpy as np
cimport numpy as np

cpdef np.ndarray[double] sqeuclidean(double[:] ts, double[:] shpt):

    cdef:
      int i, j
      int len_ts = len(ts)
      int len_shpt = len(shpt)
      np.ndarray[double] res = np.empty(len_ts-len_shpt)
      double tmp

    for i in range(len_ts-len_shpt):
        tmp = 0
        for j in range(len_shpt):
            tmp += (ts[i+j]-shpt[j])**2

        res[i] = tmp

    return res
