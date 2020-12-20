# -*- coding: utf-8 -*-
# # Timing tests - utilities

# +
import numpy as np

import mackelab_toolbox as mtb
import mackelab_toolbox.utils
# -

# ## Index iter

# Implementation in library:

mtb.utils.Code(mtb.utils.index_iter)


# Equivalent NumPy implementation:

def np_iter(A):
    it = np.nditer(A, flags=['multi_index'])
    for _ in it:
        yield it.multi_index


shape = (100, 100, 100)
A = np.zeros(shape)

# %%timeit
for _ in np_iter(A):
    pass

#     199 ms ± 1.47 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# %%timeit
for _ in mtb.utils.index_iter(A.shape):
    pass

#     27.9 ms ± 118 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)


