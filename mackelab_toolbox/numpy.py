import numbers
import numpy as np
from mackelab_toolbox import utils

############
# Update terminating types
# FIXME: A better solution would be for terminating_types to be dynamic:
# it could inspect sys.modules, and add types for loaded modules
# NB: Pint and Quantities types added in load_pint(), load_quantities()
utils._terminating_types |= {numbers.Number, np.ndarray, np.number}
utils.terminating_types = tuple(utils._terminating_types)

############
# Add byte converters
def _dtype_to_bytes(dtype: np.dtype) -> bytes:
    return utils._tobytes(str(dtype))
utils._byte_converters[np.dtype] = _dtype_to_bytes
