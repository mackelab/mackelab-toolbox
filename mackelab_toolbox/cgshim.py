# Tie-in to the theano_shim module (which I plan on renaming to `cgshim`
# if/when it gets extended to other computational graph libraries)

import numpy as np
import mackelab_toolbox.parameters as parameters
import theano_shim as shim

import logging
logger = logging.getLogger()

# Map `float` to `floatX`
# Use a function because `floatX` can be changed at runtime
cast_floatX = lambda: np.dtype(shim.config.__class__.floatX.fget(shim.config)).type
parameters.type_map[float] = cast_floatX
parameters.type_map['floatX'] = cast_floatX

# For symbolic types, we just assume that val has already been properly casted.
# We do this to keep the reference to the original variable
def cast_symbolic(value, type, name):
    if shim.is_symbolic(value):  # Catches both tensors and shared vars
        if not hasattr(value, 'name') or value.name is None:
            value.name = name
        if not np.can_cast(value.dtype, type):
            logger.error("Attempted to cast a symbolic variable of type "
                         f"{value.dtype} to type {type}, which is not a "
                         "permitted cast. Note that for symbolic variables, we "
                         "perform no casting, so types must match exactly.")
        elif not np.dtype(value.dtype) is np.dtype(type):
            logger.warning("Attempted to cast a symbolic variable of type "
                 f"{value.dtype} to type {type}."
                 "Note that for symbolic variables, we perform "
                 "no casting, so types must match exactly.")
        return value
    else:
        return NotImplemented
parameters.cast_functions.appendleft(cast_symbolic)
    # appendleft for precedence => we want to prevent casting of symbolics
