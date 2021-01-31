"""
Tie-in to the theano_shim module (which I plan on renaming to `cgshim`
if/when it gets extended to other computational graph libraries)

Provides
--------
  - cast_floatX: Callable
    Function which casts to current type of 'floatX'
  - Injects the following types into `mackelab_toolbox.typing`:
    Has the elements:
    + FloatX: retrieved as `typing.FloatX`
      A type which maps to `.typing.NPValue[shim.config.floatX]`.
    + Symbolic: retrieved as `typing.Symbolic`
      A type, compatible with pydantic, to indicate a symbolic value in
      annotations. Which type that corresponds to depends on the currently
      loaded symbolic library.
    + Shared: retrieved as `typing.Shared`
      A type, compatible with pydantic, to indicate a shared value in
      annotations. Which type that corresponds to depends on the currently
      loaded symbolic library.
    + Tensor: retrieved as `typing.Tensor`
      A type, compatible with pydantic, corresponding to the union of
      Array, Symbolic and Shared types. (In other words, allowing both Numpy
      and symbolic inputs.)
    + AnyRNG: retrieved as `typing.AnyRNG`
      A type, compatible with pydantic, corresponding to eithen NumPy's
      RandomState or Theano's shared_RandomStream.RandomStream.
      Supports import/export to json.

Side-effects
------------
Importing this module also has the following side-effects:

  - Add the `shim` namespace to mackelab_toolbox.transform.Transforms.
    It points to 'theano_shim'.
  - Add a mapping from `float` (type) to `shim.config.floatX`
  - Add a mapping from 'floatX' (string) to `shim.config.floatX`
  - Prevent mackelab_toolbox.typing.Array from parsing computational graph types
"""

import sys
import numpy as np
from collections.abc import Iterable
from typing import Union
import theano_shim as shim
import mackelab_toolbox as mtb
import mackelab_toolbox.typing as typing
    # We still use mtb.typing to avoid confusion, but this makes
    # mtb.cgshim.typing equivalent to mtb.typing
import mackelab_toolbox.utils as utils
import mackelab_toolbox.transform

# Add 'shim' to Transform namspaces if it isn't already present
mtb.transform.Transform.namespaces.setdefault('shim', shim)

# Map `float` to `floatX`
# Use a function because `floatX` can be changed at runtime
cast_floatX = lambda: np.dtype(shim.config.__class__.floatX.fget(shim.config)).type
mtb.typing.type_map[float] = cast_floatX
mtb.typing.type_map['floatX'] = cast_floatX

# Add graph types to the list of types not to be casted as arrays
# GraphTypes is a dynamic property, so we add a function
mtb.typing.add_nonarray_type(lambda: shim.config.GraphTypes)
    # lambda: shim.config.__class__.GraphTypes.fget(shim.config))

mtb.typing.add_postponed_module(
    source_module="mackelab_toolbox.cgshim_postponed",
    target_module="mackelab_toolbox.typing",
    attrs=('FloatX', 'Symbolic', 'Shared', 'Tensor',
           'AnyRNG', 'RNGStream', 'RandomStateStream')
)
# It seems stupid to list the postponed module twice, but we do want types
# defined there to be accessible from cgshim
mtb.typing.add_postponed_module(
    source_module="mackelab_toolbox.cgshim_postponed",
    target_module="mackelab_toolbox.cgshim",
    attrs=('FloatX', 'Symbolic', 'Shared', 'Tensor',
           'AnyRNG', 'RNGStream', 'RandomStateStream')
)
