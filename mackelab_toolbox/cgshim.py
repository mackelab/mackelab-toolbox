"""
Tie-in to the theano_shim module (which I plan on renaming to `cgshim`
if/when it gets extended to other computational graph libraries)

Provides
--------
  - cast_floatX: Callable
    Function which casts to current type of 'floatX'
  - types: interface for symbolic types for type hinting/annotations
    Has the elements:
    + FloatX: retrieved as `cgshim.types.FloatX`
      A type which maps to `.typing.DType[shim.config.floatX]`.
    + Symbolic: retrieved as `cgshim.types.Symbolic`
      A type, compatible with pydantic, to indicate a symbolic value in
      annotations. Which type that corresponds to depends on the currently
      loaded symbolic library.
    + Shared: retrieved as `cgshim.types.Shared`
      A type, compatible with pydantic, to indicate a shared value in
      annotations. Which type that corresponds to depends on the currently
      loaded symbolic library.
    + Tensor: retrieved as `cgshim.types.Tensor`
      A type, compatible with pydantic, corresponding to the union of
      Array, Symbolic and Shared types. (In other words, allowing both Numpy
      and symbolic inputs.)

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
# import mackelab_toolbox.typing  # Call as mtb.typing to avoid confusing
import mackelab_toolbox.utils as utils
import mackelab_toolbox.transform
# # Any module using types must be loaded at the very end, after we've updated
# # mackelab_toolbox.typing
# # FIXME: It would be REALLY nice if the types could be reliably dynamic
# #        instead of relying on import order
# too_soon = [m for m in sys.modules if m in
#             ['mackelab_toolbox.transform']]
# if len(too_soon) > 0:
#     raise ImportError("The following modules must be loaded after `cgshim`, "
#                       "otherwise they won't recognize symbolic types:\n"
#                       f"{too_soon}.")

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

import logging
logger = logging.getLogger(__file__)

# FIXME: Redundant with mtb.typing.TypeContainer
class TypeContainer(metaclass=utils.Singleton):
    def __init__(self):
        self.__validation_types = {}

    @property
    def FloatX(self):
        return mtb.typing.DType[shim.config.floatX]

    @property
    def Symbolic(self):
        """
        Return a type-hint compatible symbolic type.
        The type returned depends on which symbolic library (if any) is loaded.
        Types are cached to avoid recreating the same type more than once.
        """
        if shim.config.library == 'numpy':
            return mtb.typing.Array
        else:
            return self.get_type(shim.config.SymbolicType,
                                 'Symbolic', 'symbolic')

    @property
    def Shared(self):
        """
        Return a type-hint compatible symbolic shared type.
        The type returned depends on which symbolic library (if any) is loaded.
        Types are cached to avoid recreating the same type more than once.
        """
        if shim.config.library == 'numpy':
            return mtb.typing.Array
        else:
            return self.get_type(shim.config.SymbolicSharedType,
                                 'Shared', 'shared')

    @property
    def Tensor(self):
        """
        Return a Union type-hint compatible with any of Array, Symbolic or Shared.

        NOTE: For shared variables does not check that data is actually a
              a tensor. Should we do this ?
        """
        types = tuple(set((mtb.typing.Array, self.Symbolic, self.Shared)))
            # set to remove duplicates
        if types not in self.__validation_types:
            class ShimUnion:
                def __getitem__(self, args):
                    return Union[tuple(T[args] for T in types)]
            self.__validation_types[types] = ShimUnion()
        return self.__validation_types[types]

    def get_type(self, baseT, Tname, namedesc):
        """
        Parameters
        ----------
        baseT: type
            The type for which we want a type hint.
        Tname: str
            The name to assign to the type ("type name"). Normally capitalized.
        namedesc: str
            Name to use in error messages to describe the type. Normally not
            capitalized.
            E.g. setting `namedesc` to "symbolic", an error message may read
            "This field expects a symbolic variable."
        """
        # Structure:
        #   - Check __validation_types cache. If symbolic type is already there,
        #     skip everything and jump to the return.
        #   - Otherwise, create the typed base class TypedSymbolic, a metaclass
        #     SymbolicMeta which returns a subclass of TypedSymbolic, and a
        #     Symbolic class which has SymbolicMeta as metaclass
        #     + Store Symbolic in the cache
        #   - Return the value from the cache
        # The metaclass logic here, like the one in mtb.typing.Array, is based
        # on a suggestion by the pydantic developer:
        # https://github.com/samuelcolvin/pydantic/issues/380#issuecomment-594639970
        if baseT not in self.__validation_types:

            # For symbolic types, we just assume that val has already been properly casted.
            # We do this to keep the reference to the original variable
            class ValidatingType(baseT):
                @classmethod
                def __get_validators__(cls):
                    yield cls.validate_type

                @classmethod
                def validate_type(cls, value, field):
                    if not isinstance(value, baseT):
                        raise TypeError(f"{field.name} expects a {namedesc} variable.")
                    if cls._ndim  is not None and value.ndim != cls._ndim:
                        raise TypeError(f"{field.name} expects a variable with "
                                        f"{cls._ndim} dimensions.")
                    if getattr(value, 'name', None) is None:
                        value.name = field.name
                    if not np.can_cast(value.dtype, field.type_):
                        logger.error(
                            f"{field.name} expects a variable of type {field.type_}. "
                            f"Provided value has type {value.dtype}, which is not a "
                            "permitted cast. Note that for symbolic variables, we "
                            "perform no casting, so types must match exactly.")
                    elif not np.dtype(value.dtype) is np.dtype(field.type_):
                        logger.warning(
                            f"{field.name} expects a variable of type {field.type_}. "
                            f"Provided value has type {value.dtype}. "
                            "Note that for symbolic variables, we perform "
                            "no casting, so types must match exactly.")
                    return value

            class Meta(baseT.__class__):
                def __getitem__(self, args):
                    if isinstance(args, tuple):
                        T = args[0]
                        ndim = args[1] if len(args) > 1 else None
                        extraargs = args[2:]  # For catching errors only
                    else:
                        T = args
                        ndim = None
                        extraargs = []
                    if (not isinstance(T, type) or len(extraargs) > 0
                        or not isinstance(ndim, (int, type(None)))):
                        raise TypeError(
                            f"`{Tname}` must be specified as either "
                            f"`{Tname}[T]` or `{Tname}[T, n], where `T` is a "
                            "type and `n` is an int. (received: "
                            f"[{', '.join((str(a) for a in args))}]).")
                    return type('Symbolic', (ValidatingType,),
                                {'dtype': mtb.typing.convert_dtype(T),
                                 '_ndim': ndim})

            self.__validation_types[baseT] = Meta(Tname, (baseT,), {})

        return self.__validation_types[baseT]

typing = TypeContainer()
mtb.typing.add_numerical_type(
    (lambda: typing.Symbolic[np.number],
     lambda: typing.Shared[np.number]))
mtb.typing.add_scalar_type(
    (lambda: typing.Symbolic[np.number,0],
     lambda: typing.Shared[np.number,0]))
