import numpy as np
from collections.abc import Iterable, Callable
import mackelab_toolbox.utils as utils

############
# Constants, sentinel objects

Computed = utils.sentinel("Computed", "<computed>")
# def Computed():
#     return field(default=_COMPUTED_PARAMETER, repr=False, compare=False)

# Modules can update type_map as required
# For example, theano_shim adds the mapping from `float` type to `floatX`
# Mappings can be either types, or argument-less functions which return a type
type_map = {}

class _NotCastableToArrayType(metaclass=utils.Singleton):
    """
    Modules can add types to `NotCastableToArray`
        mtb.typing.NotCastableToArray.add(mytype)
    to prevent `pydantic` from trying to cast them to a NumPy array
    """
    def __init__(self):
        self._fixed_types = set()
        self._callable_types = set()
    def add(self, types):
        if isinstance(types, Iterable):
            for T in types:
                self.add(T)
        else:
            T = types
            if isinstance(T, Callable):
                self._callable_types.add(T)
            else:
                self._fixed_types.add(T)
    def _types(self):
        for T in self._fixed_types:
            yield T
        for T in self._callable_types:
            yield T()
    @property
    def types(self):
        return tuple(self._types())
NotCastableToArray = _NotCastableToArrayType()

####################
# Type normalization

def convert_dtype(annotation_type):
    # TODO: Combine common code with smttask.types.cast
    """
    Look up the type in type_map and see if it should be replaced.
    For example, one can define `float` to be replaced by `np.float32`.
    This keeps parameter definitions as clean and compact as possible,
    helps normalize types to a reduced standard subset, and
    allows modules to define new type mappings (for example theano_shim's
    dynamic `floatX` type).

    Special cases:
    --------------
    Callback types:
        A value in `type_map` may be specified as an argument-less function,
        in which case it will be called and the return value used as a type.
        This can be used to specify a type which may change during runtime;
        for instance, Theano's 'floatX' type.

    Returns
    -------
    type
        Returns a type T, such that `T(x)` will cast `x` as an instance of T.
        For some types (such as ndarray), one should not use the type directly
        but a helper function to cast the type. In this case we return a
        TypeFunction which performs the cast, similarly to the original helper
        function, and which also stores the expected type of the result.
    """
    T = type_map.get(annotation_type, annotation_type)
    if not isinstance(T, type) and isinstance(T, Callable):
        T = T()
    if isinstance(T, type) and issubclass(T, _DTypeType):
        T = T.dtype
    return np.dtype(T)

####################
# Custom Types for annotations / pydantic

####
# DType type

class _DTypeType(np.generic):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type
    @classmethod
    def validate_type(cls, value, field):
        # Don't cast unless necessary
        if type(value) is cls.dtype.type:
            return value
        elif np.can_cast(type(value), cls.dtype):
            return cls.dtype.type(value)
        else:
            raise TypeError(f"Cannot safely cast '{field.name}' type  "
                            f"({type(value)}) to type {cls.dtype}.")
class _DTypeMeta(type):
    def __getitem__(self, dtype):
        dtype=convert_dtype(dtype)
        return type(f'DType[{dtype}]', (_DTypeType,),
                    {'dtype': convert_dtype(dtype)})

class DType(np.generic, metaclass=_DTypeMeta):
    """
    Use this to use a NumPy dtype for type annotation; `pydantic` will
    recognize the type and execute appropriate validation/parsing.

    This may become obsolete, or need to be updated, when NumPy officially
    supports type hints (see https://github.com/numpy/numpy-stubs).

    - `DType[T]` specifies an object to be casted with dtype `T`. Any
       expression for which `np.dtype(T)` is valid is accepted.

    Example
    -------
    >>> from pydantic.dataclasses import dataclass
    >>> from mackelab_toolbox.typing import DType
    >>>
    >>> @dataclass
    >>> class Model:
    >>>     x: DType[np.float64]
    >>>     y: DType[np.int8]

    """
    pass


####
# Array type

class _ArrayType(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, value, field):
        if isinstance(value, NotCastableToArray.types):
            raise TypeError(f"Values of type {type(value)} cannot be casted "
                             "to a numpy array.")
        if isinstance(value, np.ndarray):
            # Don't create a new array unless necessary
            if cls._ndim  is not None and value.ndim != cls._ndim:
                raise TypeError(f"{field.name} expects a variable with "
                                f"{cls._ndim} dimensions.")
            if value.dtype == cls.dtype:
                result = value
            elif np.can_cast(value.dtype, cls.dtype):
                result = value.astype(cls.dtype)
            else:
                raise TypeError(f"Cannot safely cast '{field.name}' type  "
                                f"({value.dtype}) to type {cls.dtype}.")
        else:
            result = np.array(value)
            if np.can_cast(result.dtype, cls.dtype):
                if cls._ndim is not None and result.ndim != cls._ndim:
                    raise TypeError(
                        f"The shape of the data ({result.shape}) does not " "correspond to the expected of dimensions "
                        f"({cls._ndim} for '{field.name}').")
                elif result.dtype != cls.dtype:
                    result = result.astype(cls.dtype)
            else:
                raise TypeError(f"Cannot1 safely cast '{field.name}' (type  "
                                f"{result.dtype}) to type {cls.dtype}.")
        return result

class _ArrayMeta(type):
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
                "`Array` must be specified as either `Array[T]`"
                "or `Array[T, n], where `T` is a type and `n` is an int. "
                f"(received: {', '.join((str(a) for a in args))}]).")
        dtype=convert_dtype(T)
        return type(f'Array[{dtype}]', (_ArrayType,),
                    {'dtype': dtype, '_ndim': ndim})

class Array(np.ndarray, metaclass=_ArrayMeta):
    """
    Use this to specify a NumPy array type annotation; `pydantic` will
    recognize the type and execute appropriate validation/parsing.

    This may become obsolete, or need to be updated, when NumPy officially
    supports type hints (see https://github.com/numpy/numpy-stubs).

    - `Array[T]` specifies an array with dtype `T`. Any expression for which
      `np.dtype(T)` is valid is accepted.
    - `Array[T,n]` specifies an array with dtype `T`, that must have exactly
      `n` dimensions.

    Example
    -------
    >>> from pydantic.dataclasses import dataclass
    >>> from mackelab_toolbox.typing import Array
    >>>
    >>> @dataclass
    >>> class Model:
    >>>     x: Array[np.float64]      # Array of 64-bit floats, any number of dimensions
    >>>     v: Array['float64', 1]    # 1-D array of 64-bit floats


    """
    pass
