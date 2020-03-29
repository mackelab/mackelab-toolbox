
#########################
# @parameterized decorator
#########################
from collections import deque
from collections.abc import Iterable, Callable
import logging
logger = logging.getLogger(__file__)
import typing
from typing import Any, List
from dataclasses import dataclass, field
import numpy as np
import nptyping  # Stop-gap measure until types are officially supported by NumPy

import mackelab_toolbox.utils as utils

############
# Constants, sentinel objects

class _COMPUTED_PARAMETER_TYPE(metaclass=utils.SentinelMeta):
    pass
_COMPUTED_PARAMETER = _COMPUTED_PARAMETER_TYPE()
def Computed():
    return field(default=_COMPUTED_PARAMETER, repr=False, compare=False)

# Does DoNotCast provide anything more than just returning the value ?
# I don't think so
# class _DoNotCastT(metaclass=utils.SentinelMeta):
#     pass
# DoNotCast = _DoNotCastT()

# Custom type casting functions can be written and added to `cast_functions`
# (see mackelab_toolbox.cgshim.cast_symbolic for an example)
# Functions may support only certain values or types; for any unsupported
# arguments, they should return `NotImplemented`.
# Functions may return `DoNotCast` to prevent further attempts at casting and
# return the value as is. This is useful if the casted type is not a proper
# subtype of the target type; see `cast_array` for an example.
cast_functions = deque()

# Modules can update type_map as required
# For example, theano_shim adds the mapping from `float` type to `floatX`
# Mappings can be either types, or argument-less functions which return a type
type_map = {}

############
# Decorator definition

def parameterized(cls=None, **kwargs):
    """
    Build off the @dataclasses.dataclass decorator to construct an initializer
    with type-casting from class type annotations.

    Examples
    --------
    >>> from mackelab_toolbox.parameters import parameterized
    >>> @parameterized
        class Model:
            a: float
            dt: float=0.01
            def integrate(self, x0, T):
                x = x0; y=0
                a = self.a; dt = self.dt
                for t in np.arange(0, T, self.dt):
                    x += y*dt; y+=a*x*dt
                return x
    >>> m = Model(a=-0.5)
    >>> m
        "Model(a=-0.5, dt=0.01)"
    >>> m.integrate(1, 5.)
    """
    if cls is None:
        # We are being called as @parameterized()
        def wrap(cls):
            return parameterized(cls, **kwargs)
        return wrap
    else:
        # We are being called as @parameterized
        newname = cls.__name__ + '_Parameterized'
        newcls = parameterized.created_types.get(newname, None)
        ###############################
        # This hacky solution to using `Computed` to override attributes
        # could be replaced by just giving them a placeholder default, once
        # the discussion https://bugs.python.org/issue36077 is resolved.
        ###############################
        # Remove from __annotations__ the Computed attributes, before
        # `dataclass` sees them.
        if newcls is None:
            newcls = type(newname, (dataclass(cls),),
                          {'__post_init__': _parameterized_post_init})
            parameterized.created_types[newname] = newcls
        return newcls
parameterized.created_types = {}

def _parameterized_post_init(self):
    computed_fields = deque()
    for name,field in self.__dataclass_fields__.items():
        T = convert_type(field.type)
        v = getattr(self, name)
        if v is _COMPUTED_PARAMETER:
            computed_fields.append((name,T))
        elif not (isinstance(v, T.type) if isinstance(T, TypeFunction)
                  else isinstance(v, T)):
            setattr(self, name, cast(v, T, name))
    # Loop over bases in reverse order and execute "__init_computed__" methods
    bases = type(self).mro()
    for base in bases[::-1]:
        ic = getattr(base, '__init_computed__', None)
        if ic is not None:
            ic(self)
    for name,T in computed_fields:
        v = getattr(self, name)
        if v is _COMPUTED_PARAMETER:
            raise RuntimeError(f"Parameter {name} is designated as a computed "
                               f"parameter, but is not set by class {type(self)}'s "
                               "'__init_computed__()', nor that of its parents.")
        casted = cast(v, T, name)
        if casted is not v:  # `cast` did something, so update the attribute
            setattr(self, name, casted)

#############
# Functions to deal with type-casting

@dataclass
class TypeFunction:
    """
    Parameters
    ----------
    cast_function: Callable
        Function taking a single argument and returning it, casted to
        the designated type.
    type: type
        Type of the result of `cast_function`
    """
    cast_function: typing.Callable[[Any], Any]
    type: type
    def __call__(self, x):
        return self.cast_function(x)
    def __str__(self):
        return f"TypeFunction({str(self.type)})"
    def __repr__(self):
        return f"TypeFunction({repr(self.cast_function)}, {repr(self.type)})"

def cast(value, type, name):
    """
    Try every member of `cast_functions` in order, and stop after the
    first one returning something else than `NotImplemented`.

    Some casting functions (e.g. cgshim's `cast_symbolic`) require a `name`
    attribute. We require all functions to accept that argument, to keep
    signatures consistent.
    """
    if isinstance(type, TypeFunction):
        type = type.type
    if isinstance(value, type) or type is Any:
        # `Any` type can be used to short-circuit type-casting functions.
        return value
    for castfn in cast_functions:
        result = castfn(value, type, name)
        if result is not NotImplemented:
            return result
        # if result is DoNotCast:
        #     return result
    raise TypeError(f"The variable '{name}' (value: '{value}') could not "
                    f"be casted to the type '{type}'.")

def cast_with_type(value, type, name):
    """
    The default casting function: we try to call `type(value)`.
    This should generally be last in the `cast_functions` list.
    """
    if isinstance(type, Callable):
        return type(value)
    else:
        return NotImplemented
cast_functions.appendleft(cast_with_type)

def cast_array(value, type, name):
    if not issubclass(type, nptyping.Array):
        return NotImplemented
    assert not isinstance(type.generic_type, Iterable)
        # `convert_type` should already have prevented situations with
        # multiple types for arrays
    target_dtype = np.dtype(type.generic_type)
    if isinstance(value, np.ndarray):
        # Value is already an ndarray; this means we can use the specialized
        # NumPy methods for casting, but it also means that there are many
        # possible types for `value`, and in particular higher precision types
        # for which it would be unsafe to cast (e.g. float64 to float32).
        if value.dtype == target_dtype:
            result = value
        elif np.can_cast(value.dtype, target_dtype):
            result = value.astype(target_dtype)
        else:
            raise TypeError(f"Cannot safely cast '{name}' (type {value.dtype}) "
                            f"to type {target_dtype}.")
    else:
        if hasattr(value, 'astype'):
            result = value.astype(str(dtype))  # Theano only accepts str dtype
        else:
            result = np.array(value, dtype=target_dtype)
    if not isinstance(result, type):
        r = "…" if type.rows is Ellipsis else getattr(type, 'rows', None)
        c = "…" if type.cols is Ellipsis else getattr(type, 'cols', None)
        raise TypeError(f"Array {result} does not match the expected "
                        f"format Array[{type.generic_type},{r},{c}]")
    return result

cast_functions.appendleft(cast_array)

def convert_type(annotation_type):
    # TODO: Combine common code with smttask.types.cast
    """
    Look up the type in type_map and see if it should be replaced.
    For example, we could define `float` to be replaced by `np.float32`.
    This is mostly meant to keep parameter definitions as clean and compact
    as possible.
    Also performs conversions to convert type hints into castable types.

    Most of the functionality would probably be better implemented by creating
    custom types (c.f. `typing` module), but for now this solution is more
    flexible and easier for me to wrap my head around.

    Special cases:
    --------------
    Callback types:
        A value in `type_map` may be specified as an argument-less function,
        in which case it will be called and the return value used as a type.
        This can be used to specify a type which may change during runtime;
        for instance, Theano's 'floatX' type.
    Array types:
        Array types from the nptyping module are recognized, and in this case
        we return a function which performs a cast

    Returns
    -------
    type or TypeFunction instance
        Returns a type T, such that `T(x)` will cast `x` as an instance of T.
        For some types (such as ndarray), one should not use the type directly
        but a helper function to cast the type. In this case we return a
        TypeFunction which performs the cast, similarly to the original helper
        function, and which also stores the expected type of the result.
    """
    T = type_map.get(annotation_type, annotation_type)
    if not isinstance(T, type) and isinstance(T, Callable):
        T = T()
    if isinstance(T, str):
        T = np.dtype(T)
    elif issubclass(T, nptyping.Array):
        Ttype = T.generic_type
        # if isinstance(Ttype, np.dtype):
        #     dtype = Ttype
        # elif isinstance(Ttype, str):
        #     dtype = np.dtype(Ttype)
        if isinstance(Ttype, Iterable):
            raise TypeError("Array parameters should have a unique type. This "
                            f"type hint defines multiple: {Ttype}")
        Ttype = convert_type(Ttype)
            # Allow normal type specifier to be used inside Array[]
        # else:
        #     # Any valid specifier for `dtype` should be accepted.
        #     dtype = np.dtype(Ttype)
        T = nptyping.Array[Ttype, T.rows, T.cols]
        # typehint = T if not isinstance(T, TypeFunction) else T.type
        #     # Assign to typehint outside closure before def of T is changed
        # def _cast(x):
        #     if hasattr(x, 'astype'):
        #         result = x.astype(str(dtype))  # Theano only accepts str dtype
        #     else:
        #         result = np.array(x, dtype=dtype)
        #     if not isinstance(result, typehint):
        #         r = "…" if typehint.rows is Ellipsis else getattr(typehint, 'rows', None)
        #         c = "…" if typehint.cols is Ellipsis else getattr(typehint, 'cols', None)
        #         raise TypeError(f"Array {result} does not match the expected "
        #                         f"format Array[{typehint.generic_type},{r},{c}]")
        #     return result
        # T = TypeFunction(_cast, dtype.type)
    elif issubclass(T, List):
        accepted_types = typish.get_args(T)
        if accepted_types == ():
            # No specified types
            def _cast(x):
                return list(x)
            T = TypeFunction(_cast, list)
        else:
            # Cast to types, in the order they are provided
            accepted_types = [convert_type(_T) for _T in accepted_types]
                # List element types themselves might need to be converted
            result_types = [_T.type if isinstance(_T, TypeFunction) else _T
                            for _T in accepted_types]
            def subcast(x):
                if isinstance(x, result_types):
                    return x
                for _T in accepted_types:
                    try:
                        result = _T(x)
                    except (ValueError, TypeError):
                        pass
                    else:
                        return result
                raise TypeError(f"Unable to cast {x} to any of {accepted_types}.")
            def _cast(x):
                return [subcast(_x) for _x in x]
            T = TypeFunction(_cast, list)
    return T
