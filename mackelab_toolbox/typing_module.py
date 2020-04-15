"""
For almost all cases, you should not import this module directly but use

    import macklab_toolbox.typing

"""

import sys
import builtins
import importlib
import numpy as np
from collections.abc import Iterable, Callable
import mackelab_toolbox.utils as utils
import typing
from typing import Union, Type

############
# Postponed class instantiation

postponed_classes = {}
types_frozen = False

def PostponedClass(clsname: str, source_module: str, target_module: str):
    """
    Create a placeholder for a class depending on the dynamic types defined
    in `mackelab_toolbox.typing`. It provides a sensible error message if one
    attempts to access the class before it is initialized.

    Concept
    -------
    Classes with dynamic types are placed in postponed modules (Convention:
    `mymodule_postponed.py` for the postponed types of `mymodule`.)
    These modules are only imported when `freeze_types()` is called, thereby
    fixing the dynamic types.

    Parameters
    ----------
    clsname: str
        Name of the class being postponed. Must be defined in `source_module`.
    # factory: Callable  [() -> Type]
    #     Function which creates and returns the class. It will be called by
    #     `macklab_toolbox.typing.freeze_types()` and the result injected into
    #     the global namespace of the target module.
    source_module: str
        The name of the module where the class is defined. This module must
        not be imported, otherwise the definitions therein are already fixed.
    target_module: str
        The name of the module where we want to insert the class, as it would
        appear in `sys.modules`. This identifer `target_module.clsname` should
        correspond to that of the created placeholder class. (See example.)

    Example
    -------
    # mymodule.py
    >>> from pydantic import BaseModel
    >>> import mackelab_toolbox as mtb
    >>> MyClass = mtb.typing.PostponedClass('MyClass',
                                            'mymodule_postponed',
                                            'mymodule')
    # mymodule_postponed.py
    >>> from mymodule_postponed import *  # Treat _postponed as an inlined
    >>> class MyClass():                  # extension of `mymodule`
    >>>     x: mtb.typing.AnyScalarType   # Dynamic type requiring postponing
    >>> return MyClass
    At this point `mymodule.MyClass` is a placeholder, and attempting to
    access it will raise an error. It is essential that the 'mymodule'
    argument matches the key to `mymodule.py` in `sys.modules`.
    # script.py
    >>> import mymodule
    >>> # Imports and setup, e.g. with mackelab_toolbox.cgshim
    >>> mtb.typing.freeze_types()
    `freeze_types()` imports `mymodule_postponed`, thereby freezing the
    `AnyScalarType`. It then replaces `mymodule.MyClass` with the desired class,
    and from here on the class can be used. However dynamic types can no longer
    be modified.
    """
    msg = \
f"""
This is a placeholder for the class {clsname}, which depends on the
dynamic types defined in `mackelab_toolbox.typing`. Before using the
class you need to call `mackelab_toolbox.typing.freeze_types()`.
"""
    class PostponedClass:
        # __slots__ = ('source_module', 'target_module', 'target_name')
        __slots__ = ()
        def __init__(self, *args, **kwargs):
            raise RuntimeError(msg)
        def __getattr__(self, *args, **kwargs):
            raise RuntimeError(msg)
        def __getitem__(self, *args, **kwargs):
            raise RuntimeError(msg)
    PostponedClass.source_module = source_module
    PostponedClass.target_module = target_module
    PostponedClass.target_name = clsname
    PostponedClass.__name__ = clsname + "_placeholder"
    PostponedClass.__doc__  = msg

    if types_frozen:
        raise RuntimeError(f"Class {clsname} depends on dynamic types from "
                           "`mackelab_toolbox.typing`. It cannot be added "
                           "after types have be frozen with "
                           "`mackelab_toolbox.typing.freeze_types`.")
    if clsname in postponed_classes:
        raise RuntimeError(f"{clsname} was already added to the list of "
                           "postponed classes.")
    postponed_classes[clsname] = PostponedClass
    return PostponedClass

def freeze_types():
    """
    Call this function once definitions of dynamic types are finalized
    (in particular, after having called `theano_shim.load`), and before
    the actual code.
    """
    global types_frozen
    if types_frozen:
        logger.error("`mackelab_toolbox.typing.freeze_types()` was called "
                     "more than once.")
    postponed_modules = set(C.source_module for C in postponed_classes.values())
    too_soon = [m for m in postponed_modules if m in sys.modules]
    if len(too_soon) > 0:
        raise RuntimeError("The following modules contain dynamic types and "
                           "must not be loaded before those have been frozen: "
                           f"\n{too_soon}\n There is no need to import these "
                           "modules: it is done automatically by "
                           "`mtb.typing.freeze_types()`.")
    for m in postponed_modules:
        importlib.import_module(m)
    for C in postponed_classes.values():
        newC = getattr(sys.modules[C.source_module], C.target_name)
        setattr(sys.modules[C.target_module], C.target_name, newC)
    types_frozen = True

############
# Type Container

class TypeSet(set):
    def check_input(self, type):
        if isinstance(type, Iterable):
            for T in type:
                self.check_input(T)
        elif not isinstance(type, (builtins.type, Callable)):
            raise TypeError("`type` must either be a type or argument-less "
                            "callable.")
        elif not isinstance(type, builtins.type) and isinstance(type, Callable):
            T = type()
            if not ((isinstance(T, Iterable)
                     and all(isinstance(T_, builtins.type) for T_ in T))
                    or isinstance(type(), builtins.type)):
                raise TypeError("`type` must either be a type or argument-less "
                                "callable.")
        return

    def add(self, type):
        """
        Makes two changes to set.add:
          - Flattens iterable inputs
          - Ensures inputs are types (or arg-less callables returning types)
        """
        if isinstance(type, Iterable):
            for T in type:
                self.add(T)
        else:
            self.check_input(type)
            super().add(type)

    def update(self, types):
        self.check_input(types)
        super().update(types)

class TypeContainer(metaclass=utils.Singleton):
    """
    This class emulates a dynamic module, which is used as a collection of
    constants defining type collections. These constants can change, depending on whether
    e.g. a symbolic library is loaded through cgshim.
    Other modules can also add custom types to the defined collections.

    Access the unique class instance as `import mackelab_toolbox.typing`


    Attributes
    --------
    + type_map : dict
        Modules can update type_map as required
        For example, theano_shim adds the mapping from `float` type to `floatX`
        Mappings can be either types, or argument-less functions which return a type
    + AllNumericalTypes: tuple
        Anything which can be considered a number (incl. array, symbolic)
        Modules can add their own types to this list
        Use as
        >>> import mackelab_toolbox as mtb
        >>> Union[mtb.typing.AllNumericalTypes]`
        Value without additional imports:
            (int, float, DType[np.number], Array[np.number]
    + AllScalarTypes: tuple
        Similar to AllNumericalTypes, but restricted to scalars
        Use as
        >>> import mackelab_toolbox as mtb
        >>> Union[mtb.typing.AllScalarTypes]`
        Value without additional imports:
            (int, float, DType[np.number], Array[np.number, 0]
    + NotCastableToArray
        Modules can add types to `NotCastableToArray`
            mtb.typing.add_nonarray_type(mytype)
        to prevent `pydantic` from trying to cast them to a NumPy array

    Planned API changes
    ---------------
    Change interface from
        typing.add_scalar_type(mytype)
    to
        typing.AllScalarTypes.add(mytype)
    """
    # __slots__ = ('_AllNumericalTypes',
    #              '_AllScalarTypes',
    #              '_NotCastableToArrayTypes',
    #              'type_map',
    #              'json_encoders')

    # Reproduce public module functions, since we should only access the
    # typing module through this class.
    # TODO: There's probably a better way to do this
    @staticmethod
    def PostponedClass(*args, **kwargs):
        return PostponedClass(*args, **kwargs)
    @staticmethod
    def freeze_types():
        return freeze_types()

    def __init__(self):
        self._AllNumericalTypes = TypeSet([int, float])
        self._AllScalarTypes    = TypeSet([int, float])
        self._NotCastableToArrayTypes = TypeSet()
        self.type_map = {}
        self.json_encoders = {}
    @property
    def AnyNumericalType(self):
        return Union[tuple(T if isinstance(T, type) else T()
                           for T in self._AllNumericalTypes)]
    @property
    def AnyScalarType(self):
        return Union[tuple(T if isinstance(T, type) else T()
                           for T in self._AllScalarTypes)]
    @property
    def NotCastableToArray(self):
        return tuple(T if isinstance(T, type) else T()
                     for T in self._NotCastableToArrayTypes)

    def add_numerical_type(self, type):
        self._AllNumericalTypes.add(type)
    def add_scalar_type(self, type):
        self._AllScalarTypes.add(type)
    def add_nonarray_type(self, type):
        self._NotCastableToArrayTypes.add(type)

    ####################
    # Type normalization

    def convert_dtype(self, annotation_type):
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
        T = self.type_map.get(annotation_type, annotation_type)
        if not isinstance(T, type) and isinstance(T, Callable):
            T = T()
        if isinstance(T, type) and issubclass(T, _DTypeType):
            T = T.dtype
        return np.dtype(T)

typing = TypeContainer()

############
# Constants, sentinel objects

# Computed = utils.sentinel("Computed", "<computed>")
# def Computed():
#     return field(default=_COMPUTED_PARAMETER, repr=False, compare=False)


# class _NotCastableToArrayType(metaclass=utils.Singleton):
#     """
#     Modules can add types to `NotCastableToArray`
#         mtb.typing.NotCastableToArray.add(mytype)
#     to prevent `pydantic` from trying to cast them to a NumPy array
#     """
#     def __init__(self):
#         self._fixed_types = set()
#         self._callable_types = set()
#     def add(self, types):
#         if isinstance(types, Iterable):
#             for T in types:
#                 self.add(T)
#         else:
#             T = types
#             if isinstance(T, Callable):
#                 self._callable_types.add(T)
#             else:
#                 self._fixed_types.add(T)
#     def _types(self):
#         for T in self._fixed_types:
#             yield T
#         for T in self._callable_types:
#             yield T()
#     @property
#     def types(self):
#         return tuple(self._types())
# NotCastableToArray = _NotCastableToArrayType()
#

####################
# Custom Types for annotations / pydantic

####
# Slice type

class Slice:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v):
        if isinstance(v, Sequence) and v[0].lower() == 'slice':
            v = slice(*v[1])
        if not isinstance(v, slice):
            raise TypeError("Slice required.")
        return v
    @classmethod
    def __modify_schema__(cls, field_schema):
        """We need to tell pydantic how to export this type to the schema,
        since it doesn't know what to map the type to.
        """
        # See https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types
        # for the API, and https://pydantic-docs.helpmanual.io/usage/schema/
        # for the expected fields
        field_schema.update(
            type="array",
            description="('slice', [START], STOP, [STEP])"
            )
    @staticmethod
    def json_encoder(v):
        """Attach this method to a pydantic BaseModel as follows:
        >>> class MyModel(BaseModel):
        >>>     class Config:
        >>>         json_encoders = {
        >>>             slice: Slice.json_encoder
        >>>         }
        """
        if v.step is None:
            args = (v.start, v.stop)
        else:
            args = (v.start, v.stop, v.step)
        return ("slice", args)

typing.Slice = Slice

####
# DType type

class _DTypeType(np.generic):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value, field):
        if isinstance(value, np.ndarray):
            # Allow scalar arrays
            if value.ndim==0:
                value = value[()]
            else:
                raise ValueError(f"Field {field.name} expects a scalar, not "
                                 f"an array.\nProvided value: {value}.")
        # Don't cast unless necessary
        # Issubdtype allows specifying abstract dtypes like 'number', 'floating'
        # np.generic ensures isubdtype doesn't let through non-numpy types
        # (like 'float'), or objects which wrap numpy types (like 'ndarray').
        if (isinstance(value, np.generic)
            and np.issubdtype(type(value), cls.dtype.type)):
            return value
        elif np.can_cast(value, cls.dtype):
            return cls.dtype.type(value)
        else:
            raise TypeError(f"Cannot safely cast '{field.name}' type  "
                            f"({type(value)}) to type {cls.dtype}.")
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type='number')


class _DTypeMeta(type):
    def __getitem__(self, dtype):
        dtype=typing.convert_dtype(dtype)
        return type(f'DType[{dtype}]', (_DTypeType,),
                    {'dtype': typing.convert_dtype(dtype)})

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

typing.DType = DType
typing.add_numerical_type(DType[np.number])
typing.add_scalar_type(DType[np.number])

####
# Array type

class _ArrayType(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, value, field):
        if isinstance(value, typing.NotCastableToArray):
            raise TypeError(f"Values of type {type(value)} cannot be casted "
                             "to a numpy array.")
        if isinstance(value, np.ndarray):
            # Don't create a new array unless necessary
            if cls._ndim  is not None and value.ndim != cls._ndim:
                raise TypeError(f"{field.name} expects a variable with "
                                f"{cls._ndim} dimensions.")
            # Issubdtype allows specifying abstract dtypes like 'number', 'floating'
            if np.issubdtype(value.dtype, cls.dtype):
                result = value
            elif np.can_cast(value, cls.dtype):
                result = value.astype(cls.dtype)
            else:
                raise TypeError(f"Cannot safely cast '{field.name}' type  "
                                f"({value.dtype}) to type {cls.dtype}.")
        else:
            result = np.array(value)
            # Issubdtype allows specifying abstract dtypes like 'number', 'floating'
            if np.issubdtype(result.dtype, cls.dtype):
                pass
            elif np.can_cast(result, cls.dtype):
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

    @classmethod
    def __modify_schema__(cls, field_schema):
        # FIXME: Figure out how to use get schema of subfield
        field_schema.update(type ='array',
                            items={'type': 'number'})
    @classmethod
    def json_encoder(cls, v):
        """Attach this method to a pydantic BaseModel as follows:
        >>> class MyModel(BaseModel):
        >>>     class Config:
        >>>         json_encoders = {
        >>>             Array: Array.json_encoder
        >>>         }
        """
        return v.tolist()

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
        dtype=typing.convert_dtype(T)
        specifier = str(dtype)
        if ndim is not None:
            specifier += f",{ndim}"
        return type(f'Array[{specifier}]', (_ArrayType,),
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

typing.Array = Array
typing.add_numerical_type(Array[np.number])
typing.add_scalar_type(Array[np.number, 0])
typing.json_encoders.update({np.ndarray: _ArrayType.json_encoder})