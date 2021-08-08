"""
For almost all cases, you should not import this module directly but use

    import macklab_toolbox.typing
"""

import sys
from textwrap import dedent
import builtins
import importlib
import numbers
import numpy as np
import abc
from types import SimpleNamespace
from functools import lru_cache
import typing
from typing import Union, Type as BuiltinType
from typing import Iterable, Callable, Sequence, Collection, Mapping, List
from collections import namedtuple

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from mackelab_toolbox import utils
from mackelab_toolbox.units import UnitlessT

# For Array
import io
import blosc
import base64

import logging
logger = logging.getLogger(__name__)

############
# JSON encoder with optional keywords
# (copied from RenormalizedFlows)

from pydantic.json import custom_pydantic_encoder
from functools import partial
from inspect import signature

def json_kwd_encoder(obj, **kwds):
    """
    Extension of the Pydantic encoders which allows kwargs.
    This is used to optionally skip encoding the large random state of
    distributions.
    """
    if kwds:
        # Loop through the encoders and bind arguments to each encoder
        # Alternative 1: Only keep encoders with keyword args matching `kwds`
        # Alternative 2: Re-implement `custom_pydantic_encoder` here and simply
        #    pass on `kwds`. This would avoid all unnecessary creation of partials
        _json_encoders = {}
        for T, encoder in typing.json_encoders.items():
            sig = signature(encoder)
            # Only bind arguments valid for that encoder
            _kwds = {k:v for k,v in kwds.items() if k in sig.parameters}
            _json_encoders[T] = partial(encoder, **_kwds)
    else:
        _json_encoders = typing.json_encoders
    return custom_pydantic_encoder(_json_encoders, obj)

############
# Postponed class and module

postponed_modules = set()
postponed_classes = {}
types_frozen = False

@dataclass(frozen=True)
class PostponedModule:
    """
    Both `source_module` and `target_module` must be python modules.
    When `freeze_types` is called, the list of attributes `attrs` is
    retrieved from `source_module` and injected into `target_module`.
    """
    source_module: str
    target_module: str
    attrs: tuple

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
    msg = dedent(f"""
        This is a placeholder for the class {clsname}, which depends on
        dynamic types (i.e. types which depend on runtime variables, or "
        "whether another library is loaded). Before using the
        class you need to call `mackelab_toolbox.typing.freeze_types()`.
        """)
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
    Postponed modules are imported in the following order:
        - First all postponed modules, in the order they were added.
        - Then all modules for postponed classes, in the order they were added.
    """
    global types_frozen
    if types_frozen:
        logger.error("`mackelab_toolbox.typing.freeze_types()` was called "
                     "more than once.")
    # The order can matter (e.g. transform_postponed depends on being loaded
    # after AnyNumericalType is frozen), so we use dictionaries to emulate
    # ordered sets.
    cls_postponed_modules = {C.source_module:None for C in postponed_classes.values()}
    mod_postponed_modules = {m.source_module:None for m in postponed_modules}
    all_postponed_modules = {**mod_postponed_modules, **cls_postponed_modules}
    loaded_postponed_modules = [m for m in all_postponed_modules if m in sys.modules]
    if len(loaded_postponed_modules) > 0:
        raise RuntimeError("The following modules contain dynamic types and "
                           "must not be loaded before those have been frozen: "
                           f"\n{loaded_postponed_modules}\n There is no need "
                           "to import these modules: it is done automatically "
                           "within `mtb.typing.freeze_types()`.")
    for m in all_postponed_modules:
        importlib.import_module(m)
    for C in postponed_classes.values():
        newC = getattr(sys.modules[C.source_module], C.target_name)
        setattr(sys.modules[C.target_module], C.target_name, newC)
    for mdesc in postponed_modules:
        source = sys.modules[mdesc.source_module]
        target = sys.modules[mdesc.target_module]
        for attr in mdesc.attrs:
            if hasattr(target, attr):
                raise RuntimeError(
                    f"Attempted to import the postponed attribute '{attr}' "
                    f"from module '{mdesc.source_module}' to module "
                    f"'{mdesc.target_module}', but it is already present in "
                    f"{mdesc.target_module}.")
            elif hasattr(target, attr):
                raise RuntimeError(
                    f"Attempted to import the postponed attribute '{attr}' "
                    f"from module '{mdesc.source_module}' to module "
                    f"'{mdesc.target_module}', but it is not present in "
                    f"{mdesc.source_module}.")
            else:
                setattr(target, attr, getattr(source, attr))
    types_frozen = True

############
# Type Container

class TypeSet(dict):
    """
    Effectively an order-preserving set, with input checking and flattening.
    Note that the iterator returns elements in the reverse order in which they
    were added, so that later additions have precedence for type resolution.
    """
    # Although this acts as a set (no duplicate elements), we subclass dict
    # because preserving order is important (Pydantic attempts coercions in order)
    # Dict values are all assigned 'None' and discarded
    def __init__(self, iterable=()):
        for T in iterable:
            self.check_input(T)
            self[T] = None

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
        Makes two changes compared to set.add:
          - Flattens iterable inputs
          - Ensures inputs are types (or arg-less callables returning types)
        """
        if isinstance(type, Iterable):
            for T in type:
                self.add(T)
        else:
            self.check_input(type)
            self[type] = None

    def update(self, types):
        self.check_input(types)
        super().update({T:None for T in types})

    def __iter__(self):
        return iter(list(self.keys())[::-1])

# @lru_cache
# def any_type(type_list):
#     # https://github.com/samuelcolvin/pydantic/issues/1423#issuecomment-618962827
#     """
#     Return a type that behaves like ``Union[*type_list]`` but for which
#     Pydantic won't perform any coercion if the input matches one one of the types.
#     If none of the types match, than coercion is performed as usual, in the
#     order of elements in `type_list`.
#     """
#     # FIXME: Symbolics inspect the field name to assign the variable name;
#     #        this breaks that
#     @dataclass
#     class ModelValidator:
#         v: Union[tuple(type_list)]
#     class AnyType:
#         @classmethod
#         def __get_validators__(cls):
#             yield cls.validate
#         @classmethod
#         def validate(cls, value):
#             if any(isinstance(value, T) for T in cls.type_list):
#                 return value
#             else:
#                 return cls.ModelValidator(v=value).v
#     AnyType.type_list = type_list
#     AnyType.ModelValidator = ModelValidator
#     return AnyType
def any_type(type_list):
    return Union[tuple(type_list)]

JsonEncoder = namedtuple("JsonEncoder", ["encoder", "priority"])

class TypeContainer(metaclass=utils.Singleton):
    """
    This class emulates a dynamic module, which is used as a collection of
    constants defining type collections. These constants can change, depending on whether
    e.g. a symbolic library is loaded through cgshim.
    Other modules can also add custom types to the defined collections.

    Access the unique class instance as `import mackelab_toolbox.typing`

    Pydantic-compatible types
    -------------------------
    + Builtins:
      - Range
      - Slice
      - Sequence (Union of [Range, typing.Sequence]; prevents Pydantic coercing range into list)
    + Numbers:
      - Number (validation only)
      - Integral (validation only)
      - Real (validation only, except int -> float)
    + Quantities with units
      - PintValue
      - PintUnit
      - QuantitiesValue
      - QuantitiesUnit
      - QuantitiesDimension
    + NumPy types:
      - DType
      - NPValue   (any numpy scalar != 'object'; use as `NPValue[np.float64]`)
      - Array     (Use as `Array[np.float64]`, or `Array[np.float64,2]`)
      - RNGenerator  (new style numpy RNG)
      - RandomState  (old style numpy RNG)
    + Scipy types:
      - Distribution (objects from scipy.stats)

    Interaction with modules for physical quantities (Pint and Quantities)
    ------------------------

    If either the `pint` or `quantities` modules are loaded before
    `typing.freeze_types()` is called, the corresponding json_encoder is
    added to `typing.json_encoders`. Alternatively, the functions
    `typing.load_pint` or `typing.load_quantities` can be executed at any time
    to add the corresponding json encoders.

    Attributes
    --------
    - type_map : dict
        Modules can update type_map as required
        For example, theano_shim adds the mapping from `float` type to `floatX`
        Mappings can be either types, or argument-less functions which return a type
    - AllNumericalTypes: tuple
        Anything which can be considered a number (incl. array, symbolic)
        Modules can add their own types to this list
        Use as
        >>> import mackelab_toolbox as mtb
        >>> Union[mtb.typing.AllNumericalTypes]`
        Value without additional imports:
            (int, float, NPValue[np.number], Array[np.number]
    - AllScalarTypes: tuple
        Similar to AllNumericalTypes, but restricted to scalars
        Use as
        >>> import mackelab_toolbox as mtb
        >>> Union[mtb.typing.AllScalarTypes]`
        Value without additional imports:
            (int, float, NPValue[np.number], Array[np.number, 0]
    - NotCastableToArray
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
    def add_postponed_module(*args, **kwargs):
        postponed_modules.add(PostponedModule(*args, **kwargs))
    @staticmethod
    def freeze_types():
        if 'pint' in sys.modules:
            typing.load_pint()
        if 'quantities' in sys.modules:
            typing.load_quantities()
        return freeze_types()
    @property
    def types_frozen(self):
        return types_frozen
    @property
    def json_kwd_encoder(self):
        return json_kwd_encoder
    @property
    def stat_modules(self):
        return stat_modules

    def __init__(self):
        self._AllNumericalTypes = TypeSet([int, float])
        self._AllScalarTypes    = TypeSet([int, float])
        self._AllUnitTypes      = TypeSet([UnitlessT])
        self._NotCastableToArrayTypes = TypeSet()
        self.type_map = {}
        self._json_encoders = {}
    @property
    def AnyNumericalType(self):
        return any_type(tuple(T if isinstance(T, type) else T()
                        for T in self._AllNumericalTypes))
    @property
    def AnyScalarType(self):
        return any_type(tuple(T if isinstance(T, type) else T()
                        for T in self._AllScalarTypes))
    @property
    def AnyUnitType(self):
        return any_type(tuple(T if isinstance(T, type) else T()
                        for T in self._AllUnitTypes))
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
    def add_unit_type(self, type):
        self._AllUnitTypes.add(type)

    ####################
    # Less unsafe importing of arbitrary modules

    # Without `safe_modules`, a malicious actor could modify a data file
    # containing a serialized model, pointing 'module' to any file in
    # PYTHONPATH, and it would be executed during deserialization.
    # With the `safe_modules` variable, that file would have to be within one
    # of the standard packages (or the user's own package), which are
    # ostensibly already being trusted and executed.
    safe_packages = {
        'mackelab_toolbox'
    }
        # A user can add a module or package to this whitelist with
        #   mtbtyping.safe_packages.add('module_name')
        # or
        #   mtbtyping.safe_packages.update(['module1_name', 'module2_name'])
        # Only the top level package name needs to be added.

    def import_module(self, module):
        import importlib
        if not any(module.startswith(sm) for sm in self.safe_packages):
            raise ValueError(f"Module {module} is not a submodule of one of the "
                             "listed safe packages.\n"
                             f"Safe packages: {self.safe_packages}")
        return importlib.import_module(module)

    ####################
    # Managing JSON encoders

    def add_json_encoder(self, type, encoder, priority=0):
        self._json_encoders[type] = JsonEncoder(encoder, priority)

    @property
    def json_encoders(self):
        return {T:je.encoder
                for T, je in sorted(self._json_encoders.items(),
                                    key = lambda item: -item[1].priority)}
        # Use -je.priority: higher number is higher priority

    @staticmethod
    def json_like(value: typing.Any, type_str: Union[str,List[str]],
                  case_sensitive: bool=False):
        """
        Convenience fonction for checking whether a serialized value might be a
        custom serialized JSON object. All our custom serialized formats start with
        a unique type string.
        :param:value: The value for which we want to determine if it is a
            JSON-serialized object.
        :param:type_str: The type string of the type we are attempting to
            deserialize into.
        :param:case_sensitive: Whether the comparison to `type_str` should be
            case-sensitive.
        """
        if isinstance(type_str, str):
            type_str = [type_str]
        casefold = (lambda v: v) if case_sensitive else str.casefold
        return any(
            (not isinstance(value, str) and isinstance(value, Sequence) and value
             and isinstance(value[0], str) and casefold(value[0]) == casefold(type_str_))
            for type_str_ in type_str)

    ####################
    # Deserialization utilities

    def recursive_validate(self, value: Collection, validate: Callable, json_name: str):
        """
        Recurse through a list or dictionary, checking for any entry `e` for
        which `json_like(e, json_name)` is True.
        On when a match is found, apply `validate`.

        Returns a copy, of the same type as `value`.

        Intended for cases where we need to deserialize objects within a
        `validate` method (by writing our own validation, we take responsibility
        of trigger further necessary recursions)
        """
        itertypes = (list,tuple,set,frozenset)
        json_like = self.json_like
        if json_like(value, json_name):
            return validate(value)
        elif isinstance(value, Mapping):
            assert isinstance(value, dict)
            return {k: validate(v) if json_like(v, json_name)
                       else self.recursive_validate(v, validate, json_name) if isinstance(v, Collection)
                       else v
                    for k,v in value.items()}
        elif isinstance(value, itertypes):
            return type(value)(validate(v) if json_like(v, json_name)
                               else self.recursive_validate(v, validate, json_name) if isinstance(v, Collection)
                               else v
                               for v in value)
        else:
            return value

    ####################
    # Type normalization

    def convert_nptype(self, annotation_type):
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
        if isinstance(T, type) and issubclass(T, np.generic):
            # It's important not to call `np.dtype(T)` if T is already a NumPy
            # type, since it prevents generics (e.g. `np.dtype(np.number) is np.float64`)
            # It also triggers a deprecation warning for this reason.
            # T = T.dtype
            pass
        else:
            T = np.dtype(T).type
        return T

    ####################
    # Conditionally loaded modules

    @staticmethod
    def load_pint():
        import pint
        typing.add_json_encoder(pint.quantity.Quantity, PintValue.json_encoder)
        typing.add_json_encoder(pint.unit.Unit, PintUnit.json_encoder)
        typing.add_unit_type(PintUnit)

    @staticmethod
    def load_quantities():
        import quantities
        # Must have higher priority than numpy types
        typing.add_json_encoder(quantities.quantity.Quantity,
                                QuantitiesValue.json_encoder,
                                priority=5)
        typing.add_json_encoder(quantities.dimensionality.Dimensionality,
                                QuantitiesUnit.json_encoder,
                                priority=5)
        typing.add_unit_type(QuantitiesUnit)

typing = TypeContainer()

####################
# Custom Types for annotations / pydantic

####
# Complex values

typing.add_json_encoder(complex, lambda z: str(z))

####
# Range type

class Range:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v):
        if (isinstance(v, Sequence) and isinstance(v[0], str)
            and v[0].lower() == 'range'): # JSON decoder
            v = range(*v[1])
        if not isinstance(v, range):
            raise TypeError(f"{v} is not of type `range`.")
        return v
    @classmethod
    def __modify_schema__(cls, field_schema):
        """We need to tell pydantic how to export this type to the schema,
        since it doesn't know what to map the type to.
        """
        # See https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types
        # for the API, and https://pydantic-docs.helpmanual.io/usage/schema/
        # for the expected fields
        # TODO: Use 'items' ?
        field_schema.update(
            type="array",
            description="('range', START, STOP, [STEP])"
            )
    @staticmethod
    def json_encoder(v):
        if v.step is None:
            args = (v.start, v.stop)
        else:
            args = (v.start, v.stop, v.step)
        return ("range", args)

typing.Range = Range
typing.add_json_encoder(range, Range.json_encoder)
# Range.register(range)

####
# Slice type

class Slice:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v):
        if isinstance(v, Sequence) and v[0].lower() == 'slice': # JSON decoder
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
        # TODO: Use 'items' ?
        field_schema.update(
            type="array",
            description="('slice', START, STOP, [STEP])"
            )
    @staticmethod
    def json_encoder(v):
        """Attach this method to a pydantic BaseModel as follows:
        >>> class MyModel(BaseModel):
        >>>     class Config:
        >>>         json_encoders = mtb.typing.json_encoders
        """
        if v.step is None:
            args = (v.start, v.stop)
        else:
            args = (v.start, v.stop, v.step)
        return ("slice", args)

typing.Slice = Slice
typing.add_json_encoder(slice, Slice.json_encoder)
# Slice.register(slice)

################
# Port / overrides of typing types for pydantic

#  Pydantic recognizes typing.Sequence, but treats it as a shorthand for
#  Union[List, Tuple] (but with equal precedence). This means that other
#  sequence types like `range` are not recognized.
typing.Sequence = Union[Range, Sequence]

####
# Saving types themselves

class Type(BuiltinType):
    """
    A specialization of Type which supports serialization & deserialization.
    This is accomplished by storing the module and class name, and then
    reimporting the module during deserialization.
    Only modules within packages included in the `typing.safe_packages`
    whitelist are allowed to be reimported.

    .. warning:: If a type is defined in main module, it will be serialized
       with module "__main__". This makes it impossible to deserialize.

    """
    class Data(BaseModel):
        module: str
        name  : str
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v):
        if isinstance(v, type):
            return v
        else:
            data = cls.Data(**v)
            if data.module == "__main__":
                raise ValueError(
                    f"Class '{data.name}' was serialized with module '__main__'. "
                    "To allow classes to be deserialized, ensure they are defined "
                    "in a separate module and imported into the main exec script.")
            module = typing.import_module(data.module)
            C = getattr(module, data.name)
            return C
    @classmethod
    def json_encoder(cls, v):
        return cls.Data(module=v.__module__, name=v.__qualname__)
typing.Type = Type
typing.add_json_encoder(type, Type.json_encoder)

####
# Light structs

class IndexableNamespace(SimpleNamespace):
    """
    A structure allowing elements to be accessed either by attribute or index.
    Inherits from `SimpleNamespace` and adds dict-like access.

    Intended as a simple unstructured container types, like `dict` and
    `SimpleNamespace`, that simply associate names to variables.

    Motivation: Attribute access is often cleaner and more compact, but
    awkward when the attribute name is a variable. For simple structures, there
    seems no reason not to allow both.

    .. Note:: In contrast to a dictionary, the iterator returns both keys and
       values. This allows an `IndexableNamespace` to be used in place of a
       pydantic `~pydantic.BaseModel`.

    Example
    -------
    >>> ns = IndexableNamespace(name='John', phone='555', age=20)
    >>> print(ns.name)
    >>> attr = 'phone'
    >>> print(ns[attr])
    >>> for attr, value in ns:
    >>>   print(attr, value)

    **Usage with Pydantic**

    To use within a Pydantic model, add `IndexableNamespace.json_encoder` to
    the model's encoders:

    >>> class Foo(Pydantic.BaseModel):
    >>>   ns: IndexableNamespace
    >>>   ...
    >>>   class Config:
    >>>     json_encoders: {'ns': IndexableNamespace.json_encoder}
    """

    # Dict-like interface
    def __getitem__(self, key):
        return self.__dict__[key]
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    def __contains__(self, key):
        return key in self.__dict__

    # Required to behave like a mapping, otherwise Pydantic gets confused
    def __iter__(self):
        return iter(self.__dict__.items())
    def keys(self):
        return self.__dict__.keys()
    def values(self):
        return self.__dict__.values()
    def items(self):
        return self.__dict__.items()
        
    # Convert to plain dict
    def as_dict(self):
        return {k: v.as_dict() if isinstance(v, IndexableNamespace) else v
                for k, v in self}

    # Encoder/decoder required for use within a Pydantic model
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value):
        # `value` can be any mapping
        return cls(**value)
    @classmethod
    def json_encoder(cls, value, **kwargs):
        if not isinstance(value, IndexableNamespace):
            logger.error("`IndexableNamespace.json_encoder` expects an "
                         f"IndexableNamespace as `value`; received {value} "
                         f"(type: {type(value)}). Continuing, but behaviour "
                         "is undefined.")
        return value.__dict__
    def json(self):
        return self.json_encoder(self)

typing.IndexableNamespace = IndexableNamespace
typing.add_json_encoder(IndexableNamespace, IndexableNamespace.json_encoder)

################
# Recognizing unit types
#
# class SimplePydanticType:
#     """
#     Base class reducing boilerplate when defining Pydantic-compatible types,
#     with better symmetry of the json encoder+decoder.
#
#     Subclass should define three methods:
#
#     >>> @staticmethod(v):
#     >>>   return isinstance(v, ...)
#     >>> @staticmethod
#     >>> def json_encoder(v):
#     >>>   return ...
#     >>> @staticmethod
#     >>> def json_decoder(v):
#     >>>   return ...
#
#     They must all be decorated by either @staticmethod or @classmethod.
#     """
#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate
#     @classmethod
#     def validate(cls, v):
#         if cls.is_target_type(v):
#             return v
#         else:
#             return cls.json_decoder(v)

class PintValue:
    """
    Before parsing a serialized PintValue, one needs to set the Pint unit
    registry. There should only be one unit registry within a project, so this
    is set as a *class* variable of `PintValue`.
    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> from mackelab_toolbox.typing import PintUnit
    >>> PintUnit.ureg = ureg
    """
    @classmethod
    def __get_validators__(cls):
        # partial doesn't work because Pydantic expects a particular signature
        yield from (lambda v: cls.json_decoder(v, noerror=True),
                    cls.validate_value)

    @staticmethod
    def validate_value(v):
        if isinstance(v, PintUnit.pint().Quantity):
            return v
        raise TypeError

    @staticmethod
    def json_encoder(v):
        return ("PintValue", v.to_tuple())
    @classmethod
    def json_decoder(cls, v, noerror=False):
        pint = PintUnit.pint()
        if PintUnit.ureg is None:
            raise RuntimeError(
                "The Pint unit registry is not set. Do this by assigning to "
                "`PintUnit.ureg` before attempting to parse a Pint value.")
        if isinstance(v, (tuple,list)) and len(v) > 0 and v[0] == "PintValue":
            return PintUnit.ureg.Quantity.from_tuple(v[1])
        elif noerror:
            # Let another validator try to parse the value
            return v
        else:
            raise ValueError("Input is incompatible with PintValue.json_decoder. "
                   f"Input value: {v} (type: {type(v)})")

class PintUnit:
    # TODO: Type check assignment to ureg
    # TODO?: ureg, pint as (meta)class properties ?
    ureg: "pint.registry.UnitRegistry" = None
    @staticmethod
    def pint():
        pint = sys.modules.get('pint', None)
        if pint is None:
            raise ValueError("'pint' module is not loaded.")
        return pint

    @classmethod
    def __get_validators__(cls):
        yield from (lambda v: cls.json_decoder(v, noerror=True),
                    lambda v: cls.validate_value(v, noerror=True),
                    cls.validate_unit)

    @staticmethod
    def validate_unit(v):
        if isinstance(v, PintUnit.pint().Unit):
            return v
        else:
            raise TypeError
    @staticmethod
    def validate_value(v, noerror=False):
        if isinstance(v, PintUnit.pint().Quantity):
            if v.magnitude != 1:
                raise ValueError("Quantities can only be converted to units "
                                 "if they have unit magnitude.")
            return v.units
        elif noerror:
            # Let another validator try to parse the value
            return v
        else:
            raise TypeError

    @staticmethod
    def json_encoder(v):
        return ("PintUnit", (str(v),))
    @classmethod
    def json_decoder(cls, v, noerror=False):
        pint = PintUnit.pint()
        if PintUnit.ureg is None:
            raise RuntimeError(
                "The Pint unit registry is not set. Do this by assigning to "
                "`PintUnit.ureg` before attempting to parse a Pint value.")
        if isinstance(v, (tuple, list)) and len(v) > 0 and v[0] == "PintUnit":
            return PintUnit.ureg.Unit(v[1][0])
        elif noerror:
            # Let another validator try to parse the value
            return v
        else:
            raise ValueError("Input is incompatible with PintUnit.json_decoder. "
                   f"Input value: {v} (type: {type(v)})")

class QuantitiesValue():
    @classmethod
    def __get_validators__(cls):
        yield from (cls.json_noerror,  # For whatever reason, lambda doesn't work
                    cls.validate_value)

    @classmethod
    def json_noerror(cls, v):
        return cls.json_decoder(v, noerror=True)

    @classmethod
    def validate_value(cls, v):
        pq = QuantitiesUnit.pq()
        if isinstance(v, pq.Quantity):
            return v
        raise TypeError

    @staticmethod
    def json_encoder(v):
        return ("QuantitiesValue", (v.magnitude, str(v.dimensionality)))
    @staticmethod
    def json_decoder(v, noerror=False):
        pq = sys.modules.get('quantities', None)
        if pq is None:
            raise ValueError("'Quantities' module is not loaded.")
        elif (isinstance(v, (tuple,list))
              and len(v) > 0 and v[0] == "QuantitiesValue"):
            return pq.Quantity(v[1][0], units=v[1][1])
        elif noerror:
            # Let another validator try to parse the value
            return v
        else:
            raise ValueError("Input is incompatible with QuantitiesValue.json_decoder. "
                   f"Input value: {v} (type: {type(v)})")

class QuantitiesUnit(QuantitiesValue):
    """
    Exactly the same as QuantitiesValue, except that we enforce the magnitude
    to be 1. In contrast to Pint, Quantities doesn't seem to have a unit type.
    """
    @staticmethod
    def pq():
        pq = sys.modules.get('quantities', None)
        if pq is None:
            raise ValueError("'Quantities' module is not loaded.")
        return pq

    @classmethod
    def validate_value(cls, v, field=None):
        if field is None:
            field = SimpleNamespace(name="")
        pq = QuantitiesUnit.pq()
        if isinstance(v, pq.Quantity):
            if v.magnitude != 1:
                raise ValueError(f"Field {field.name}: Units must have "
                                 "magnitude of one.")
            return v
        raise TypeError

class QuantitiesDimension():

    @classmethod
    def __get_validators__(cls):
        yield from (lambda v: cls.json_decoder(v, noerror=True),
                    lambda v: cls.validate_value(v, noerror=True),
                    cls.validate_dim)

    @staticmethod
    def validate_dim(v):
        pq = QuantitiesUnit.pq()
        if isinstance(v, pq.dimensionality.Dimensionality):
            return v
        raise TypeError
    @staticmethod
    def validate_value(v, noerror=False):
        pq = QuantitiesUnit.pq()
        if isinstance(v, pq.Quantity):
            return v.dimensionality
        elif noerror:
            return v
        else:
            raise TypeError

    @staticmethod
    def json_encoder(v):
        return ("QuantitiesDimension", (str(v),))
    @staticmethod
    def json_decoder(v, noerror=False):
        pq = sys.modules.get('quantities', None)
        if pq is None:
            raise ValueError("'Quantities' module is not loaded.")
        if isinstance(v, (tuple,list)) and len(v) > 0 and v[0] == "QuantitiesDimension":
            return pq.quantity.validate_dimensionality(v[1][0])
        elif noerror:
            # Let another validator try to parse
            return v
        else:
            raise ValueError("Input is incompatible with QuantitiesDimension.json_decoder. "
                   f"Input value: {v} (type: {type(v)})")

typing.PintValue = PintValue
typing.PintUnit  = PintUnit
typing.QuantitiesValue = QuantitiesValue
typing.QuantitiesUnit = QuantitiesUnit
typing.QuantitiesDimension = QuantitiesDimension
# JSON encoders added by typing.load_pint/load_quantities

################
# Types based on numbers module

class Number(numbers.Number):
    """
    This type does not support coercion, only validation.
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value, field=None):
        if field is None:
            field = SimpleNamespace(name="")
        if not isinstance(value, numbers.Number):
            raise TypeError(f"Field {field.name} expects a number. "
                            f"It received {value} [type: {type(value)}].")
        return value
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="number")
typing.Number = Number
typing.add_numerical_type(Number)
typing.add_scalar_type(Number)

class Integral(numbers.Integral):
    """
    This type does not support coercion, only validation.
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value, field=None):
        if field is None:
            field = SimpleNamespace(name="")
        if not isinstance(value, numbers.Integral):
            raise TypeError(f"Field {field.name} expects an integer. "
                            f"It received {value} [type: {type(value)}].")
        return value
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="integer")
typing.Integral = Integral
# Don't add to numerical/scalar_type: covered by Number

class Real(numbers.Real):
    """
    This type generally does not support coercion, only validation.
    Exception: integer values are cast with `float`.
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value, field=None):
        if field is None:
            field = SimpleNamespace(name="")
        if not isinstance(value, numbers.Real):
            raise TypeError(f"Field {field.name} expects a real number. "
                            f"It received {value} [type: {type(value)}].")
        elif isinstance(value, numbers.Integral):
            # Convert ints to floating point
            return float(value)
        return value
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="number")
typing.Real = Real
# Don't add to numerical/scalar_type: covered by Number

################
# DType

class DType:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value):
        return np.dtype(value)
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="str")
    @classmethod
    def json_encoder(cls, value):
        return str(value)
typing.DType = DType
typing.add_json_encoder(np.dtype, DType.json_encoder)

################
# NPValue type

def infer_numpy_type_to_cast(nptype, value):
    """
    This function tries to determine which concrete numpy type to use to cast
    `value`, given the desired type `nptype`.
    The challenge is that `nptype` may be an abstract type, so we
    have to work through the possible hierarchy until we find the
    most appropriate concrete type. We only do the most common cases;
    we can extend later to the complete tree.
    """
    # If `nptype` is an NPValue subclass, get the true numpy type
    nptype = getattr(nptype, 'nptype', nptype)
    assert issubclass(nptype, np.generic), f"{nptype} is not a subclass of `np.generic`"
    if nptype is np.generic:
        raise NotImplementedError("Unclear how we should cast np.generic.")
    if issubclass(nptype, np.flexible):  # void, str, unicode
        # Assume concrete type
        return nptype
    elif not issubclass(nptype, np.number):  # bool, object
        assert (nptype is np.bool_ or nptype is np.object_), f"Expected NumPy type 'bool' or 'object'; received '{nptype}'"
        return nptype
    else:  # Number
        if issubclass(nptype, np.integer):
            if nptype is np.integer:
                return np.int_
            else:
                # Assume concrete type
                return nptype
        elif issubclass(nptype, np.inexact):
            if issubclass(nptype, np.complexfloating):
                if nptype is np.complexfloating:
                    return np.complex_
                else:
                    # Assume concrete type
                    return nptype
            elif issubclass(nptype, np.floating):
                if nptype is np.floating:
                    return np.float_
                else:
                    # Assume concrete type
                    return nptype
            else:
                assert nptype is np.inexact, f"Expected NumPy type 'inexact'; received '{nptype}'"
                # We try to guess which type to use between float and complex.
                # We make a rudimentary check for complex, and fall back to float.
                if isinstance(value, complex):
                    return np.complex_
                if isinstance(value, str) and ('j' in value or 'i' in value):
                    return np.complex_
                else:
                    return np.float_
        else:
            assert nptype is np.number, f"Expected NumPy type 'number'; received '{nptype}'"
            # We try to guess which type to use between int, float and complex.
            # We make a rudimentary check for int, complex, and fall back to float.
            if isinstance(value, int):
                return np.int_
            elif isinstance(value, complex):
                return np.complex_
            elif isinstance(value, str):
                if 'j' in value or 'i' in value:
                    return np.complex_
                elif '.' in value:
                    return np.float_
                else:
                    return np.int_
            else:
                return np.float_

class _NPValueType(np.generic):
    nptype = None

    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value, field=None):
        # NOTE:
        # We make an exception to always allow casting from a string.
        # This allows for complex values, which may not be converted to numbers
        # by the JSON deserializer and still be represented as strings
        if field is None:
            field = SimpleNamespace(name="")
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
            and np.issubdtype(type(value), cls.nptype)):
            return value
        elif (np.can_cast(value, cls.nptype)
              or np.issubdtype(getattr(value, 'dtype', np.dtype('O')), np.dtype(str))):
            # Exception for strings, as stated above
            nptype = infer_numpy_type_to_cast(cls.nptype, value)
            return nptype(value)
        else:
            raise TypeError(f"Cannot safely cast '{field.name}' type  "
                            f"({type(value)}) to type {cls.nptype}.")
    @classmethod
    def __modify_schema__(cls, field_schema):
        if np.issubdtype(cls.nptype, np.integer):
            field_schema.update(type="integer")
        else:
            field_schema.update(type="number")

    @classmethod
    def json_encoder(cls, v):
        """See typing.json_encoders."""
        return v.item()  #  Convert Numpy to native Python type

class _NPValueMeta(type):
    def __getitem__(self, nptype):
        nptype=typing.convert_nptype(nptype)
        nptype_str = nptype.__name__
        return type(f'NPValue[{nptype_str}]', (_NPValueType,),
                    {'nptype': nptype})

class NPValue(np.generic, metaclass=_NPValueMeta):
    """
    Use this to use a NumPy dtype for type annotation; `pydantic` will
    recognize the type and execute appropriate validation/parsing.

    This may become obsolete, or need to be updated, when NumPy officially
    supports type hints (see https://github.com/numpy/numpy-stubs).

    - `NPValue[T]` specifies an object to be casted with dtype `T`. Any
       expression for which `np.dtype(T)` is valid is accepted.

    .. Note:: Difference with `DType`. The annotation `NPValue[np.int8]`
    matches any value of the same type as would be returned by `np.int8`.
    `DType` describes an instance of `dtype` and would match `np.dtype('int8')`,
    but also `np.dtype(float)`, etc.

    Example
    -------
    >>> from pydantic.dataclasses import dataclass
    >>> from mackelab_toolbox.typing import NPValue
    >>>
    >>> @dataclass
    >>> class Model:
    >>>     x: NPValue[np.float64]
    >>>     y: NPValue[np.int8]

    """
    pass

typing.NPValue = NPValue
typing.add_numerical_type(NPValue[np.number])
typing.add_scalar_type(NPValue[np.number])
typing.add_json_encoder(np.generic, _NPValueType.json_encoder)

####
# Array type

# TODO: Configurable threshold for compression
# TODO: Allow different serialization methods:
#       - str(A.tolist())
#       - base64(blosc) (with configurable keywords for blosc)
#       - base64(zlib)
#       - external file

class _ArrayType(np.ndarray):
    nptype = None   # This must be a type (np.int32, not np.dtype('int32'))
    _ndim = None

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value, field=None):
        # NOTE:
        # We make an exception to always allow casting from a string.
        # This allows for cases where an array is serialized as a list
        # of strings, rather than a string of a list.
        if field is None:
            field = SimpleNamespace(name="")
        if isinstance(value, typing.NotCastableToArray):
            raise TypeError(f"Values of type {type(value)} cannot be casted "
                             "to a numpy array.")

        if typing.json_like(value, "Array"):
            if 'compression' in value[1]:
                encoding = value[1]['encoding']
                compression = value[1]['compression']
                assert encoding == 'b85', f"Array reports encoding '{encoding}'; should be 'b85'"
                assert compression in ['blosc', 'none'], f"Array records compression '{compression}'; should be 'blosc' or 'none'"
                encoded_array = value[1]['data']
                v_bytes = base64.b85decode(encoded_array)
                if compression == 'blosc':
                    v_bytes = blosc.decompress(v_bytes)
                with io.BytesIO(v_bytes) as f:
                    decoded_array = np.load(f)
                return decoded_array
            else:
                return np.array(value[1]['data'], dtype=value[1]['dtype'])

        elif isinstance(value, np.ndarray):
            # Don't create a new array unless necessary
            if cls._ndim  is not None and value.ndim != cls._ndim:
                raise TypeError(f"{field.name} expects a variable with "
                                f"{cls._ndim} dimensions.")
            # Issubdtype allows specifying abstract dtypes like 'number', 'floating'
            if cls.nptype is None or np.issubdtype(value.dtype, cls.nptype):
                result = value
            elif (np.can_cast(value, cls.nptype)
                  or np.issubdtype(value.dtype, np.dtype(str))):
                # We make a exception to always allow casting strings
                nptype = infer_numpy_type_to_cast(cls.nptype, value)
                result = value.astype(nptype)
            else:
                raise TypeError(f"Cannot safely cast '{field.name}' (type  "
                                f"{value.dtype}) to type {cls.nptype}.")
        else:
            result = np.array(value)
            # HACK: Since np.array(…) will accept almost anything, we use
            #       heuristics to try to detect when array construction has
            #       probably failed
            if (isinstance(result, Sequence)
                and any(isinstance(x, Iterable)
                        and not isinstance(x, (np.ndarray, str, bytes))
                        for x in result)):
               # Nested iterables should be unwrapped into an n-d array
               # When this fails (if types or depths are inconsistent), then
               # only the outer level is unwrapped.
               raise TypeError(f"Unable to cast {value} to an array.")
            # Check that array matches expected shape and dtype
            if cls._ndim is not None and result.ndim != cls._ndim:
                raise TypeError(
                    f"The dimensionality of the data (dim: {result.ndim}, "
                    f"shape: {result.shape}) does not correspond to the "
                    f"expected of dimensions ({cls._ndim} for '{field.name}').")
            # Issubdtype allows specifying abstract dtypes like 'number', 'floating'
            if cls.nptype is None or np.issubdtype(result.dtype, cls.nptype):
                pass
            elif (np.can_cast(result, cls.nptype)
                  or np.issubdtype(result.dtype, np.dtype(str))):
                # We make a exception to always allow casting strings
                nptype = infer_numpy_type_to_cast(cls.nptype, result)
                result = result.astype(nptype)
            else:
                raise TypeError(f"Cannot safely cast '{field.name}' (type  "
                                f"{result.dtype}) to an array of type {cls.nptype}.")
        return result

    @classmethod
    def __modify_schema__(cls, field_schema):
        # FIXME: Figure out how to use schema of subfield
        field_schema.update(type ='array',
                            items={'type': 'number'})

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
        if isinstance(T, np.dtype):
            T = T.type
        elif isinstance(T, str):
            T = np.dtype(T).type
        if (not isinstance(T, type) or len(extraargs) > 0
            or not isinstance(ndim, (int, type(None)))):
            # if isinstance(args, (tuple, list)):
            #     argstr = ', '.join((str(a) for a in args))
            # else:
            #     argstr = str(args)
            raise TypeError(
                "`Array` must be specified as either `Array[T]`"
                "or `Array[T, n], where `T` is a type and `n` is an int. "
                f"(received: {args}]).")
        nptype=typing.convert_nptype(T)
        # specifier = str(nptype)
        specifier = nptype.__name__
        if ndim is not None:
            specifier += f",{ndim}"
        return type(f'Array[{specifier}]', (_ArrayType,),
                    {'nptype': nptype, '_ndim': ndim})

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
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
        # raise NotImplementedError("You must specify a data type with Array, "
        #                           "e.g. Array[np.float64].")
    @classmethod
    def validate(cls, v, field=None):
        return _ArrayType.validate(v, field=None)

    @staticmethod
    def json_encoder(v, compression='blosc', encoding='b85'):
        """See typing.json_encoders."""
        threshold = 100  # ~ break-even point for 64-bit floats, blosc, base85
        if v.size <= threshold:
            # For short array, just save it as a string
            # Since the string loses type information, save the dtype as well.
            return ('Array', {'data': v.tolist(),
                              'dtype': v.dtype})
        else:
            # Save longer arrays in base85 encoding, with short summary
            if encoding != 'b85':
                raise NotImplementedError("The only supported encoding "
                                          "currently is 'b85'.")
            if compression not in ['none', 'blosc', None]:
                raise NotImplementedError("The only supported compressions "
                                          "currently are 'blosc' and 'none'.")
            elif compression is None:
                compression = 'none'
            with io.BytesIO() as f:  # Use file object to keep bytes in memory
                np.save(f, v)        # Convert array to plateform-independent bytes  (`tobytes` not meant for storage)
                v_bytes = f.getvalue()
            # Compress and encode the bytes to a compact string representation
            if compression == 'blosc':
                v_bytes = blosc.compress(v_bytes)
            v_b85 = base64.b85encode(v_bytes)
            # Set print threshold to ensure str returns a summary
            with np.printoptions(threshold=threshold):
                v_sum = str(v)
            return ('Array', {'encoding': f'{encoding}',
                              'compression': f'{compression}',
                              'data': v_b85,
                              'summary': v_sum})

_ArrayType.json_encoder = Array.json_encoder

typing.Array = Array
# >>>> FIXME: The lines below don't work because generic type np.number -> np.float64
# >>>>        Especially bad because it prevents specifying 0-dim for scalar
typing.add_numerical_type(Array[np.number])
typing.add_scalar_type(Array[np.number, 0])
typing.add_json_encoder(np.ndarray, Array.json_encoder)

#####
# Numpy random generators

class RNGenerator(np.random.Generator):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value, field=None):
        if field is None:
            field = SimpleNamespace(name='')
        if isinstance(value, np.random.Generator):
            return value
        elif typing.json_like(value, "RNGenerator"):
        #elif (isinstance(value, dict)
        #      and 'bit_generator' in value and 'state' in value):
            # Looks like a json-serialized state dictionary
            state = value[1]
            BG = getattr(np.random, state['bit_generator'])()
            BG.state = state
            return np.random.Generator(BG)
        else:
            raise TypeError(f"Field {field.name} expects an instance of "
                            f"np.random.Generator.\nProvided value: {value} "
                            f"(type: {type(value)}).")
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type='array',
                            items=[{'type': 'string',
                                    'type': 'object'}])
    @classmethod
    def json_encoder(cls, v):
        # Generators are containers around a BitGenerator; it's the state of
        # BitGenerator that we need to save. Default BitGenerator is 'PCG64'.
        # State is a dict with two required fields: 'bit_generator' and 'state',
        # plus 1-3 extra fields depending on the generator.
        # 'state' field is itself a dictionary, which may contain arrays of
        # type uint32 or uint64
        return ("RNGenerator", v.bit_generator.state)
            # Pydantic will recursively encode state entries, and use Array's
            # json_encoder when needed

class RandomState(np.random.RandomState):
    """Pydantic typing support for the legacy RandomState object."""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value, field=None):
        if field is None:
            field = SimpleNamespace(name='')
        if isinstance(value, np.random.RandomState):
            return value
        elif typing.json_like(value, "RandomState"):
            # Looks like a json-serialized state tuple
            rs = np.random.RandomState()
            rng_state = value[1]
            # Deserialize arrays that were serialized in the Array format
            for i, v in enumerate(rng_state):
                if typing.json_like(v, 'Array'):
                    rng_state[i] = _ArrayType.validate(v)
            # Set the RNG to the state saved in the file
            rs.set_state(rng_state)
            return rs
        else:
            field_name = getattr(field, 'name', "")
            raise TypeError(f"Field {field_name} expects an instance of "
                            f"np.random.RandomState.\nProvided value: {value}")
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
        type='array',
        items=[{'type': 'string'},
               {'type': 'array',
                'items': [{'type': 'array', 'items': {'type': 'integer'}},
                          {'type': 'integer'},
                          {'type': 'integer'},
                          {'type': 'number'}]}
              ])
    @classmethod
    def json_encoder(cls, v):
        return ("RandomState", v.get_state())

typing.RNGenerator = RNGenerator
typing.RandomState = RandomState
typing.add_json_encoder(np.random.Generator, RNGenerator.json_encoder)
typing.add_json_encoder(np.random.RandomState, RandomState.json_encoder)

# ####
# SciPy statistical distributions

import scipy as sp
import scipy.stats

# FIXME: Is there a public name for frozen data types ?
RVFrozen = sp.stats._distn_infrastructure.rv_frozen
MvRVFrozen = sp.stats._multivariate.multi_rv_frozen
MvNormalFrozen = sp.stats._multivariate.multivariate_normal_frozen
# List of modules searched for distribution names; precedence is given to
# modules earlier in the list
stat_modules = [sp.stats]

# Making an ABC allows to declare subclasses with @Distribution.register
class Distribution(abc.ABC):
    """
    Pydantic-aware type for SciPy _frozen_ distributions. A frozen distribution
    is one for which the parameters (like `loc` and `scale`) are fixed.
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_dispatch

    @classmethod
    def validate_dispatch(cls, v):
        """
        Find the correct distribution class by inspecting the modules in
        `stat_modules`, and call its `validate` method on `v`.
        If the distribution class does not define `validate`, use
        `Distribution.default_validate`.

        .. Note:: Since  `default_validate` works with all standard scipy.stats
           distributions, it is rarely needed to define custom `validate`
           methods.
        """
        if isinstance(v, (RVFrozen, MvRVFrozen)):
            # Already a frozen dist; nothing to do
            return v
        elif isinstance(v, (sp.stats._distn_infrastructure.rv_generic,
                            sp.stats._multivariate.multi_rv_generic)):
            raise TypeError("`Distribution` expects a frozen random variable; "
                            f"received '{v}' with unspecified parameters.")
        elif typing.json_like(v, "Distribution"):
            dist = None
            for module in stat_modules:
                dist = getattr(module, v[1], None)
                if dist:
                    break
            if dist is None:
                raise ValueError(f"Unable to deserialize distribution '{v[1]}': "
                                 "unrecognized distribution name.")
            if hasattr(dist, 'validate'):
                return dist.validate(v)
            else:
                return cls.default_validate(v)
        else:
            v_s = str(v)
            if len(v_s) > 50:
                v_s = v_s[:49] + '…'
            raise TypeError(f"Value `{v_s}` is neither a distribution object "
                            "nor a recognized serialization of a distribution object.")

    @classmethod
    def default_validate(cls, v):
        dist = None
        for module in stat_modules:
            dist = getattr(module, v[1], None)
            if dist:
                break
        assert dist is not None
        if len(v) > 5 or len(v) < 4:
            logger.warning("Malformed serialization: Serialized Distribution "
                           f"expects 4 or 5 fields; received {len(v)}.")
        if len(v) == 5:
            rng_state = v[4]
            if typing.json_like(rng_state, "RandomState"):
                rng_state = RandomState.validate(rng_state)
            elif typing.json_like(rng_state, "RNGenerator"):
                rng_state = RNGenerator.validate(rng_state)
        else:
            rng_state = None
        kwds = v[3]
        if not isinstance(kwds, dict):
            raise TypeError("The fourth field of a serialized Distribution "
                            f"should be a dictionary; received {type(kwds)}.")
        kwds = {k: Array.validate(v) if typing.json_like(v, "Array") else v
                for k, v in kwds.items()}
        args = v[2]
        if not isinstance(args, Sequence):
            raise TypeError("The third field of a serialized Distribution "
                            f"should be a tuple or list; received {type(kwds)}.")
        args = [Array.validate(v) if typing.json_like(v, "Array") else v
                for v in args]
        # Special case for mixture distribution: all recursive deserialization
        for key, val in kwds.items():
            if key == 'dists':
                if isinstance(val, Iterable):
                    kwds[key] = [Distribution.validate_dispatch(dist)
                                 if typing.json_like(dist, "Distribution")
                                 else dist
                                 for dist in val]
                else:
                    # There should not be a code path which leads here,
                    # but if one _did_ exist, this seems the most reasonable
                    kwds[key] = (Distribution.validate_dispatch(val)
                                 if typing.json_like(dist, "Distribution")
                                 else dist)
        # Special case for distributions as arguments (e.g. transformed)
        # (Could this be combined with the special case for mixtures ?)
        for i, arg in enumerate(args[:]):
            if typing.json_like(arg, "Distribution"):
                args[i] = Distribution.validate_dispatch(arg)
        for key, val in kwds.items():
            if typing.json_like(val, "Distribution"):
                kwds[key] = Distribution.validate_dispatch(val)
        # Finally, reconstruct the serialized distribution
        frozen_dist = dist(*args, **kwds)
        frozen_dist.random_state = rng_state
            # The random state is actually tied to the internal, non-frozen dist object, but we can ignore that for our purposes
        return frozen_dist


    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type='array',
                            items=[{'type': 'string'},
                                   {'type': 'string'},  # dist name
                                   {'type': 'array'},   # dist args
                                   {'type': 'array'},   # dist kwds
                                   {'type': 'array'}    # random state (optional) Accepted: int, old RandomState, new RNGenerator
                                   ])

    @staticmethod
    def json_encoder_RVFrozen(v, include_rng_state=True):
        if v.args:
            logger.warning(
                "For the most consistent and reliable serialization of "
                "distributions, consider specifying them using only keyword "
                f"parameters. Received for distribution {v.dist.name}:\n"
                f"Positional args: {v.args}\nKeyword args: {v.kwds}")
        random_state = v.dist._random_state if include_rng_state else None
        return ("Distribution", v.dist.name, v.args, v.kwds, random_state)
    @staticmethod
    def json_encoder_MvRVFrozen(v, include_rng_state=True):
        # We need to special case multivariate distributions, because they follow
        # a different convention (and don't have a standard 'kwds' attribute)
        dist = v._dist
        if isinstance(v, MvNormalFrozen):
            name = "multivariate_normal"
            args = ()
            kwds = dict(mean=v.mean, cov=v.cov)
        else:
            raise ValueError(
                "The json_encoder for `Distribution` needs to be special "
                "cased for each multivariate distribution, and this has "
                f"not yet been done for '{dist}'.")
        random_state = dist._random_state if include_rng_state else None
        return ("Distribution", name, args, kwds, random_state)

typing.Distribution = Distribution
typing.add_json_encoder(RVFrozen, Distribution.json_encoder_RVFrozen)
typing.add_json_encoder(MvRVFrozen, Distribution.json_encoder_MvRVFrozen)
