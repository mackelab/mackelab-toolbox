import sys
import importlib
import inspect
import operator
import functools

from typing import Callable, Sequence, _Final
from types import new_class

from mackelab_toolbox import serialize
from .typing_module import typing

from numbers import Number
import numpy as np
PlainArg = (Number, str, np.ndarray)

__all__ = ["PlainArg", "PureFunction", "PartialPureFunction", "CompositePureFunction"]

class PureFunctionMeta(type):
    _instantiated_types = {}
    def __getitem__(cls, args):
        """
        Returns a subclass of `PureFunction`. Args may consist of
        - The callable type (in the same format as `~typing.Callable`).
        - Module names. These are used to define namespaces which should be
          imported into the local namespace during deserialization.
        - Both a callable type and module names.
        
        .. Note::
           Types cannot be specified as strings – string arguments are assumed
           to be module names.
        """
        # Parse the arguments
        callableT = {'inT': None, 'outT': None}
        modules = []
        for a in args:
            if isinstance(a, str):
                modules.append(a)
            elif inspect.ismodule(a):
                for nm, m in sys.modules.items():
                    if m is a:
                        modules.append(nm)
                        break
                else:
                    raise AssertionError(f"Module {a} not found in `sys.modules`.")
            elif isinstance(a, list):
                if callableT['inT'] is not None:
                    raise TypeError("Only one input type argument may be specified to"
                                     f"`PureFunction`. Received {callableT['inT']} and {a}.")
                callableT['inT'] = a
            elif isinstance(a, (_Final, type)) or a is None:
                if callableT['outT'] is not None:
                    raise TypeError("Only one output type argument may be specified to"
                                     f"`PureFunction`. Received {callableT} and {a}.")
                if a is None:
                    a = type(None)  # This is what Callable does automatically anyway, and it allows us to check below that either both of inT, outT were passed, or neither
                callableT['outT'] = a
            else:
                raise TypeError("Arguments to the `PureFunction` type can "
                                "consist of zero or one type and zero or more "
                                f"module names. Received {a}, which is of type "
                                f"type {type(a)}.")
        # Treat the callable type, if present
        if (callableT['inT'] is None) != (callableT['outT'] is None):
            raise TypeError("Either both the input and output type of a "
                            "PureFunction must be specified, or neither.")
        if callableT['inT']:
            assert callableT['outT'] is not None
            baseT = Callable[callableT['inT'], callableT['outT']]
            argstr = f"{callableT['inT']}, {callableT['outT']}"
        else:
            baseT = Callable
            argstr = ""
        # Treat the module names, if present
        if modules:
            if argstr:
                argstr += ", "
            argstr += ", ".join(modules)
        # Check if this PureFunction has already been created, and if not, do so
        key = (cls, baseT, tuple(modules))
        if key not in cls._instantiated_types:
            PureFunctionSubtype = new_class(
                f'{cls.__name__}[{argstr}]', (cls,))
            cls._instantiated_types[key] = PureFunctionSubtype
            PureFunctionSubtype.modules = cls.modules + modules
        # Return the PureFunction type
        return cls._instantiated_types[key]

class PureFunction(metaclass=PureFunctionMeta):
    """
    A Pydantic-compatible function type, which supports deserialization.
    A “pure function” is one with no side-effects, and which is entirely
    determined by its inputs.

    Accepts also partial functions, in which case an instance of the subclass
    `PartialPureFunction` is returned.

    .. Warning:: Deserializing functions is necessarily fragile, since there
       is no way of guaranteeing that they are truly pure.
       When using a `PureFunction` type, always take extra care that the inputs
       are sane.

    .. Note:: Functions are deserialized without the scope in which they
       were created.

    .. Hint:: If ``f`` is meant to be a `PureFunction`, but defined as::

       >>> import math
       >>> def f(x):
       >>>   return math.sqrt(x)

       then it has dependency on ``math`` which is outside its scope, and is
       thus impure. It can be made pure by putting the import inside the
       function::

       >>> def f(x):
       >>>   import math
       >>>   return math.sqrt(x)

    .. Note:: Like `Callable`, `PureFunction` allows to specify the type
       within brackets: ``PureFunction[[arg types], return y]``. However the
       returned type doesn't support type-checking.

    .. WIP: One or more modules can be specified to provide definitions for
       deserializing the file, but these modules are not serialized with the
       function.
    """
    modules = []  # Use this to list modules that should be imported into
                  # the global namespace before deserializing the function
    subtypes= {}  # Dictionary of {JSON label: deserializer} pairs.
                  # Use this to define additional PureFunction subtypes
                  # for deserialization.
                  # JSON label is the string stored as first entry in the
                  # serialized tuple, indicating the type.
                  # `deserializer` should be a function (usually the type's
                  # `validate` method) taking the serialized value and
                  # returning the PureFunction instance.
    # Instance variable
    func: Callable

    def __new__(cls, func=None):
        # func=None allowed to not break __reduce__ (due to metaclass)
        # – inside a __reduce__, it's fine because __reduce__ will fill __dict__ after creating the empty object
        if isinstance(func, functools.partial) and not issubclass(cls, PartialPureFunction):
            # Redirect to PartialPureFunction constructor
            # FIXME: What if we get here from CompositePureFunction.__new__
            try:
                Partial = cls.__dict__['Partial']
            except KeyError:
                raise TypeError(f"{func} is a partial functional but '{cls}' "
                                "does not define a partial variant.")
            else:
                return Partial(func)
        # if cls is PureFunction and isinstance(func, functools.partial):
        #     # Redirect to PartialPureFunction constructor
        #     # FIXME: What if we get here from CompositePureFunction.__new__
        #     return PartialPureFunction(func)
        return super().__new__(cls)
    def __init__(self, func):
        if hasattr(self, 'func'):
            # This is our second pass through __init__, probably b/c of __new__redirect
            assert hasattr(self, '__signature__')
            return
        self.func = func
        # Copy attributes like __name__, __module__, ...
        functools.update_wrapper(self, func)
        self.__signature__ = inspect.signature(func)
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    ## Function arithmetic ##
    def __abs__(self):
        return CompositePureFunction(operator.abs, self)
    def __neg__(self):
        return CompositePureFunction(operator.neg, self)
    def __pos__(self):
        return CompositePureFunction(operator.pos, self)
    def __add__(self, other):
        if other == 0:  # Allows using sum([PureFn]) without creating unnecessary Composite functions
            return self
        return CompositePureFunction(operator.add, self, other)
    def __radd__(self, other):
        if other == 0:  # Idem
            return self
        return CompositePureFunction(operator.add, other, self)
    def __sub__(self, other):
        if other == 0:  # Idem
            return self
        return CompositePureFunction(operator.sub, self, other)
    def __rsub__(self, other):
        return CompositePureFunction(operator.sub, other, self)
    def __mul__(self, other):
        return CompositePureFunction(operator.mul, self, other)
    def __rmul__(self, other):
        return CompositePureFunction(operator.mul, other, self)
    def __truediv__(self, other):
        return CompositePureFunction(operator.truediv, self, other)
    def __rtruediv__(self, other):
        return CompositePureFunction(operator.truediv, other, self)
    def __pow__(self, other):
        return CompositePureFunction(operator.pow, self, other)

    ## Serialization / deserialization ##
    # The attribute '__func_src__', if it exists,
    # is required for deserialization. This attribute is added by
    # mtb.serialize when it deserializes a function string.
    # We want it to be attached to the underlying function, to be sure
    # the serializer can find it
    @property
    def __func_src__(self):
        return self.func.__func_src__
    @__func_src__.setter
    def __func_src__(self, value):
        self.func.__func_src__ = value

    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value):
        if isinstance(value, PureFunction):
            pure_func = value
        elif isinstance(value, Callable):
            pure_func = PureFunction(value)
        elif isinstance(value, str):
            modules = [importlib.import_module(m_name) for m_name in cls.modules]
            global_ns = {k:v for m in modules
                             for k,v in m.__dict__.items()}
            # Since decorators are serialized with the function, we should at
            # least make the decorators in this module available.
            local_ns = {'PureFunction': PureFunction,
                        'PartialPureFunction': PartialPureFunction,
                        'CompositePureFunction': CompositePureFunction}
            pure_func = serialize.deserialize_function(
                value, global_ns, local_ns)
            # It is possible for a function to be serialized with a decorator
            # which returns a PureFunction, or even a subclass of PureFunction
            # In such a case, casting as PureFunction may be destructive, and
            # is at best useless
            if not isinstance(pure_func, cls):
                pure_func = cls(pure_func)
        elif (isinstance(value, Sequence) and len(value) > 0):
            label = value[0]
            if label == "PartialPureFunction":
                pure_func = PartialPureFunction._validate_serialized(value)
            elif label == "CompositePureFunction":
                pure_func = CompositePureFunction._validate_serialized(value)
            elif label in cls.subtypes:
                pure_func = cls.subtypes[label](value)
            else:
                cls.raise_validation_error(value)
        return pure_func

    # TODO: Add arg so PureFunction subtype can be specified in error message
    @classmethod
    def raise_validation_error(cls, value):
        raise TypeError("PureFunction can be instantiated from either "
                        "a callable, "
                        "a Sequence `([PureFunction subtype name], func, bound_values)`, "
                        "or a string. "
                        f"Received {value} (type: {type(value)}).")

    @staticmethod
    def json_encoder(v):
        if isinstance(v, PartialPureFunction):
            return PartialPureFunction.json_encoder(v)
        elif isinstance(v, CompositePureFunction):
            return CompositePureFunction.json_encoder(v)
        elif isinstance(v, PureFunction):
            f = v.func
        elif isinstance(v, Callable):
            f = v
        else:
            raise TypeError("`PureFunction.json_encoder` only accepts "
                            f"functions as arguments. Received {type(v)}.")
        return serialize.serialize_function(f)

class PartialPureFunction(PureFunction):
    """
    A `PartialPureFunction` is a function which, once made partial by binding
    the given arguments, is pure (it has no side-effects).
    The original function may be impure.
    """
    def __init__(self, partial_func):
        super().__init__(partial_func)

    @classmethod
    def _validate_serialized(cls, value):
        if not (isinstance(value, Sequence)
                and len(value) > 0 and value[0] == "PartialPureFunction"):
            cls.raise_validation_error(value)
        assert len(value) == 3, f"Serialized PartialPureFunction must have 3 elements. Received {len(tvalue)}"
        assert isinstance(value[1], str), f"Second element of serialized PartialPureFunction must be a string.\nReceived {value[1]} (type: {type(value[1])}"
        assert isinstance(value[2], dict), f"Third element of serialized PartialPureFunction must be a dict.\nReceived {value[2]} (type: {type(value[2])})"
        func_str = value[1]
        bound_values = value[2]
        modules = [importlib.import_module(m_name) for m_name in cls.modules]
        global_ns = {k:v for m in modules
                         for k,v in m.__dict__.items()}
        func = serialize.deserialize_function(func_str, global_ns)
        if isinstance(func, cls):
            raise NotImplementedError(
                "Was a partial function saved from function decorated with "
                "a PureFunction decorator ? I haven't decided how to deal with this.")
        return cls(functools.partial(func, **bound_values))


    @staticmethod
    def json_encoder(v):
        if isinstance(v, PureFunction):
            func = v.func
        elif isinstance(v, Callable):
            func = v
        else:
            raise TypeError("`PartialPureFunction.json_encoder` accepts only "
                            "`PureFunction` or Callable arguments. Received "
                            f"{type(v)}.")
        if not isinstance(func, functools.partial):
            # Make a partial with empty dict of bound arguments
            func = functools.partial(func)
        if isinstance(func.func, functools.partial):
            raise NotImplementedError("`PartialPureFunction.json_encoder` does not "
                                      "support nested partial functions at this time")
        return ("PartialPureFunction",
                serialize.serialize_function(func.func),
                func.keywords)

PureFunction.Partial = PartialPureFunction

class CompositePureFunction(PureFunction):
    """
    A lazy operation composed of an operation (+,-,*,/) and one or more terms,
    at least one of which is a PureFunction.
    Non-pure functions are not allowed as arguments.

    Typically obtained after performing operations on PureFunctions:
    >>> f = PureFunction(…)
    >>> g = PureFunction(…)
    >>> h = f + g
    >>> isinstance(h, CompositePureFunction)  # True

    .. important:: Function arithmetic must only be done between functions
       with the same signature. This is NOT checked at present, although it
       may be in the future.
    """
    def __new__(cls, func=None, *terms):
        return super().__new__(cls, func)
    def __init__(self, func, *terms):
        if func not in operator.__dict__.values():
            raise TypeError("CompositePureFunctions can only be created with "
                            "functions defined in " "the 'operator' module.")
        for t in terms:
            if isinstance(t, Callable) and not isinstance(t, PureFunction):
                raise TypeError("CompositePureFunction can only compose "
                                "constants and other PureFunctions. Invalid "
                                f"argument: {t}.")
        self.func = func
        self.terms = terms
        if not getattr(self, '__name__', None):
            self.__name__ = "composite_pure_function"

    # TODO? Use overloading (e.g. functools.singledispatch) to avoid conditionals ?
    def __call__(self, *args):
        return self.func(*(t(*args) if isinstance(t, Callable) else t
                           for t in self.terms))

    @classmethod
    def _validate_serialized(cls, value):
        "Format: ('CompositePureFunction', [op], [terms])"
        if not (isinstance(value, Sequence)
                and len(value) > 0 and value[0] == "CompositePureFunction"):
            cls.raise_validation_error(value)
        assert len(value) == 3
        assert isinstance(value[1], str)
        assert isinstance(value[2], Sequence)
        func = getattr(operator, value[1])
        terms = []
        for t in value[2]:
            if (isinstance(t, str)
                or isinstance(t, Sequence) and len(t) and isinstance(t[0], str)):
                # Nested serializations end up here.
                # First cond. catches PureFunction, second cond. its subclasses.
                terms.append(PureFunction.validate(t))
            elif isinstance(t, PlainArg):
                # Either Number or Array – str is already accounted for
                terms.append(t)
            else:
                raise TypeError("Attempted to deserialize a CompositePureFunction, "
                                "but the following value is neither a PlainArg "
                                f"nor a PureFunction: '{value}'.")
        return cls(func, *terms)

    @staticmethod
    def json_encoder(v):
        if isinstance(v, CompositePureFunction):
            assert v.func in operator.__dict__.values()
            return ("CompositePureFunction",
                    v.func.__name__,
                    v.terms)
        else:
            raise NotImplementedError
            
# Add JSON encoders
typing.add_json_encoder(PureFunction, PureFunction.json_encoder)
typing.add_json_encoder(PartialPureFunction, PartialPureFunction.json_encoder)
