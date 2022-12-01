import logging
from warnings import warn
import builtins
import operator
import inspect
from itertools import chain
from dataclasses import dataclass, field
from .utils import Singleton

from types import FunctionType
from typing import Callable, Optional

# These are added to the namespace when deserializing a function
import math
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Config(metaclass=Singleton):
    trust_all_inputs: bool = False
    default_namespace: dict = field(
        default_factory=lambda: {'__builtins__': __builtins__,
                                 'np': np, 'math': math})
config = Config()

def split_decorators(s):
    s = s.strip()
    decorator_lines = []
    while s[0] == "@":
        line, s = s.split("\n", 1)
        decorator_lines.append(line)
        s = s.lstrip()
        # print(line); print(decorator_lines); print(s)
    return decorator_lines, s


import ast
try:
    import astunparse
except ModuleNotFoundError:
    pass
import textwrap
def remove_comments(s, on_fail='warn'):
    """
    Remove comments and doc strings from Python source code passed as string.
    Based on https://stackoverflow.com/a/56285204

    This function will fail if the string `s` is not valid Python code.

    By default, if an error is raised, the string `s` is
    is returned unchanged and a warning is printed. This follows from the
    idea that `remove_comments` is a 'nice to have' feature, and we would
    rather still save / operator on an unchanged `s` than terminate.

    :param on_fail: Default: 'warn'. Change to 'raise' to raise the error
        instead of simply printing a warning.
    """
    try:
        lines = astunparse.unparse(ast.parse(textwrap.dedent(s))).split('\n')
    except Exception as e:
        if on_fail == 'warn':
            warn(f"{str(e)}\n`remove_comments` encountered the error above. "
                 "The string was returned unmodified.")
            return s
        else:
            raise e
    out_lines = []
    for line in lines:
        if line.lstrip()[:1] not in ("'", '"'):
            out_lines.append(line)
    return '\n'.join(out_lines)

def serialize_function(f):
    """
    WIP. Encode a function into a string.
    Accepts only definitions of the form::

        def func_name():
            do_something

    or::

        @decorator
        def func_name():
            do_something

    This excludes, e.g. lambdas and dynamic definitions like ``decorator(func_name)``.
    However there can be multiple decorators.

    Upon decoding, the string is executed in place with :func:`exec`, and the
    user is responsible for ensuring any names referred to within the function
    body are available in the decoder's scope.
    """
    from .typing import PureFunction  # NB: If this was in the same module as PureFunction, we could put the import at the top

    if f in operator.__dict__.values():
        "Special case serializing builtin functions"
        return f"operator.{f.__name__}"
    elif hasattr(f, '__func_src__'):
        return f.__func_src__
    elif isinstance(f, PureFunction):
        return f  # Should be serialized by Pydantic
    elif isinstance(f, FunctionType):
        s = remove_comments(inspect.getsource(f))
        decorator_lines, s = split_decorators(s)
        if not s.startswith("def "):
            raise ValueError(
                f"Cannot serialize the following function:\n{s}\n"
                "It should be a standard function defined in a file; lambda "
                "expressions are not accepted.")
        return "\n".join(chain(decorator_lines, [s]))
    else:
        raise TypeError(f"Type {type(f)} is not recognized as a "
                        "serializable function.")

def deserialize_function(s: str,
                globals: Optional[dict]=None, locals: Optional[dict]=None):
    """
    WIP. Decode a function from a string.
    Accepts strings of the following forms::

        def func_name(x, y):
            do_something

    or::

        @decorator
        def func_name(x, y):
            do_something
            
    or::
    
        lambda x,y: do_something
        
    or::
    
        x,y -> do_something

    (The last form is equivalent to a lambda function, and is provided as a
    convenience shorthand.)
    Not accepted are dynamic definitions like ``decorator(func_name)``.
    However there can be multiple decorators using the '@' syntax.

    The string is executed in place with :func:`exec`, and the arguments
    `globals` and `locals` can be used to pass defined names.
    The two optional arguments are passed on to `exec`; the deserialized
    function is injected into `locals` if passed, otherwised into `global`.

    .. note:: The `locals` namespace will not be available within the function.
       So while `locals` may be used to define decorators, generally `globals`
       is the namespace to use.

    .. note:: A few namespaces are added automatically to globals; by default,
       these are ``__builtins__``, ``np`` and ``math``. This can
       be changed by modifying the module variable
       `~mackelab_toolbox.serialization.config.default_namespace`.

    .. note:: Both `locals` and `globals` will be mutated by the call (in
       particular, the namespaces mentioned above are added to `globals` if not
       already present). If this is not desired, consider making a shallow copy
       of the dicts before passing to `deserialize`.
       
    .. note:: While _de_serialization of lambda functions is possible,
       serializing them is not currently supported in general. (This is because
       the output of `inspect.getsource` depends on the context, and we want to
       avoid fragile heuristics.)
       _Re_serialization of a lambda function is possible however.
    """
    msg = ("Cannot decode serialized function. It should be a string as "
           f"returned by inspect.getsource().\nReceived value:\n{s}")
    # First check if this is a builtin; if so, exit early
    if isinstance(s, str) and s.startswith("operator."):
        return getattr(operator, s[9:])
    # Not a builtin: must deserialize string
    if not config.trust_all_inputs:
        raise RuntimeError(
        "Deserialization of functions saved as source code requires executing "
        "them with `exec`, and is only attempted if "
        "`mackelab_toolbox.serialize.config.trust_all_inputs` is set to `True`.")
    if globals is None and locals is not None:
        # `exec` only takes positional arguments, and this combination is not possible
        raise ValueError("[deserialize]: Passing `locals` argument requires "
                         "also passing `globals`.")
    if isinstance(s, str):
        s_orig = s
        if "def " in s:
            f = _deserialize_def(s, globals, locals)
        elif "lambda " in s or s.count("->") == 1:
            f = _deserialize_lambda(s, globals, locals)
        else:
            raise ValueError(msg)
        # Store the source with the function, so it can be serialized again
        f.__func_src__ = s_orig
        return f
    else:
        raise ValueError(msg)

def _deserialize_def(s, globals, locals):
    decorator_lines, s = split_decorators(s)
    if not s[:4] == "def ":
        raise ValueError(msg)
    fname = s[4:s.index('(')].strip() # Remove 'def', anything after the first '(', and then any extra whitespace
    s = "\n".join(chain(decorator_lines, [s]))
    if globals is None:
        globals = config.default_namespace.copy()
        exec(s, globals)
        f = globals[fname]
    elif locals is None:
        globals = {**config.default_namespace, **globals}
        exec(s, globals)
        f = globals[fname]
    else:
        globals = {**config.default_namespace, **globals}
        exec(s, globals, locals)  # Adds the function to `locals` dict
        f = locals[fname]
    return f
    
def _deserialize_lambda(s, globals, locals):
    # `s` is either of the form `lambda x,…: <expr>` or `x,… -> <expr>`
    s = s.strip()
    if not s.startswith("lambda"):
        assert s.count("->") == 1
        inp, out = s.split("->")
        s = f"lambda {inp}: {out}"
    else:
        assert "->" not in s
    if globals is None:
        globals = config.default_namespace.copy()
    else:
        globals = {**config.default_namespace, **globals}
    return eval(s, globals, locals)

json_encoders = {FunctionType: serialize_function}
