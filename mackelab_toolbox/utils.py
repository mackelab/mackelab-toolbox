# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version,-kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# ********************* Organization of this module **************************
# *                                                                          *
# * This file is split into sections, each with its own set of imports.      *
# * This makes it easy to split off a section into an independent module     *
# * when/if it gets big enough.                                              *
# *                                                                          *
# * Sections:                                                                *
# *   - Safeguards                                                           *
# *   - Iteration utilities                                                  *
# *   - Specialized types                                                    *
# *   - String utilities                                                     *
# *   - Hashing                                                              *
# *   - Numerical types                                                      *
# *   - Dictionary utilities                                                 *
# *   - Unit conversion utilities                                            *
# *   - Profiling                                                            *
# *   - Stashing                                                             *
# *   - Sentinel values                                                      *
# *   - Introspection / Class-hacking / Metaprogramming  -> meta.py          *
# *   - IPython / Jupyter Notebook utilities                                 *
# *   - Logging utilities                                                    *
# *   - Misc. utilities                                                      *
# *                                                                          *
# ****************************************************************************

# %%
"""
Mackelab utilities

Collection of useful short snippets.

Created on Tue Nov 28 2017
Author: Alexandre René
"""

from reprlib import repr  # Used by 'profiling'; imported here since it replaces standard `repr`
import logging
logger = logging.getLogger(__name__)

# %% tags=["remove-cell"]
exenv = "script"
if __name__ != "__main__":
    exenv = "module"
elif False:
    pass

    # %% tags=["remove-cell"]
    exenv = "jbook"

    # %% tags=["skip-execution", "remove-cell"]
    exenv = "notebook"

# %% [markdown]
# ## Safeguards ########################
# Attempt to detect certain hard to debug errors

# %%
from collections.abc import Iterable
import builtins

def isinstance(obj, class_or_tuple):
    """
    Wraps `isinstance` with an extra check to detect if a return value of
    `False` is due to a reimported module.
    """
    r = builtins.isinstance(obj, class_or_tuple)
    if not r:
        same_name = False
        if builtins.isinstance(class_or_tuple, Iterable):
            same_name = any(type(obj).__name__ == cls.__name__
                            for cls in class_or_tuple)
        else:
            same_name = (type(obj).__name__ == class_or_tuple.__name__)
        if same_name:
            logger.warning(
                "Object type does not match any of those given, but shares its "
                "name '{}' with at least one of them. You may have imported "
                "different classes with the same name, or imported the same "
                "one more than once. "
                "This can happen when you use `importlib.reload()`."
                .format(type(obj).__name__))
    return r

# %% [markdown]
# ## Iteration utilities ########################

# %%
import itertools
from collections.abc import Iterable
from typing import Tuple

# TODO: Add merge theano_shim.config terminating types with utils.terminating_types
_terminating_types = {str, bytes}
terminating_types = tuple(_terminating_types)  # NB: isinstance() only works with tuples
    # TODO: Move to a config object with @property
    #       When we do this, remove the update from typing_module and include in @property
    # TODO: Merge with theano_shim.config.terminating_types (see .theano)

def flatten(*l, terminate=None):
    """
    Flatten any Python iterable. Taken from https://stackoverflow.com/a/2158532.

    Parameters
    ----------
    l: iterable
        Iterable to flatten
    terminate: tuple of types
        Tuple listing types which should not be expanded. E.g. if l is a nested
        list of dictionaries, specifying `terminate=dict` or `terminate=(dict,)`
        would ensure that the dictionaries are treated as atomic elements and
        not expanded.
        By default terminates on the types listed in `terminating_types`.
    """
    # Normalize `terminate` argument
    if terminate is None:
        terminate = terminating_types
    if not isinstance(terminate, Iterable):
        terminate = (terminate,)
    else:
        terminate = tuple(terminate)
    # Flatten `l`
    for el in l:
        if (isinstance(el, Iterable)
            and not isinstance(el, terminate)):
            for ell in el:
                yield from flatten(ell, terminate=terminate)
        else:
            yield el

class SizedIterable:
    """
    Iterable object which knows its length. Note this class is intended for use
    with iterables that don't provide their length (e.g. generators), and as
    such cannot check that the provided length is correct.
    """
    def __init__(self, iterable, length):
        self.iterable = iterable
        self.length = length
    def __iter__(self):
        return iter(self.iterable)
    def __len__(self):
        return self.length

FixedGenerator = SizedIterable  # Alias for old name

def index_iter(shape: Tuple[int]) -> itertools.product:
    """
    Return an iterator which produces each multi-index corresponding to the
    given shape.

    >>> from mackelab_toolbox import index_iter
    >>> list(index_iter((2,2))
        [(0, 0), (0, 1), (1, 0), (1, 1)]

    This is functionally equivalent, but roughly 10x faster, than using
    `numpy.ndindex` ::

    >>> list(np.ndindex((2,2)))

    (The timing with `ndindex` is the comparable to using `np.nditer` to extract
    only the shape, which suggests that may be the source of the overhead.)
    """
    return itertools.product(*(range(s) for s in shape))

# %% [markdown]
# ## Specialized types ############################

# %%
import abc
from enum import Enum
from collections import OrderedDict
from collections.abc import Iterable, Callable

class LongList(list):
    """
    A subclass of `list` which abridges its representation when it is long,
    in the same way NumPy shortens the display of long arrays.
    Will shorten if:
        - The full string representation is longer than `self.threshold`
          (default: 300)
        - AND the number elements is greater than 5

    The threshold value can be changed by assignment::
    >>> from macklab_toolbox.utils import LongList
    >>> l1 = LongList(range(700))
    >>> print(l1)
    '<List of 700 elements> [0, 1...698, 699]'
    >>> l2 = LongList(range(8))
    [0, 1, 2, 3, 4, 5, 6, 7]
    >>> LongList.threshold = 10            # Change default
    >>> print(l2)
    <List of 8 elements> [0, 1...6, 7]
    >>> l2.threshold = 300                 # Change only for l2
    >>> print(l2)
    [0, 1, 2, 3, 4, 5, 6, 7]
    """
    threshold = 300
    def __repr__(self):
        s = super().__repr__()
        if len(self) <= 5 or len(s) <= self.threshold:
            return s
        else:
            return f"<List of {len(self)} elements> " \
                   f"[{self[0]}, {self[1]}...{self[-2]}, {self[-1]}]"


class OrderedEnum(Enum):
    """
    Copied from python docs:
    https://docs.python.org/3.6/library/enum.html#orderedenum
    """
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

class SanitizedDict(dict):
    """Not implemented. See `SanitizedOrderedDict`."""
    def __init__(self, *args, _warn=True, **kwargs):
        if _warn:
            logger.warning("mackelab_toolbox.utils.SanitizedDict is not implemented")
        super().__init__(*args, **kwargs)

class SanitizedOrderedDict(SanitizedDict, OrderedDict):
    """
    Subclass of OrderedDict with sanitized keys.
    Any key query is passed through a user-defined 'sanitization' function
    before searching for a match in the dictionary.

    This is intended for user-facing dictionaries, to allow the user to access
    entries without needing to remember the exact key, as long as the one they
    specify is "close enough".
    """

    def __init__(self, *args, sanitize="", ignore_whitespace=True, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs: Same as for OrderedDict
        sanitize: function, or list of characters
            If a function, used to sanitize inputs.
            If a list of characters, sanitization is to remove all characters
            in that list.
            Can be left unspecified if the only thing we want to sanitize is
            whitespace.
        ignore_whitespace: bool
            Whether to also ignore all whitespace. Ignored if `sanitize` is
            a function.
        """
        if isinstance(sanitize, Callable):
            self.sanitize = sanitize
        elif isinstance(sanitize, Iterable):
            def f(s):
                for c in sanitize:
                    s = s.replace(c, '')
                if ignore_whitespace:
                    # Remove all whitespace
                    s = ''.join(s.split())
                return s
            self.sanitize = f

        return super().__init__(*args, _warn=False, **kwargs)

    def __setitem__(self, key, value):
        if isinstance(key, Iterable) and not isinstance(key, (str, bytes)):
            key = type(key)([self.sanitize(k) for k in key])
        else:
            key = self.sanitize(key)
        return super().__setitem__(key, value)
    def __getitem__(self, key):
        if isinstance(key, Iterable) and not isinstance(key, (str, bytes)):
            key = type(key)([self.sanitize(k) for k in key])
        else:
            key = self.sanitize(key)
        return super().__getitem__(key)

    def newdict(self, *args, **kwargs):
        """
        Return an empty OrderedDict with the same sanitization.
        """
        return SanitizedOrderedDict(*args, sanitize=self.sanitize, **kwargs)

class TypeDict(OrderedDict):
    """
    A dictionary using types as keys. A key will match any of its subclasses;
    if there are multiple possible matches, the earliest in the dictionary
    takes precedence.
    """
    def __getitem__(self, key):
        if not isinstance(key, type):
            raise TypeError(f"TypeDict keys must be types")
        for k, v in self.items():
            if issubclass(key, k):
                return v
        else:
            raise KeyError(f"Type {key} is not a subclass of any of this "
                           "TypeDict's type keys.")

    def __setitem__(self, key, value):
        if not isinstance(key, type):
            raise TypeError(f"TypeDict keys must be types")
        return super().__setitem__(key, value)

    def __contains__(self, key):
        return (isinstance(key, type) and
                any(issubclass(key, k) for k in self))

@staticmethod
@abc.abstractmethod
def abstractattribute():
    """
    A placeholder for an abstract attribute; use as

    >>> import abc
    >>> from mackelab_toolbox.utils import abstractattribute
    >>> class Foo(abc.ABC):
    >>>   bar = abstractattribute

    A caveat is that the printed message reports an abstract method rather
    than abstract attribute.
    """
    pass

###
# Recursive setattr and getattr.

# Taken from https://stackoverflow.com/questions/31174295/getattr-and-setattr-#on-nested-objects.
# See also https://gist.github.com/wonderbeyond/d293e7a2af1de4873f2d757edd580288
####
from functools import reduce
def rsetattr(obj, attr, val):
    """
    Recursive setattr. Use as `setattr(foo, 'bar.baz', 1)`.

    Source: https://stackoverflow.com/questions/31174295/getattr-and-setattr-#on-nested-objects
    See also: https://gist.github.com/wonderbeyond/d293e7a2af1de4873f2d757edd580288
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)
def rgetattr(obj, attr, *args):
    """
    Recursive getattr. Use as `getattr(foo, 'bar.baz', None)`.

    Source: https://stackoverflow.com/questions/31174295/getattr-and-setattr-#on-nested-objects
    See also: https://gist.github.com/wonderbeyond/d293e7a2af1de4873f2d757edd580288
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return reduce(_getattr, [obj] + attr.split('.'))

# %% [markdown]
# ## String utilities ################

# %%
from collections.abc import Iterable
import numpy as np

def array_to_str(arr: np.ndarray, precision: int=3) -> str:
    """
    Convert an array into a string with the given precision.

    TODO: Change precision to sig digits.
    """
    if not getattr(arr, 'ndim', True) or not isinstance(arr, Iterable):
        return f"{arr:.{precision}}"
    else:
        return "[" + ", ".join(array_to_str(x) for x in arr) + "]"

def array_to_latex_string(arr: np.ndarray, precision: int=3) -> str:
    raise NotImplementedError

def sciformat(num, sigdigits=1, minpower=None) -> str:
    """
    Return a string representation of `num` with a given number of significant
    digits.
    Use scientific notation if it is larger than `10^minpower`.

    The output is intended mostly for figure production and uses TeX
    notation ('10^' for the power, '\\cdot' for multiplication).

    .. Note:: Core Python formatting has a `'g'` option, along with variants,
    which does something similar. The main difference is that `sciformat`
    will always print the specified number of significant digits, and its
    output is suited for TeX formatting.

    Parameters
    ----------
    num: number
        number to convert to string
    sigdigits: int >= 1
        number of significant digits to keep
    minpower: int | None
        Minimum power to use scientific notation. Default value of None
        forces scientific notation.

    Returns
    -------
    str
    """
    import math
    if sigdigits < 1:
        logger.warning("`sciformat`: Number of significant digits should be "
                       "greater or eqal to 1.")
    p = int(math.floor(math.log10(num)))
        # Note if we use 'np' instead of 'math':
        # don't use `astype(int)` here, as otherwise 10**p fails with p<0
    if minpower is not None and p < minpower:
        decimal_positions = sigdigits - p - 1
        numstr = round(num, decimal_positions)
        if decimal_positions >= 0:
            return str(int(num))
        else:
            return str(num)
    m = round(num / 10**p, sigdigits)
    if sigdigits == 1:
        m = int(m)
    if m == 1:
        mstring = ""
    else:
        mstring = str(m)
    if p != 0:
        pstring = "10^{{{}}}".format(p)
    else:
        pstring = "10^{{{}}}".format(p)
    if mstring != "" and pstring != "":
        dotstr = " \\cdot "
    else:
        dotstr = ""
    return mstring + dotstr + pstring

def strip_comments(s, comment_mark='#'):
    """
    Remove single line comments from plain text.
    Searches for `comment_mark` and removes everything follows,
    up to but excluding the next newline.
    """
    return '\n'.join(line.partition(comment_mark)[0].rstrip()
                     for line in s.splitlines())

from string import Formatter
class ExtendedFormatter(Formatter):
    """An extended format string formatter

    Formatter with extended conversion symbols for upper/lower case and
    capitalization.

    Source: https://stackoverflow.com/a/46160537
    """
    def convert_field(self, value, conversion):
        """ Extend conversion symbol
        Following additional symbol has been added
        * l: convert to string and low case
        * u: convert to string and up case

        default are:
        * s: convert with str()
        * r: convert with repr()
        * a: convert with ascii()
        """

        if conversion == "u":
            return str(value).upper()
        elif conversion == "l":
            return str(value).lower()
        elif conversion == "c":
            return str(value).capitalize()
        # Do the default conversion or raise error if no matching conversion found
        super().convert_field(value, conversion)

        # return for None case
        return value
formatter = ExtendedFormatter()
def format(s, *args, **kwargs):
    return formatter.format(s, *args, **kwargs)

# %% [markdown]
# ## Hashing ###############

# %%
import hashlib
from collections.abc import Iterable, Sequence, Collection, Mapping
from dataclasses import is_dataclass, fields
from enum import Enum
# from .utils import terminating_types, TypeDict

def stablehash(o):
    """
    The builtin `hash` is not stable across sessions for security reasons.
    This `stablehash` can be used when consistency of a hash is required, e.g.
    for on-disk caches.

    For obtaining a usable digest, see the convenience functions
    `stablehexdigest`, `stablebytesdigest` and `stableintdigest`.
    `stabledigest` is a synonym for `stableintdigest` and is suitable as the
    return value of a `__hash__` method.

    .. Note:: For exactly the reason stated above, none of the hash functions
       in this module are cryptographically secure.

    .. Tip:: The following function can be used to calculate the likelihood
       of a hash collision::

           def p_coll(N, M):
             '''
             :param N: Number of distinct hashes. For a 6 character hex digest,
                this would be 16**6.
             :param M: Number of hashes we expect to create.
             '''
             logp = np.sum(np.log(N-np.arange(M))) - M*np.log(N)
             return 1-np.exp(logp)

    Returns
    -------
    HASH object
    """
    return hashlib.sha1(_tobytes(o))
def stablehexdigest(o) -> str:
    """
    Returns
    -------
    str
    """
    return stablehash(o).hexdigest()
def stablebytesdigest(o) -> bytes:
    """
    Returns
    -------
    bytes
    """
    return stablehash(o).digest()
def stableintdigest(o, byte_len=4) -> int:
    """
    Suitable as the return value of a `__hash__` method.

    .. Note:: Although this method is provided, note that the purpose of a
       digest (a unique fingerprint) is not the same as the intended use of
       the `__hash__` magic method (fast hash tables, in particular for
       dictionaries). In the latter case, a certain degree of hash collisions
       is in fact desired, since that is required for the most efficient tables.
       Because this function uses SHA1 to obtain almost surely unique digests,
       it is much slower than typical `__hash__` implementations. This can
       become noticeable if it is involved in a lot of dictionary lookups.

    Parameters
    ----------
    o : object to hash (see `stablehash`)
    byte_len : int, Optional (default: 4)
        Number of bytes to keep from the hash. A value of `b` provides at most
        `8**b` bits of entropy. With `b=4`, this is 4096 bits and 10 digit
        integers.

    Returns
    -------
    int
    """
    return int.from_bytes(stablebytesdigest(o)[:byte_len], 'little')
stabledigest = stableintdigest

# Extra functions to converting values to bytes specialized to specific types
# This is the mechanism to use to add support for types outside your own control
# (i.e. for which it is not possible to add a __bytes__ method).
_byte_converters = TypeDict()

def _tobytes(o) -> bytes:
    """
    Utility function for converting an object to bytes. This is used for the
    state digests, and thus is designed with the following considerations:

    1. Different inputs should, with high probability, return different byte
       sequences.
    2. The same inputs should always return the same byte sequence, even when
       executed in a new session (in order to satisfy the 'stable' description).
       Note that this precludes using an object's `id`, which is sometimes
       how `hash` is implemented.
       It also precludes using `hash` or `__hash__`, since that function is
       randomly salted for each new Python session.
    3. It is NOT necessary for the input to be reconstructable from
       the returned bytes.

    ..Note:: To avoid overly complicated byte sequences, the distinction
       guarantee is not preserved across types. So `_tobytes(b"A")`,
       `_tobytes("A")` and `_tobytes(65)` all return `b'A'`.
       So multiple inputs can return the same byte sequence, as long as they
       are unlikely to be used in the same location to mean different things.

    **Supported types**
    - None
    - bytes
    - str
    - int
    - float
    - Enum
    - type
    - dataclasses, as long as their fields are supported types.
      + NOTE: At present we allow hashing both frozen and non-frozen dataclasses
    - Any object implementing a ``__bytes__`` method
    - Mapping
    - Sequence
    - Collection
    - Any object for which `bytes(o)` does not raise an exception

    TODO?: Restrict to immutable objects ?

    Raises
    ------
    TypeError:
        - If `o` is a consumable Iterable.
        - If `o` is of a type for which `_to_bytes` is not implemented.
    """
    # byte converters for specific types
    if o is None:
        # TODO: Would another value more appropriately represent None ? E.g. \x00 ?
        return b""
    elif isinstance(o, bytes):
        return o
    elif isinstance(o, str):
        return o.encode('utf8')
    elif isinstance(o, int):
        l = ((o + (o<0)).bit_length() + 8) // 8  # Based on https://stackoverflow.com/a/54141411
        return o.to_bytes(length=l, byteorder='little', signed=True)
    elif isinstance(o, float):
        return o.hex().encode('utf8')
    elif isinstance(o, Enum):
        return _tobytes(o.value)
    elif isinstance(o, type):
        return _tobytes(f"{o.__module__}.{o.__qualname__}")
    elif is_dataclass(o):
        # DEVNOTE: To restrict this to immutable dataclasses, check `o.__dataclass_params__.frozen`
        return _tobytes(tuple((f.name, getattr(o, f.name)) for f in fields(o)))
    # Generic byte encoders. These methods may not be ideal for each type, or
    # even work at all, so we first check if the type provides a __bytes__ method.
    elif hasattr(o, '__bytes__'):
        return bytes(o)
    elif type(o) in _byte_converters:
        return _byte_converters[type(o)](o)
    elif isinstance(o, Mapping) and not isinstance(o, terminating_types):
        return b''.join(_tobytes(k) + _tobytes(v) for k,v in o.items())
    elif isinstance(o, Sequence) and not isinstance(o, terminating_types):
        return b''.join(_tobytes(oi) for oi in o)
    elif isinstance(o, Collection) and not isinstance(o, terminating_types):
        return b''.join(_tobytes(oi) for oi in sorted(o))
    elif isinstance(o, Iterable) and not isinstance(o, terminating_types):
        raise ValueError("Cannot compute a stable hash for a consumable Iterable.")
    else:
        try:
            return bytes(o)
        except TypeError:
            # As an ultimate fallback, attempt to use the same decomposition
            # that pickle would
            try:
                state = o.__getstate__()
            except Exception:
                breakpoint()
                raise TypeError("mackelab_toolbox.utils._tobytes does not know how "
                                f"to convert values of type {type(o)} to bytes. "
                                "One way to solve this is may be to add a "
                                "`__bytes__` method to that type. If that is "
                                "not possible, you may also add a converter to "
                                "mackelab_toolbox.utils.byte_converters.")
            else:
                return _tobytes(state)


# %% [markdown]
# ## Numerical types ##########################

# %%
from numbers import Number, Integral, Real
import numpy as np

def min_scalar_type(x):
    """
    Custom version of `numpy.min_scalar_type` which will not downcast a float
    unless it can be done without loss of precision.
    """
    if np.issubdtype(x, np.floating):
        if x == np.float16(x):
            return np.dtype('float16')
        elif x == np.float32(x):
            return np.dtype('float32')
        else:
            return np.dtype('float64')
    elif np.issubdtype(x, np.complexfloating):
        raise NotImplementedError
    elif np.issubdtype(x, np.integer):
        return np.min_scalar_type(x)
    else:
        raise TypeError("Unsupported type '{}'.".format(type(x)))

def int_if_close(x, tol=100, npint=None, allow_power10=True):
    """
    Similar to numpy's `real_if_close`, casts the value to an int if it is
    close to an integer value.

    Parameters
    ----------
    x : numerical value
        Input array.
    tol : float
        Tolerance in machine epsilons.
    npint: numpy type
        Integer type to use for numpy inputs. Defaults to `numpy.int64`.
    allow_power10: bool
        If True, will also try to round to a power of 10, so e.g. 0.009999998
        would be replaced with 0.01.
    Returns
    -------
    out : int | float
    """

    if npint is None: npint = np.int64
    if ( isinstance(x, Integral)
         or (hasattr(x, 'dtype') and np.issubdtype(x.dtype, np.integer)) ):
        return x
        #pass
    if isinstance(x, np.ndarray):
        cond = (abs(x - np.rint(x)) < tol * np.finfo(x.dtype.type).eps).all()
    else:
        cond = abs(x - np.rint(x)) < tol * np.finfo(x).eps
    if cond:
        if isinstance(x, (np.ndarray, np.number)):
            #return np.rint(x).astype(npint)
            x = np.rint(x).astype(npint)
        else:
            #return int(round(x))
            x = int(round(x))
    else:
        #return x
        pass

    if allow_power10:
        pwr = int_if_close(np.log10(x), tol, npint, allow_power10=False)
        if isinstance(pwr, Integral):
            x = 10**int(pwr)   # Need `int()` b/c numpy doesn't allow neg. pwr

    return x

def less_close(x1, x2, rtol=1e-5, atol=1e-8):
    """
    'less than or equal' test where 'isclose' is used to test
    equality. `atol` and `rtol` are parameters for `isclose`.
    """
    return x1 < x2 or np.isclose(x1, x2, rtol=rtol, atol=atol)

def greater_close(x1, x2, rtol=1e-5, atol=1e-8):
    """
    'greater than or equal' test where 'isclose' is used to test
    equality. `atol` and `rtol` are parameters for `isclose`.
    """
    return x1 > x2 or np.isclose(x1, x2, rtol=rtol, atol=atol)

def broadcast_shapes(*shapes):
    """
    Compute the shape resulting from a broadcasting operation.

    If A1, A2 and A3 have shapes S1, S2 and S3, then the resulting array
    A = A1 * A2 * A3 (where * can be any broadcasting operation) has shape S.
    This function takes S1, S2, S3 as arguments and returns S.
    """
    As = (np.broadcast_to(np.ones(1), shape) for shape in shapes)
    return np.broadcast(*As).shape

# A dictionary mapping string representations to numpy types like np.float32
# Note that these aren't the same as numpy dtypes
# str_to_NPValue ={
#     'int8'   : np.int8,
#     'int16'  : np.int16,
#     'int32'  : np.int32,
#     'int64'  : np.int64,
#     'uint8'  : np.uint8,
#     'uint16' : np.uint16,
#     'uint32' : np.uint32,
#     'uint64' : np.uint64,
#     'float16': np.float16,
#     'float32': np.float32,
#     'float64': np.float64
#     }


# %% [markdown]
# ## Dictionary utilities #######################

# %%
from collections import OrderedDict

def sort_dict(d: dict, key: bool=None, reverse: bool=False) -> dict:
    """
    Sort a dictionary in-place. The `key` and `reverse` arguments are
    passed on to `sorted`.

    .. Note:: This method is more efficient when `d` is an OrderedDict.
    """
    sorted_key = sorted(d, key=key, reverse=reverse)
    if isinstance(d, OrderedDict):
        for k in sorted_key:
            d.move_to_end(k)
    else:
        sorted_dict = {k: d[k] for k in sorted_key}
        d.clear()
        for k, v in sorted_dict.items():
            d[k] = v
    return d

def comparedicts(dict1, dict2):
    """
    Recursively compare nested dictionaries. Works correctly with Numpy values
    (calls `all()` on the result)
    """
    if set(dict1) != set(dict2): return False
    for k, v1 in dict1.items():
        v2 = dict2[k]
        if isinstance(v1, dict):
            if not isinstance(v2, dict): return False
            if not comparedicts(v1, v2): return False
        r = (v1 == v2)
        try:
            r = r.all()
        except AttributeError:
            pass
        # Ensure we got a bool
        if not isinstance(r, (bool, np.bool_)):
            raise ValueError(
                "Comparison of values {} and {} did not yield a boolean."
                .format(v1, v2))
        if not r: return False
    return True

def prune_dict(data: dict, keys: set):
    """
    Recursively remove a set of keys from nested dictionaries.
    Returns a copy of `data`.
    """
    data = data.copy()
    if not isinstance(keys, set):
        raise TypeError("`prune_dict` expects a 'set' as `keys` "
                        f"argument; received {keys} (type: {type(keys)}).")
    for k in keys:
        data.pop(k, None)
    for k, v in data.items():
        if isinstance(v, dict):
            data[k] = prune_dict(v, keys)
    return data

# %% [markdown]
# ## Unit conversion utilities ############################

# %%
from collections.abc import Iterable

def mm2in(size, *args):
    """
    Convert size tuple from millimetres to inches.

    Examples
    --------
    >>> from mackelab_toolbox.utils import mm2in
    >>> mm2in(45)
    1.771…
    >>> mm2in(10, 20, 30, 40)
    (0.393…, 0.787…, 1.181…, 1.574…)
    >>> mm2in([50, 60, 70])
    [1.968…, 2.362…, 2.755…]
    """
    if len(args) > 0:
        if isinstance(size, Iterable):
            raise TypeError("When giving multiple arguments to `mm2in`, the "
                            "first must not be an iterable.")
        return (size/25.4,) + tuple(s/25.4 for s in args)
    elif not isinstance(size, Iterable):
        return size / 25.4
    else:
        return type(size)(s/25.4 for s in size)

# %% [markdown]
# ## Profiling #####################
#
# - `TimeThis` computes the execution time for the code inside its context.
#   Use this for finding *where*, in a segment of code, a big bottleneck is located.
#
# - `timeit` is a convenience wrapper for a function in the *timeit* module.
#   It *repeatedly executes* the given code a large number of times, in order
#   to get higher precision timings.
#   Use one of the `timeit` variants for *comparing* the speed of two alternative implementations.


# %%
import sys
from sys import getsizeof, stderr
import builtins
import re
from time import perf_counter
from itertools import chain
from collections import deque, ChainMap
from statistics import mean, stdev
from dataclasses import dataclass
from timeit import repeat as timeit_repeat

# from reprlib import repr  # At top of this module

from typing import Optional, Union, Callable, List, Literal

def nocolored(s, *arg, **kwargs):
    return s
try:
    from termcolor import colored
except ModuleNotFoundError:
    colored = nocolored

# %% [markdown]
# ### `TimeThis`
#
# Profiling helper for slow (> millisecond) code segments.
# This context manager is not particularly careful with overhead, and hence isn't appropriate for submillisecond measures.

# %%
class TimeThis:
    """
    Profiling helper for slow (> millisecond) code segments.
    This decorator is not particularly careful with overhead, and hence isn't
    appropriate for submillisecond measures.

    Usage::
    >>> with TimeThis("Big loop"):
    >>>     sum(range(1000000))
        Big loop: 14.73 ms

    This is roughly equivalent to::
    >>> t1 = time.perf_counter()
    >>> sum(range(1000000))
    >>> t2 = time.perf_counter()
    >>> print("Big loop: {t2-t2:.2f} ms")
        Big loop: 14.73 ms

    Another thing it does is keep a global time counter, to measure the time
    since the last call. So running the code again would now print two lines::
    >>> with TimeThis("Big loop"):
    >>>     sum(range(1000000))
        Time since last timing context: 429.25 s
        Big loop: 14.73 ms

    Thus, if there are multiple TimeThis contexts within a function, one can
    see if a time consuming process exists between them.

    The default is to print the timing result with `print`. This can be changed
    by providing a different callable when instantiating the context::
    >>> logger = logging.getLogger(__name__)
    >>> with TimeThis("Big loop", output=logger.debug):
    >>>     sum(range(1000000))
    To make the new output function the default, assign it to the class::
    >>> TimeThis.output = staticmethod(logger.debug)

    By default, times longer than 1 second and 1 min are highlighted by
    printing them in color (repectively in blue and yellow). This can be turned
    off for a specific context by passing ``color=False``, or globally by
    setting ``TimeThis.color = False``. The threshold values are hard-coded.

    For complete control over the output, the pair of methods
    `output_function` and `output_last_Δ` are provided. (The latter prints the
    "Time since last timing context" line when `TimeThis` is called more than
    once.) These can also be provided to the context:
    >>> def log_time(name, Δ):
            logger.debug(f"{name} (exec time): {Δ} s")
    >>> def log_time_Δ(name, Δ):
    >>>     pass  # Deactivate printing of inter-context timing.
    >>> with TimeThis("Big loop",
                      output_function=log_time, output_last_Δ=log_time_Δ):
    >>>     sum(range(1000000))
    Or assigned as defaults::
    >>> TimeThis.output_function = staticmethod(log_time)
    >>> TimeThis.output_last_Δ = staticmethod(log_time_Δ)

    To turn off timing for all contexts without removing them from code, do
    >>> TimeThis.on = False

    .. limitation:: `TimeThis` contexts can be nested, but the reported
       inter-context time is then ill-defined. The within-context time
       should be fine.
    """
    on     = True
    last_t = None
    output = print
    color  = True

    def __init__(self, name=None, color: Optional[bool]=None,
                 output         : Optional[Callable[[str],None]]=None,
                 output_function: Optional[Callable[[str,float],None]]=None,
                 output_last_Δ  : Optional[Callable[[str,float],None]]=None):
        self.name = name
        if color is not None:
            self.color = color
        if output is not None:
            self.output = output
        if output_function is not None:
            self.output_function
        if output_last_Δ is not None:
            self.output_last_Δ = output_last_Δ
    def __enter__(self):
        self.t1 = perf_counter()
    def __exit__(self, exc_type, exc_val, exc_tb):
        t2 = perf_counter()
        if self.on:
            if TimeThis.last_t:
                self.output_last_Δ(self.name, self.t1 - TimeThis.last_t)
            self.output_function(self.name, t2-self.t1)
        TimeThis.last_t = t2

    # Colours with good contrast on both dark & light bg:
    # blue < yellow < magenta (dark bg)
    # yellow < blue < magenta (light bg)
    @property
    def colored(self):
        if self.color:
            return colored
        else:
            return nocolored
    def output_function(self, name, Δ):
        if name:
            name += ": "
        else:
            name = ""
        if Δ < 1:
            self.output(f"{name}{Δ*1000:.2f} ms")
        elif Δ < 60:
            self.output(self.colored(f"{name}{Δ:.2f} s", 'blue'))
        else:
            self.output(self.colored(f"{name}{Δ/60:.1f} s", 'yellow'))
    def output_last_Δ(self, name, Δ):
        prefix = "Time since last TimeThis context: "
        if Δ < 1:
            self.output(f"{prefix}{Δ*1000:.2f} ms")
        elif Δ < 60:
            self.output(self.colored(f"{prefix}{Δ:.2f} s", 'blue'))
        else:
            self.output(self.colored(f"{prefix}{Δ/60:.2f} s", 'yellow'))

# %% [markdown]
# ### `timeit`
#
# Wrapper for timeit.repeat. For when you want the convenience of
# IPython's `%timeit` magic outside of IPython.
#
# Difference with *timeit.repeat*:
# - Defaults to using `globals()` for the globals.
#   (Pass `globals=None` to prevent this.)
# - By default, the *minimum* is shown instead of the mean (as recommended in the *timeit* docs).
# - Returns a `TimeitResult` object, which provides statistics as
#   attributes and pretty printing.
#

# %%
@dataclass
class TimeitResult:
    number : int
    repeat : int
    results: List[int]
    unit   : Literal['ns','μs','ms','s']="s"
    fmt    : str="{min:.2f} ± {std:.2f} {unit} per loop (min ± std. dev. of {repeat} runs, {number} loops each)"
    per_loop: bool=False
        # Different from %timeit: we report min instead of mean, and always use the same units
    def __str__(self):
        attrs = [attr for attr in dir(self) if attr in self.fmt]
        return self.fmt.format(**{attr: getattr(self, attr) for attr in attrs})
    def _repr_pretty_(self, p, cycle):
        assert not cycle
        return p.text(str(self))
    def _repr_markdown_(self):
        s = str(self)
        units = type(self).__annotations__["unit"].__args__  # Get unit strings from type annotation
        # Heuristic: highlight numbers with decimals (excludes 'number' and 'repeat')
        s = re.sub(r"(\d+\.\d+)", r"**\1**", s)
        # Highlight units – must be surrounded by spaces
        s = re.sub(" (" + "|".join(units) + ") ", r" **\1** ", s)
        return s
    def __post_init__(self):
        # Convert results to per loop values
        if not self.per_loop:
            self.results = [r/self.number for r in self.results]
            self.per_loop = True
        # Determine best units
        if self.unit == "s":  # Hack: other units can be forced, but not 's'
            for scale, unit in [(1, "s"), (1e-3, "ms"), (1e-6, "μs"), (1e-9, "ns")]:
                if self.min > scale:
                    break
            # NB: If we exhaust the loop, we get ns as desired
            self.results = [r/scale for r in self.results]
            self.unit = unit
    @property
    def min(self):
        return min(self.results)
    @property
    def max(self):
        return max(self.results)
    @property
    def mean(self):
        return mean(self.results)
    @property
    def std(self):
        return 0 if len(self.results) <= 1 else stdev(self.results)

def timeit(stmt='pass', setup='pass', repeat=5, number=1000000,
           globals='globals', fmt=None):
    """
    Wrapper for timeit.repeat. For when you want the convenience of
    IPython's `%timeit` magic outside of IPython.

    Difference with *timeit.repeat*:
    - Defaults to using `globals()` in the caller's frame for the globals.
      (Pass `globals=None` to prevent this.)
    - Returns a `TimeitResult` object, which provides statistics as
      attributes and pretty printing.

    Use `fmt` to customize the display string. For even more control,
    instead of a str, pass a subclass of `TimeitResult` to `fmt`.
    
    .. Todo:: Add 'auto' option to `number`, and emulate how `%timeit` selects
       the number of loops automatically.
    """

    if globals == "globals":
        globals = sys._getframe(1).f_globals
    res = timeit_repeat(
        stmt, setup, repeat=repeat, number=number, globals=globals)
    result_container = TimeitResult
    kwargs = {}
    if fmt is None:
        pass
    elif isinstance(fmt, str):
        kwargs["fmt"] = fmt
    elif isinstance(fmt, type) and issubclass(fmt, TimeitResult):
        result_container = fmt
    return result_container(
        number=number, repeat=repeat, results=res, unit='s', **kwargs)


# %% [markdown]
# ### `total_size`
#
# Return the total memory used by an object, including that of references it contains.
#
# #### Extending
#
# Modules can add handlers for their types to `_total_size_handlers`.

# %%
def total_size(o, handlers={}, verbose=False):
    """
    Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.

    If a type is unrecognized, a TypeError is raised.

    To support additional types, additional handlers can be passed as arguments.
    Librairies can also add their types to the `_total_size_handlers` dictionary.

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    .. Note:: To make it easier to override defaults, handlers are tried in
       reverse order:
       - All user specified `handlers` before global `_total_size_handlers`
       - Within a dict, in reverse order (latest ones first).

    .. Note:: Heavily based on https://code.activestate.com/recipes/577504/,
       with the following changes:
       - Python 2 support is dropped
       - An error is raised if no handler is found for a type
       - Handlers for non-container types must also be provided, and should return None
       - Handlers can return `NotImplemented`, letting other handlers try
         to handle an object
       - Entries in the handler dictionary can use strings for their key,
         allowing to define handlers without worrying whether a user has the
         required module installed (and without causing an unnecessary import)
    """
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    # Initialize handlers
    # Loop over keys, and replace strings keys with corresponding type
    _all_handlers = ChainMap(handlers, _total_size_handlers)  # Give precedence to user-specified handlers
    all_handlers = {}
    for key, handler in reversed(list(_all_handlers.items())):
        if isinstance(key, type):
            all_handlers[key] = handler

        elif isinstance(key, str):
            if "." not in key:
                raise NotImplementedError(
                    "I'm not sure in which situation one would need a str key "
                    "that doesn't include a module in total_size's handlers "
                    "dict; waiting for a use case.")
            else:
                # 'key' is a dotted name: first part indicates the package
                # where type is defined.
                # Problem: given "foo.bar.MyType", we don't know whether to do
                # `from foo.bar import MyType` or `from foo import bar.MyType`.
                # So we try all combinations, giving precedence to more
                # specified packages (so `from foo.bar` would be tried first).
                # For each combination, we check whether that package is
                # imported:  if not, it is not possible (without irresponsible
                # manipulation of imports) for `o` to have this type, and thus
                # we don't need to check for it
                modname = key
                name = ""
                for k in range(key.count(".")):
                    modname, name_prefix = key.rsplit(".", 1)
                    name = name_prefix + ("." + name if name else "")
                    mod = sys.modules.get(modname)
                    if mod:
                        T = getattr(mod, name)
                        if not isinstance(T, type):
                            raise TypeError(f"`total_size` has a handler for "
                                            f"type {key}, but this is not a "
                                            f"type. (It is a {type(T)}).")
                        all_handlers[T] = handler
                        break
                # If we exit the loop without finding anything, it means the
                # required module is not available.

        else:
            raise TypeError("Keys for `total_size` dictionary of handlers "
                            f"should be types or strings; received {type(key)}.")

    # Define the recursive `sizeof` function
    def sizeof(o):
        # if not isinstance(o, (tuple, str, dict, int, float)):
        #     import pdb; pdb.set_trace()
        if o is None or id(o) in seen:  # Do not double count the same object. `None` acts as a termination value, and is anyway always 'seen'
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                it = handler(o)
                if it is NotImplemented:
                    continue
                elif it is not None:
                    s += sum(map(sizeof, it))
                break
        else:
            raise TypeError("`total_size` has no handler for objects of type "
                            f"{type(o)}")
        return s

    # Apply the recursive `sizeof` function
    return sizeof(o)

_scalar_handler = lambda o: None
_dict_handler = lambda d: chain.from_iterable(d.items())
_total_size_handlers = {
    int: _scalar_handler,
    float: _scalar_handler,
    str: _scalar_handler,
    tuple: iter,
    list: iter,
    deque: iter,
    dict: _dict_handler,
    set: iter,
    frozenset: iter,
   }

# %%

def total_size_handler(T: Union[str, type]):
    """
    Register a handler for applying the function `total_size` to objects of
    type `T`. To avoid unnecessary imports, it is allowed to specify types as
    import strings: if the containing module has not been imported, the
    handler is skipped (since then it is extremely unlikely that a variable
    of that type has been defined).

    Handlers are executed in the reverse order in which they are specified.
    """
    def decorator(handler):
        if T in _total_size_handlers:
            logger.warning(f"Overwriting `total_size` handler for type {T}.")
        _total_size_handlers[T] = handler
    return decorator

# %% [markdown]
# Measuring sizes of NumPy arrays:
# - `.nbytes` returns the number of bytes consumed by the elements of the array, and *only*.
#   This excludes overhead (104 bytes) but includes the memory of elements in views.
# - `getsizeof` it returns only the size of the overhead (since the actual data is referenced).
# - Doing `.nbytes` + overhead is unsatisfactory, because it can lead to double counting.
#   For example, in the following case, it would count almost twice the actually used memory for array `A`:
#
#       [A[:-1], A[1:]]
#
# - When given a view, we return the size of the view itself, plus the *fullsize* of its underlying array.
#   The `seen` tracking in `total_size` ensures that the underlying array is counted only once.
#
#   In the case where an object *only* uses a partial view of an array, this would produce a larger number
#   then the memory actually used. However, in such cases one could argue that the full array size should
#   still be counted (since it is somehow used), and since that leads to much simpler computer logic, it
#   is the interpretation we take.

# %%
def _numpy_array_handler(o):
    # Recall that the recursive `sizeof` already calls `sys.getsizeof` on `o`)
    # Question: Are there ndarray attributes we should also return ?
    if o.base is not None:
        yield o.base

def _numpy_dtype_handler(o):
    ".. Caution:: This implementation is not highly researched"
    if o is o.base:
        # This is not a new dtype; it's not adding memory
        # (QUESTION: Can there be bases which aren't defined in the numpy packages ? Maybe those should still be counted ?)
        return None

    for attr in dir(o):
        val = getattr(o, attr)
        baseval = getattr(o.base, attr)
        if val is not baseval:  # Objects allocated as part of the class definition shouldn't be counted against the instance's footprint
            yield val     # DTypes seem to be C structs, so their keys don't add to the memory footprint

# Use a string key so that we don't introduce an unnecessary (and potentially
# fatal) numpy import
_total_size_handlers["numpy.ndarray"] = _numpy_array_handler
_total_size_handlers["numpy.dtype"] = _numpy_dtype_handler

# %% [markdown]
# The bit of code below compares `total_size` to `sys.getsizeof`, which doesn't include the size of the elements in containers.

# %%
if exenv in {"jbook", "notebook"}:
    import sys
    from tabulate import tabulate

    b = b"abcdefghijklmnopqrst"
    s = "abcdefghijklmnopqrst"
    flst = [float(x) for x in range(20)]

    print("Confirm that utils.getsizeof does not include size of contents:\n")
    print("Size of b:                ", sys.getsizeof(b))
    print("Size of s:                ", sys.getsizeof(s))
    print("Size of list containing s:", sys.getsizeof([s]))
    print("total_size([s]):         ", total_size([s]), f" (= {sys.getsizeof(s)} + {sys.getsizeof([s])})")
    print()

    sizes = range(5)
    types = (str, tuple, list, set, frozenset)

    print("=====================================")
    print("sys.getsizeof (values are independent of content):\n\n")
    for data, _types in zip((s, flst), (types, types[:-1])):
        print("data: ", data)
        print()
        res = []
        if data == s:
            # Include sizes with a pure bytes object, for comparison with str
            res += [["bytes (byte data)", *(sys.getsizeof(b[:n]) for n in sizes)]]
        res += [[T.__name__, *(sys.getsizeof(T(data[:n])) for n in sizes)]
                for T in _types]
        print(tabulate(res, headers = ["# elements →"] + list(sizes)))
        print()

    print("=====================================")
    print("total_size (values depend on content):\n\n")
    for data, _types in zip((s, flst), (types, types[:-1])):
        print("data: ", data)
        print()
        res = []
        if data == s:
            # Include sizes with a pure bytes object, for comparison with str
            res += [["bytes (byte data)", *(sys.getsizeof(b[:n]) for n in sizes)]]
        res += [[T.__name__, *(total_size(T(data[:n])) for n in sizes)]
                for T in _types]
        print(tabulate(res, headers = ["# elements →"] + list(sizes)))
        print()


    from collections import defaultdict
    import numpy as np

    sizes = [0, 10, 100, 1000]
    sysgso = defaultdict(lambda: [])
    gfs = defaultdict(lambda: [])
    for n in sizes:
        iarr32 = np.arange(n, dtype=np.int32)
        farr64 = np.arange(n, dtype=np.float64)
        sysgso["int32"].append(sys.getsizeof(iarr32))
        sysgso["float64"].append(sys.getsizeof(farr64))
        sysgso["int32   (A[:])"].append(sys.getsizeof(iarr32[:]))
        sysgso["float64 (A[:])"].append(sys.getsizeof(farr64[:]))
        gfs["int32"].append(total_size(iarr32))
        gfs["float64"].append(total_size(farr64))
        gfs["int32   (A[:])"].append(total_size(iarr32[:]))
        gfs["float64 (A[:])"].append(total_size(farr64[:]))
        gfs["int32   (A[::4])"].append(total_size(iarr32[::4]))
        gfs["float64 (A[::4])"].append(total_size(farr64[::4]))
        gfs["int32 ([A, A])"].append(total_size([iarr32, iarr32]))

    print("=====================================")
    print("getsizeof – NumPy array\n")
    print(tabulate([[k] + v for k,v in sysgso.items()],
                   headers = ["# elements →"] + list(sizes)))

    print("\n\n=====================================")
    print("total_size – NumPy array\n")
    print(tabulate([[k] + v for k,v in gfs.items()],
                   headers = ["# elements →"] + list(sizes)))


# %% [markdown]
# ## Stashing #####################

# %%
from collections import OrderedDict, deque
from collections.abc import MutableSequence, MutableSet, MutableMapping
from copy import deepcopy

class Stash:
    """
    Classes which use variables to record internal state can be in a situation
    where they need to be in a fresh default state for an operation, but
    rather than flushing their state before the operation, we would rather
    reinstate it when we are done.
    We store the variables to stash as string names rather than pointers to
    the variables themselves, because pointers can change.

    See `mackelab/sinn.histories.History` for a real-life example.

    Usage
    -----
    Instantiate:
        stash = Stash(obj, (attr1 to stash, default val1),
                           (attr2 to stash, default val2), ...)
        # => stash associated to `obj`
    Stash:
        stash()
        # Stash attributes of `obj` and replace with their default vals.
    Pop:
        stash.pop()
        # Set attributes back to their most recently stash values, and
        # remove those values from the stash.
    """
    def __init__(self, obj, *stash_attrs):
        """
        Parameters
        ----------
        obj: class instance
            The object to which we want to attach the stash.
        stash_attrs: iterable of tuples
            Each `(attrname, default, [bool])` tuple is expanded and passed to
            `add_stash_attr`.
            If `default` is a string, it is interpreted as an attribute name
            by default; see `add_stash_attr`.
        """
        self._obj = obj
        self._stash_attrs = OrderedDict()
        self._stash = deque()
        for t in stash_attrs:
            self.add_stash_attr(*t)

    def add_stash_attr(self, attr, default, default_is_string=False):
        """
        Parameters
        ----------
        attr: str
            Class attribute name
        default: str | variable
            If string, interpreted as a class attribute name. To prevent this
            and treat the default as a plain string, set `default_is_string`.
            Otherwise, treated as a default value (e.g. `True` or `0` and set
            as is.
        default_is_string: bool
            Set to true to treat the default value as a string instead of an
            attribute name. Has no effect if `default` is not a string.
        """
        if isinstance(default, (MutableSequence, MutableSet, MutableMapping)):
            default = deepcopy(default)
        self._stash_attrs[attr] = (default, default_is_string)

    def __call__(self):
        self._stash.append(
            [getattr(self._obj, attr) for attr in self._stash_attrs])
        for attr, (default, default_is_string) in self._stash_attrs.items():
            if isinstance(default, str) and not default_is_string:
                setattr(self._obj, attr, getattr(self._obj, default))
            else:
                # If we don't copy a mutable default when we use it, it can
                # get modified and become incorrect for future stashes
                if isinstance(default, (MutableSequence, MutableSet, MutableMapping)):
                    default = deepcopy(default)
                setattr(self._obj, attr, default)

    def pop(self):
        for attr, val in zip(self._stash_attrs, self._stash.pop()):
            setattr(self._obj, attr, val)


# %% [markdown]
# ## Sentinel values ####################

# %%
class Singleton(type):
    """Singleton metaclass
    Based on the pattern for numpy._globals._NoValue

    Although singletons are usually an anti-pattern, I've found them useful in
    a few cases, notably for a configuration class storing dynamic attributes
    in the form of properties.
    Before using a singleton, consider these alternate, more
    pythonic options:

      - Enum
        For defining unique constants
      - SimpleNamespace
        For defining configuration containers

    Example
    -------
    >>> from mackelab_toolbox.utils import Singleton
    >>> import sys
    >>>
    >>> class Config(metaclass=Singleton):
    >>>     def num_modules(self):
    >>>         return len(sys.modules)
    >>> config = Config()

    Attempting to create a new instance just returns the original one.
    >>> config2 = Config()
    >>> config is config2  # True
    """
    def __new__(metacls, name, bases, dct):
        cls = super().__new__(metacls, name, bases, dct)
        cls.__instance = None
        # We need to patch __clsnew__ into __new__.
        # 1. Don't overwrite cls.__new__ if one of the parents is already a Singleton
        #    (Otherwise, the child will try to assign two or more different __new__
        #     functions to __super_new)
        if any(isinstance(supercls, metacls) for supercls in cls.mro()[1:]):
            pass
        # 2. Don't overwrite cls.__new__ if it exists
        else:
            for supercls in cls.mro():
                # Ensure we don't assign __clsnew__ to __super_new, other we get
                # infinite recursion
                if supercls.__new__ != metacls.__clsnew__:
                    assert not hasattr(cls, f"_{metacls.__name__}__super_new"), "Multiple Singleton metaclasses have clashed in an unexpected way."
                    cls.__super_new = supercls.__new__
                    break
        cls.__new__ = metacls.__clsnew__
        return cls
    @staticmethod
    def __clsnew__(cls, *args, **kwargs):
        # ensure that only one instance exists
        if not cls.__instance:
            cls.__instance = cls.__super_new(cls, *args, **kwargs)
        return cls.__instance

def sentinel(name, repr_str=None):
    """
    Create a singleton object to use as sentinel value.
    Based on the pattern for numpy._globals._NoValue

    A sentinel value is an object, like NumPy's `_NoValue`, that can be assigned
    to a variable to signal a particular state.
    It is guaranteed not equal to any other value (in contrast to `None` or 0),
    and guaranteed to only ever have one instance (so can be used in tests like
    `a is sentinel_object`).

    Example
    -----
    >>> class _NoValueType(metaclass=SentinelMeta)
            pass
    >>> NoValue = _NoValueType("<no value>")
    >>>
    >>> def f(x, y=NoValue):
    >>>     if y is NoValue:
    >>>         y = 1
    >>>     return x*y
    """
    if name not in sentinel.__instances:
        if repr_str is None:
            repr_str = f"<{name}>"
        def __repr__(self):
            return repr_str
        SentinelCls = Singleton(
            name, (), {'__repr__': __repr__})
        sentinel.__instances[name] = SentinelCls()
    return sentinel.__instances[name]
sentinel.__instances = {}

# %% [markdown]
# ## Introspection / Class-hacking / Metaprogramming ###################

# %%
from .meta import *


# %% [markdown]
# ## IPython / Jupyter Notebook utilities #########################

# %%
class PDF:
    """
    Create an object from a pdf file which allows it to be viewed in a notebook.
    Also exports well to latex with nbconvert.
    https://stackoverflow.com/a/19470377
    """
    def __init__(self, pdf, size=(200,200)):
        self.pdf = pdf
        self.size = size

    def _repr_html_(self):
        return ('<iframe src={0} width={1[0]} height={1[1]}></iframe>'
                .format(self.pdf, self.size))

    def _repr_latex_(self):
        return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)

# Functions for displaying source code in an IPython cell
# These are especially useful for documenting a module in a notebook
import inspect
try:
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter
    from IPython.display import HTML, display
    pygments_loaded = True
except ModuleNotFoundError:
    pygments_loaded = False

if pygments_loaded:
    pythonlexer = PythonLexer(encoding='chardet')
    htmlformatter = HtmlFormatter()
class CodeStr:
    """
    Return a highlighted string of code for display in a notebook output cell.
    Todo: correct export to Latex
    """
    def __init__(self, code):
        """
        Parameters
        ----------
        code: str
        """
        self.code = code
        display(HTML("""
        <style>
        {pygments_css}
        </style>
        """.format(pygments_css=HtmlFormatter().get_style_defs('.highlight'))))

    def _repr_html_(self):
        return HTML(data=highlight(self.code, pythonlexer, htmlformatter)) \
            ._repr_html_()

def Code(*obj, sep=''):
    """
    Extract object source code with `inspect.getsource` and return a string
    with syntax highlighting suitable for a notebook output cell.

    Parameters
    ----------
    *obj: objects for which we want to print the source code
    sep:  (keyword only) Optional separator, if multiple objects are given.
          This is appended to the already present newline.
    """
    # s
    src = sep.join([inspect.getsource(s) for s in obj])
    return CodeStr(src.strip())  # .strip() removes final newline

from typing import Union
from pathlib import Path
from datetime import datetime
from socket import gethostname
try:
    import git
except ModuleNotFoundError:
    git = None

class GitSHA:
    """
    Return an object that nicely prints the SHA hash of the current git commit.
    Displays as formatted HTML in a Jupyter Notebook, otherwise a simple string.

    .. Hint:: This is especially useful for including a git hash in a report
       produced with Jupyter Book. Adding a cell `GitSHA() at the bottom of
       notebook with the tag 'remove-input' will print the hash with no visible
       code, as though it was part of the report footer.

    Usage:
    >>> GitSHA()
    myproject main #3b09572a
    """
    css: str= "color: grey; text-align: right"
    # Default values used when a git repository can’t be loaded
    path  : str="No git repo found"
    branch: str=""
    sha   : str=""
    hostname: str=""
    timestamp: str=None
    def __init__(self, path: Union[None,str,Path]=None, nchars: int=8,
                 sha_prefix: str='#', show_path: str='stem',
                 show_branch: bool=True, show_hostname: bool=False,
                 datefmt: str="%Y-%m-%d"):
        """
        :param:path: Path to the git repository. Defaults to CWD.
        :param:nchars: Numbers of SHA hash characters to display. Default: 8.
        :param:sha_prefix: Character used to indicate the SHA hash. Default: '#'.
        :param:show_path: How much of the repository path to display.
            'full': Display the full path.
            'stem': (Default) Only display the directory name (which often
                    corresponds to the implied repository name)
            'none': Don't display the path at all.
        :param:datefmt: The format string to pass to ``datetime.strftime``.
            To not display any time at all, use an empty string.
            Default format is ``2000-12-31``.
        """
        ## Set attributes that should always work (don't depend on repo)
        if datefmt:
            self.timestamp = datetime.now().strftime(datefmt)
        else:
            self.timestamp = ""
        if show_hostname:
            self.hostname = gethostname()
        else:
            self.hostname = ""
        ## Set attributes that depend on repository
        # Try to load repository
        if git is None:
            # TODO?: Add to GitSHA a message saying that git python package is not installed ?
            return
        try:
            repo = git.Repo(search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            # Skip initialization of repo attributes and use defaults
            return
        self.repo = repo
        self.sha = sha_prefix+repo.head.commit.hexsha[:nchars]
        if show_path.lower() == 'full':
            self.path = repo.working_dir
        elif show_path.lower() == 'stem':
            self.path = Path(repo.working_dir).stem
        elif show_path.lower() == 'none':
            self.path = ""
        else:
            raise ValueError("Argument `show_path` should be one of "
                             "'full', 'stem', or 'none'")
        self.branch = ""
        if show_branch:
            try:
                self.branch = repo.active_branch.name
            except TypeError:  # Can happen if on a detached head
                pass

    def __str__(self):
        return " ".join((s for s in (self.timestamp, self.hostname, self.path, self.branch, self.sha)
                         if s))
    def __repr__(self):
        return self.__str__()
    def _repr_html_(self):
        hoststr = f"&nbsp;&nbsp;&nbsp;host: {self.hostname}" if self.hostname else ""
        return f"<p style=\"{self.css}\">{self.timestamp}{hoststr}&nbsp;&nbsp;&nbsp;git: {self.path} {self.branch} {self.sha}</p>"

# %% [markdown]
# ## Logging utilities #####################

# %%
class lesslog:
    """
    Context manager for temporarily changing the log level of loggers.
    By default, all messages from the specified loggers are silenced except
    those of level ERROR.
    (If no logger is specified, all loggers are set to this level.)
    The log level of each logger is restored when exiting the context.

    Example usage::

    >>> import logging
    >>> import matplotlib.pyplot as plt
    >>> from mackelab_toolbox.utils import lesslog, sciformat
    >>> logging.basciConfig()
    >>> with lesslog("matplotlib"):
    >>>   # The line below would normally print a warning
    >>>   plt.scatter(range(5), range(5), c=(.1, .3, .5))
    """


    def __init__(self, loggers="", level=logging.ERROR):
        if isinstance(loggers, str):
            loggers = [loggers]
        self.loggers = [logging.getLogger(logger) for logger in loggers]
        self.in_context_level = level
        self.out_of_context_levels = None
    def __enter__(self):
        self.out_of_context_levels = [logger.level for logger in self.loggers]
        for logger in self.loggers:
            logger.level = self.in_context_level
    def __exit__(self, exc_type, exc_val, exc_tb):
        for logger, level in zip(self.loggers, self.out_of_context_levels):
            logger.level = level
        self.out_of_context_levels

# %% [markdown]
# ## Misc. utilities #####################

# # %%
# # TODO: Pre-evaluate strings to some more efficient expression, so
# #       we don't need to parse the string every time.
# #       > This is now done in `mackelab_toolbox.transform`
# # TODO: Update and use with `mackelab_toolbox.transform` ?
# import simpleeval
# import ast
# import operator
# import numpy as np
# class StringFunction:
#     """See `mackelab_toolbox.transform.Transform`, which is more mature."""
#     # Replace the "safe" operators with their standard forms
#     # (simpleeval implements safe_add, safe_mult, safe_exp, which test their
#     #  input but this does not work with non-numerical types.)
#     _operators = simpleeval.DEFAULT_OPERATORS
#     _operators.update(
#         {ast.Add: operator.add,
#          ast.Mult: operator.mul,
#          ast.Pow: operator.pow})
#     # Allow evaluation to find operations in standard namespaces
#     namespaces = {'np': np}

#     def __init__(self, expr, args):
#         """
#         Parameters
#         ----------
#         expr: str
#             String to evaluate.
#         args: iterable of strings
#             The function argument names.
#         """
#         self.expr = expr
#         self.args = args
#     def __call__(self, *args, **kwargs):
#         names = {nm: arg for nm, arg in zip(self.args, args)}
#         names.update(kwargs)  # FIXME: Unrecognized args ?
#         names.update(self.namespaces)  # FIXME: Overwriting of arguments ?
#         try:
#             res = simpleeval.simple_eval(
#                 self.expr,
#                 operators=self._operators,
#                 names=names)
#         except simpleeval.NameNotDefined as e:
#             e.args = ((e.args[0] +
#                        "\n\nThis may be due to a module function in the transform "
#                        "expression (only numpy, as 'np', are "
#                        "available by default).\nIf '{}' is a module or class, you can "
#                        "make it available by adding it to the function namespace: "
#                        "`StringFunction.namespaces.update({{'{}': {}}})`.\nSuch a line would "
#                        "typically be included at the beginning of the execution script "
#                        "(it does not need to be in the same module as the one where "
#                        "the string function is defined, as long as it is executed first)."
#                        .format(e.name, e.name, e.name),)
#                       + e.args[1:])
#             raise
#         return res

from collections.abc import Iterable
class SkipCounter(int):
    """
    An integer counter which automatically skips certain values.

    The original use case for this class was for building arrays of plots:
    `plt.subplot` indexes plots left-to-right, top-to-bottom,
    making it inconvenient to arrange multiple sequences as columns.
    With a SkipCounter, we can define which grid indices to leave blank,
    and simply increment by 1 for each plot.
    I've found better ways of doing this since then
    (assigning names to the axes returned by `plt.subplots(n,m)`), but maybe
    there are other uses for this.

    Usage example
    -------------
    from scipy import stats
    # Create a list of four distributions
    dists = [stats.norm(0, 1), stats.norm(3, 1),
             stats.gamma(a=1), stats.gamma(a=3)]
    x = np.linspace(-3, 5)
    ys = [d.pdf(x) for d in dists]

    # Plot the distributions on a 3x3 grid, skipping middle and corners
    k = SkipCounter(0, skips=(1, 3, 5, 7, 9))
    for y in ys:
        k += 1
        plt.subplot(3, 3, k)
        plt.bar(x, y, width=1)

    # TODO: Find a better example for the corners
    k = SkipCounter(0, skips=(2, 4, 5, 6, 8))
    for y in ys:
        k += 1
        plt.subplot(3, 3, k)
        plt.plot(x, y)
    """
    def __new__(cls, value=0, skips=None, low=None, high=None,
                *args, **kwargs):
        counter = super().__new__(cls, value)
        counter.skips = skips if skips is not None else ()
        if not isinstance(counter.skips, Iterable):
            counter.skips = (counter.skips,)
        counter.low = low if low is not None else value
        counter.high = high
        return counter

    def __repr__(self):
        return "<SkipCounter (skips: {}) @ {}>".format(self.skips,
                                                       self)

    def _get_new_value(self, i):
        value = int(self)
        step = np.sign(i)
        high = (self.high if self.high is not None
                else value + abs(i) + len(self.skips))
        while i != 0:
            value += step
            if value not in self.skips:
                i -= step
            # Safety assertions to avoid infinite loops
            assert(value >= self.low)
            assert(value <= high)
        return value

    def __add__(self, i):
        return SkipCounter(self._get_new_value(i),
                           self.skips, self.low, self.high)

    def __sub__(self, i):
        return self + -i
