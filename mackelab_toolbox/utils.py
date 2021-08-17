# -*- coding: utf-8 -*-

# ******************* Organization of this module ************************** #
#                                                                            #
# This file is split into sections, each with its own set of imports.        #
# This makes it easy to split off a section into an independent module       #
# when/if it gets big enough.                                                #
#                                                                            #
# Sections:                                                                  #
#   - Safeguards                                                             #
#   - Iteration utilities                                                    #
#   - Specialized types                                                      #
#   - String utilities                                                       #
#   - Hashing                                                                #
#   - Numerical types                                                        #
#   - Dictionary utilities                                                   #
#   - Unit conversion utilities                                              #
#   - Profiling                                                              #
#   - Stashing                                                               #
#   - Sentinel values                                                        #
#   - Introspection / Class-hacking / Metaprogramming  -> meta.py            #
#   - IPython / Jupyter Notebook utilities                                   #
#   - Misc. utilities                                                        #
#                                                                            #
# ************************************************************************** #

"""
Mackelab utilities

Collection of useful short snippets.

Created on Tue Nov 28 2017
Author: Alexandre René
"""

import logging
logger = logging.getLogger(__name__)

########################
# Safeguards – Attempt to detect certain hard to debug errors
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

########################
# Iteration utilities
import itertools
from collections.abc import Iterable
from typing import Tuple

terminating_types = (str, bytes)

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

class FixedGenerator:
    """
    Generator object which knows its length. Note this class is intended for use
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

############################
# Specialized types
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

################
# String utilities

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

###############
# Hashing
import hashlib
from collections.abc import Iterable, Mapping

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

def _tobytes(o) -> bytes:
    # byte converters for specific types
    if isinstance(o, bytes):
        return o
    elif isinstance(o, str):
        return o.encode('utf8')
    elif isinstance(o, int):
        l = ((o + (o<0)).bit_length() + 8) // 8  # Based on https://stackoverflow.com/a/54141411
        return o.to_bytes(length=l, byteorder='little', signed=True)
    elif isinstance(o, float):
        return o.hex().encode('utf8')
    # Generic byte encoders. These methods may not be ideal for each type, or
    # even work at all, so we first check if the type provides a __bytes__ method.
    elif hasattr(o, '__bytes__'):
        return bytes(o)
    elif isinstance(o, Mapping):
        return b''.join(_tobytes(k) + _tobytes(v) for k,v in o.items())
    elif isinstance(o, Iterable):
        return b''.join(_tobytes(oi) for oi in o)
    else:
        return bytes(o)

##########################
# Numerical types
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

#######################
# Dictionary utilities

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

############################
# Unit conversion utilities
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

#####################
# Profiling

from time import perf_counter
from typing import Optional, Callable

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
    when instantiating the context::
    >>> logger = logging.getLogger(__name__)
    >>> def log_time(name, Δ):
            logger.debug(f"{name} (exec time): {Δ} s")
    >>> with TimeThis("Big loop", output=log_time):
    >>>     sum(range(1000000))

    To turn off timing for all contexts without removing them from code, do
    >>> TimeThis.on = False

    .. limitation:: `TimeThis` contexts can be nested, but the reported
       between-context time is then ill-defined. The within-context time
       should be fine.
    """
    on = True
    last_t = None

    def __init__(self, name=None,
                 output: Optional[Callable[[str,float],None]]=None,
                 output_last_Δ: Optional[Callable[[str,float],None]]=None):
        if output is None:
            output = self.default_output
        if output_last_Δ is None:
            output_last_Δ = self.default_output_last_Δ
        self.name = name
        self.output = output
        self.output_last_Δ = output_last_Δ
    def __enter__(self):
        self.t1 = perf_counter()
    def __exit__(self, exc_type, exc_val, exc_tb):
        t2 = perf_counter()
        if self.on:
            if TimeThis.last_t:
                self.output_last_Δ(self.name, self.t1 - TimeThis.last_t)
            self.output(self.name, t2-self.t1)
        TimeThis.last_t = t2

    @staticmethod
    def default_output(name, Δ):
        if name:
            name += ": "
        else:
            name = ""
        if Δ < 1:
            print(f"{name}{Δ*1000:.2f} ms")
        else:
            print(f"{name}{Δ:.2f} s")
    @staticmethod
    def default_output_last_Δ(name, Δ):
        prefix = "Time since last TimeThis context: "
        if Δ < 1:
            print(f"{prefix}{Δ*1000:.2f} ms")
        else:
            print(f"{prefix}{Δ:.2f} s")


#####################
# Stashing

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


####################
# Sentinel values

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
    """
    def __new__(metacls, name, bases, dct):
        cls = super().__new__(metacls, name, bases, dct)
        cls.__instance = None
        # Don't overwrite cls.__new__ if it exists
        for supercls in cls.mro():
            # Ensure we don't assign __clsnew__ to __super_new, other we get
            # infinite recursion
            if supercls.__new__ != metacls.__clsnew__:
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

###################
# Introspection / Class-hacking / Metaprogramming

from .meta import *

#########################
# IPython / Jupyter Notebook utilities

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

from pathlib import Path
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
    css = "color: grey; text-align: right"
    def __init__(self, path=None, nchars=8, sha_prefix='#',
                 show_path='stem', show_branch=True):
        """
        :param:path: Path to the git repository. Defaults to CWD.
        :param:nchars: Numbers of SHA hash characters to display. Default: 8.
        :param:sha_prefix: Character used to indicate the SHA hash. Default: '#'
        :param:show_path: How much of the repository path to display.
            'full': Display the full path.
            'stem': (Default) Only display the directory name (which often
                    corresponds to the implied repository name)
            'none': Don't display the path at all.
        """
        repo = git.Repo(search_parent_directories=True)
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
        if show_branch:
            self.branch = repo.active_branch.name
        else:
            self.branch = ""
    def __str__(self):
        return " ".join((self.path, self.branch, self.sha))
    def __repr__(self):
        return self.__str__()
    def _repr_html_(self):
        return f"<p style=\"{self.css}\">git: {self.path} {self.branch} {self.sha}</p>"

#####################
# Misc. utilities

# TODO: Pre-evaluate strings to some more efficient expression, so
#       we don't need to parse the string every time.
#       > This is now done in `mackelab_toolbox.transform`
# TODO: Update and use with `mackelab_toolbox.transform` ?
import simpleeval
import ast
import operator
import numpy as np
class StringFunction:
    """See `mackelab_toolbox.transform.Transform`, which is more mature."""
    # Replace the "safe" operators with their standard forms
    # (simpleeval implements safe_add, safe_mult, safe_exp, which test their
    #  input but this does not work with non-numerical types.)
    _operators = simpleeval.DEFAULT_OPERATORS
    _operators.update(
        {ast.Add: operator.add,
         ast.Mult: operator.mul,
         ast.Pow: operator.pow})
    # Allow evaluation to find operations in standard namespaces
    namespaces = {'np': np}

    def __init__(self, expr, args):
        """
        Parameters
        ----------
        expr: str
            String to evaluate.
        args: iterable of strings
            The function argument names.
        """
        self.expr = expr
        self.args = args
    def __call__(self, *args, **kwargs):
        names = {nm: arg for nm, arg in zip(self.args, args)}
        names.update(kwargs)  # FIXME: Unrecognized args ?
        names.update(self.namespaces)  # FIXME: Overwriting of arguments ?
        try:
            res = simpleeval.simple_eval(
                self.expr,
                operators=self._operators,
                names=names)
        except simpleeval.NameNotDefined as e:
            e.args = ((e.args[0] +
                       "\n\nThis may be due to a module function in the transform "
                       "expression (only numpy, as 'np', are "
                       "available by default).\nIf '{}' is a module or class, you can "
                       "make it available by adding it to the function namespace: "
                       "`StringFunction.namespaces.update({{'{}': {}}})`.\nSuch a line would "
                       "typically be included at the beginning of the execution script "
                       "(it does not need to be in the same module as the one where "
                       "the string function is defined, as long as it is executed first)."
                       .format(e.name, e.name, e.name),)
                      + e.args[1:])
            raise
        return res

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
