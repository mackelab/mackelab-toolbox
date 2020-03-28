# -*- coding: utf-8 -*-
"""
  Mackelab utilities

Collection of useful short snippets.

Created on Tue Nov 28 2017
@author: alex
"""

# To keep keep this module as light as possible, imports required for specific
# functions are kept within that function
import logging
logger = logging.getLogger(__file__)
import builtins
from collections import OrderedDict
from collections.abc import Iterable, Callable
from enum import Enum

########################
# Functions imported by __init__.py into top level

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
                " different classes with the same name, or imported the same "
                " one more than once. "
                "This can happen when you use `importlib.reload()`."
                .format(type(obj).__name__))
    return r

########################

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

###
# Recursive setattr and getattr.

# Taken from https://stackoverflow.com/questions/31174295/getattr-and-setattr-#on-nested-objects.
# See also https://gist.github.com/wonderbeyond/d293e7a2af1de4873f2d757edd580288
####
from functools import reduce
def rsetattr(obj, attr, val):
    """Recursive setattr. Use as `setattr(foo, 'bar.baz', 1)`."""
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)
def rgetattr(obj, attr, *args):
    """Recursive getattr. Use as `getattr(foo, 'bar.baz', None)`."""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return reduce(_getattr, [obj] + attr.split('.'))

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

def strip_comments(s, comment_mark='#'):
    """
    Remove single line comments from plain text.
    Searches for `comment_mark` and removes everything follows,
    up to but excluding the next newline.
    """
    return '\n'.join(line.partition(comment_mark)[0].rstrip()
                     for line in s.splitlines())

def stablehash(o):
    """
    Builtin `hash` is not stable across sessions for security reasons.
    This function can be used when consistency of a hash is required, e.g.
    for on-disk caches.
    """
    import hashlib
    return hashlib.sha1(_tobytes(o)).hexdigest()

def _tobytes(o):
    if isinstance(o, bytes):
        return o
    elif isinstance(o, str):
        return o.encode('utf8')
    elif isinstance(o, Iterable):
        return b''.join(_tobytes(oi) for oi in o)
    else:
        return bytes(o)

def min_scalar_type(x):
    """
    Custom version of `numpy.min_scalar_type` which will not downcast a float
    unless it can be done without loss of precision.
    """
    import numpy as np  # Only import numpy if needed
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
    Similar to numpy's `real_if_close`, casts the value to an int if it close
    to an integer value.

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
    from numbers import Number, Integral, Real
    import numpy as np

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

def fully_qualified_name(o):
    """
    Return fully qualified name for a class or function

    Parameters
    ----------
    o: Type (class) or function

    Returns
    -------
    str
    """
    # Based on https://stackoverflow.com/a/13653312
    name = getattr(o, '__qualname__', getattr(o, '__name__', None))
    if name is None:
        raise TypeError("Argument has no `__qualname__` or `__name__` "
                        "attribute. Are you certain it is a class or function?")
    module = o.__module__
    if module is None or module == builtins.__name__:
        return name
    else:
        return module + '.' + name

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
            if not dictcompare(v1, v2): return False
        r = (v1 == v2)
        try:
            r = r.all()
        except AttributeError:
            pass
        # Ensure we got a bool
        if not isinstance(r, bool):
            raise ValueError(
                "Comparison of values {} and {} did not yield a boolean."
                .format(v1, v2))
        if not r: return False
    return True

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

def sciformat(num, sigdigits=1, minpower=None):
    """
    Return a string representation of `num` with a given
    number of significant digits.
    Use scientific notation if it is larger than `10^minpower`.

    Parameters
    ----------
    num: number
        number to convert to string
    sigdigits: int >= 1
        number of significant digits to keep
    minpower: int | None
        Minimum power to use scientific notation. Default value of None
        forces scientific notation.
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

def mm2in(size, *args):
    """Convert size tuple from millimetres to inches."""
    if len(args) > 0:
        if isinstance(size, Iterable):
            raise TypeError("When giving multiple arguments to `mm2in`, the "
                            "first must not be an iterable.")
        return (size/25.4,) + tuple(s/25.4 for s in args)
    elif not isinstance(size, Iterable):
        return size / 25.4
    else:
        return type(size)(s/25.4 for s in size)

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

from collections import OrderedDict, deque
from collections.abc import MutableSequence, MutableSet, MutableMapping
from copy import deepcopy
class Stash:
    """
    Classes which use variables to record internal state be in a situation
    where they need to be in a fresh default state for an operation, but
    rather than flushing their state before the operation, we would rather
    reinstate when we are done.
    We store the variables to stash as string names rather than pointers to
    the variables themselves, because pointers can change.

    Usage
    -----
    Instantiate:
        stash = Stash({vars to stash: default values})
    Stash:
        stash()
    Pop:
        stash.pop()
    """
    def __init__(self, obj, *stash_attrs):
        """
        Parameters
        ----------
        obj: class instance
            The object to which we want to attach the stash.
        stash_attrs: iterable of tuples
            Each `(attrname, default, [string])` tuple is expanded and passed to
            `add_stash_attr`.
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
                setattr(self._obj, attr, default)

    def pop(self):
        for attr, val in zip(self._stash_attrs, self._stash.pop()):
            setattr(self._obj, attr, val)


# TODO: Pre-evaluate strings to some more efficient expresison, so
#       we don't need to parse the string every time. simpleeval can do this
import simpleeval
import ast
import operator
import numpy as np
import scipy as sp
class StringFunction:
    # Replace the "safe" operators with their standard forms
    # (simpleeval implements safe_add, safe_mult, safe_exp, which test their
    #  input but this does not work with non-numerical types.)
    _operators = simpleeval.DEFAULT_OPERATORS
    _operators.update(
        {ast.Add: operator.add,
         ast.Mult: operator.mul,
         ast.Pow: operator.pow})
    # Allow evaluation to find operations in standard namespaces
    namespaces = {'np': np,
                  'sp': sp}

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
                       "expression (only numpy and scipy, as 'np' and 'sp', are "
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

class SkipCounter(int):
    """
    An integer counter which automatically skips certain values.

    The original use case for this class was for building arrays of plots:
    `plt.subplot` indexes plots left-to-right, top-to-bottom,
    making it inconvenient to arrange multiple sequences as columns.
    With a SkipCounter, we can define which grid indices to leave blank,
    and simply increment by 1 for each plot.

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

class class_or_instance_method:
    """
    Method decorator which sets `self` to be either the class (when method
    is called on the class) on the instance (when method is called on an
    instance). Adapted from https://stackoverflow.com/a/48809254.
    """
    def __init__(self, method, instance=None, owner=None):
        self.method = method
        self.instance = instance
        self.owner = owner

    def __get__(self, instance, owner=None):
        return type(self)(self.method, instance, owner)

    def __call__(self, *args, **kwargs):
        clsself = self.instance if self.instance is not None else self.owner
        return self.method(clsself, *args, **kwargs)

# A dictionary mapping string representations to numpy types like np.float32
# Note that these aren't the same as numpy dtypes
# str_to_nptype ={
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
