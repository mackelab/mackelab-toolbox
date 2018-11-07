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
from collections import Iterable, Callable, OrderedDict
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

def flatten(l, terminate=()):
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
    """
    # Normalize `terminate` argument
    if not isinstance(terminate, Iterable):
        terminate = (terminate,)
    else:
        terminate = tuple(terminate)
    # Flatten `l`
    for el in l:
        if (isinstance(el, Iterable)
            and not isinstance(el, (str, bytes) + terminate)):
            yield from flatten(el, terminate=terminate)
        else:
            yield el

def strip_comments(s, comment_mark='#'):
    """
    Remove single line comments from plain text.
    Searches for `comment_mark` and removes everything follows,
    up to but excluding the next newline.
    """
    return '\n'.join(line.partition(comment_mark)[0].rstrip()
                     for line in s.splitlines())

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
        #return x
        pass
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
            logger.warning("mackelab.utils.SanitizedDict is not implemented")
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

# TODO: Find a way to pre-evaluate strings to some more efficient expresison, so
#       we don't need to parse the string every time.
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
