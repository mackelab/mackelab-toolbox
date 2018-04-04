# -*- coding: utf-8 -*-
"""
  Mackelab utilities

Collection of useful short snippets.

Created on Tue Nov 28 2017
@author: alex
"""

from collections import Iterable, Callable, OrderedDict

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

class SanitizedDict(dict):
    def __init__(self, *args, **kwargs):
        logger.warning("mackelab.utils.SanitizedDict is not implemented")
        super().__init__(*args, **kwargs)

class SanitizedOrderedDict(OrderedDict, SanitizedDict):
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

        return super().__init__(*args, **kwargs)

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
