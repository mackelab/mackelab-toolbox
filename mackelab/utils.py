# -*- coding: utf-8 -*-
"""
  Mackelab utilities

Collection of useful short snippets.

Created on Tue Nov 28 2017
@author: alex
"""

import collections

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
    if not isinstance(terminate, collections.Iterable):
        terminate = (terminate,)
    else:
        terminate = tuple(terminate)
    # Flatten `l`
    for el in l:
        if (isinstance(el, collections.Iterable)
            and not isinstance(el, (str, bytes) + terminate)):
            yield from flatten(el, terminate=terminate)
        else:
            yield el

