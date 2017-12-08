# -*- coding: utf-8 -*-
"""
  Mackelab utilities

Collection of useful short snippets.

Created on Tue Nov 28 2017
@author: alex
"""

import collections

def flatten(l):
    """Flatten any Python iterable. Taken from https://stackoverflow.com/a/2158532."""
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

