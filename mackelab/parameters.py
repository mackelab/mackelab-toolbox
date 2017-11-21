# -*- coding: utf-8 -*-
"""
Provides the 'expand_params' method for expanding a parameter description strings.
This allows to use one descriptor to specify multiple parameter sets.



Created on Wed Sep 20 13:19:53 2017

@author: alex
"""

from collections import deque, OrderedDict, namedtuple
from types import SimpleNamespace
import numpy as np
import scipy as sp

from . import iotools

##########################
# Transformed parameters
##########################

import simpleeval
import ast
import operator

class Transform:
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

    def __new__(cls, *args, **kwargs):
        # Make calling Transform on a Transform instance just return the instance
        if len(args) > 0 and isinstance(args[0], Transform):
            # Don't return a new instance; just return this one
            return args[0]
        else:
            return super().__new__(cls)

    def __init__(self, transform_desc):
        # No matter what we do in __new__, __init__ is called, so we need
        # to check this again.
        if not isinstance(transform_desc, Transform):
            xname, expr = transform_desc.split('->')
            self.xname = xname.strip()
            self.expr = expr.strip()


    def __call__(self, x):
        names = {self.xname: x}
        names.update(self.namespaces)
        try:
            res = simpleeval.simple_eval(
                self.expr,
                operators=Transform._operators,
                names=names)
        except simpleeval.NameNotDefined as e:
            e.args = ((e.args[0] +
                       "\n\nThis may be due to a module function in the transform "
                       "expression (only numpy and scipy, as 'np' and 'sp') are "
                       "available by default.\nIf '{}' is a module or class, you can "
                       "make it available by adding it to the transform namespace: "
                       "`Transform.namespaces.update({{'{}': {}}})`.\nSuch line would "
                       "typically be included at the beginning of the execution script "
                       "(it does not need to be in the same module as the one where "
                       "the transform is defined, as long as it is executed before)."
                       .format(e.name, e.name, e.name),)
                      + e.args[1:])
            raise
        return res

    @property
    def desc(self):
        return self.xname + " -> " + self.expr

class TransformedVar:
    def __init__(self, desc, *args, orig=None, new=None):
        """
        Should only pass either `orig` or `new`
        """
        if len(args) > 0:
            raise TypeError("TransformedVar() takes only one positional argument.")
        if not( (orig is None) != (new is None) ):  #xor
            raise ValueError("Exactly one of `orig`, `new` must be specified.")
        self.to = Transform(desc.to)
        self.back = Transform(desc.back)
        if orig is not None:
            #assert(shim.issymbolic(orig))
            self.orig = orig
            self.new = self.to(self.orig)
        elif new is not None:
            #assert(shim.issymbolic(new))
            self.new = new
            self.orig = self.back(new)
        names = [nm.strip() for nm in desc.name.split('->')]
        assert(len(names) == 2)
        if self.orig.name is None:
            self.orig.name = names[0]
        else:
            assert(self.orig.name == names[0])
        if self.new.name is None:
            self.new.name = names[1]
        else:
            assert(self.new.name == names[1])

class NonTransformedVar:
    """Provides an interface consistent with TransformedVar."""
    def __init__(self, orig):
        self.orig = orig
        self.to = lambda x: x
        self.back = lambda x: x
        self.new = orig

###########################
# Parameter file expansion
###########################

ExpandResult = namedtuple("ExpandResult", ['strs', 'done'])

def expand_params(param_str, fail_on_unexpanded=False, parser=None):
    """
    Expand a parameter description into multiple descriptions.
    The default parser expands contents of the form "*[a, b, ...]" into multiple
    files with the starred expression replaced by "a", "b", ….

    The default parser expands on '*' and recognizes '()', '[]' and '{}' as
    brackets. This can be changed by explicitly passing a custom parser.
    The easiest way to obtain a custom parser is to instantiate one with
    Parser, and change its 'open_brackets', 'close_brackets', 'separators' and
    'expanders' attributes.

    Parameters
    ----------
    param_str: str
        The string descriptor for the parameters.
    fail_on_unexpanded: bool (default False)
        (Optional) Specify whether to fail when an expansion character is found
        but unable to be expanded. By default such an error is ignored, but
        if your parameter format allows it, consider setting it to True to
        catch formatting errors earlier.
    parser: object
        (Optional) Only required if one wishes to replace the default parser.
        The passed object must provide an 'extract_blocks' method, which itself
        returns a dictionary of Parser.Block elements.

    Returns
    -------
    list of strings
        Each element is a complete string description for oneset of parameters.
    """

    if parser is None:
        parser = Parser()
    param_strs = [strip_comments(param_str)]
    done = False
    while not done:
        res_lst = [_expand(s, fail_on_unexpanded, parser) for s in param_strs]
        assert(isinstance(res.strs, list) for res in res_lst)
        param_strs = [s for res in res_lst for s in res.strs]
        done = all(res.done for res in res_lst)

    return param_strs

def expand_param_file(param_path, output_path,
                      fail_on_unexpanded=False, parser=None,
                      max_files=1000):
    """
    Load the file located at 'param_path' and call expand_params on its contents.
    The resulting files are saved at the location specified by 'output_pathn',
    with a number appended to each's filename to make it unique.

    Parameters
    ----------
    param_path: str
        Path to a parameter file.
    output_path: str
        Path to which the expanded parameter files will be saved. If this is
        'path/to/file.ext', then each will be saved as 'path/to/file_1.ext',
        'path/to/file_2.ext', etc.
    fail_on_unexpanded: bool (default False)
        (Optional) Specify whether to fail when an expansion character is found
        but unable to be expanded. By default such an error is ignored, but
        if your parameter format allows it, consider setting it to True to
        catch formatting errors earlier.
    parser: object
        (Optional) Only required if one wishes to replace the default parser.
        The passed object must provide an 'extract_blocks' method, which itself
        returns a dictionary of Parser.Block elements.
    max_files: int
        (Optional) Passed to iotools.get_free_file. Default is 1000.

    Returns
    -------
    None
    """
    with open(param_path, 'r') as f:
        src_str = f.read()

    pathnames = []
    for ps in expand_params(src_str, fail_on_unexpanded, parser):
        f, pathname = iotools.get_free_file(output_path, bytes=False,
                                            force_suffix=True,
                                            max_files=max_files)
        f.write(ps)
        f.close()
        pathnames.append(pathname)

    #print("Parameter files were written to " + ', '.join(pathnames))
        # TODO: Use logging
    return pathnames

def strip_comments(s):
    return '\n'.join(line.partition('#')[0].rstrip()
                     for line in s.splitlines())

def _expand(s, fail_on_unexpanded, parser):
    #param_strs = [s]  # Start with a single parameter string
    blocks = parser.extract_blocks(s)
    for i, c in enumerate(s):
        if c in parser.expanders:
            if i+1 in blocks:
                block = blocks[i+1]
                expanded_str = [s[:i] + str(el) + s[block.stop:]
                                for el in block.elements.values()]
                # Return list of expanded strings and continuation flag
                return ExpandResult(expanded_str, False)
            elif fail_on_unexpanded:
                raise ValueError("Expansion identifier '*' at position {} "
                                 "must be followed by a bracketed expression.\n"
                                 "Context: '{}'."
                                 .format(i, s[max(i-10,0):i+10]))
    # Found nothing to expand; return the given string and termination flag
    # Wrap the string in a list to match the expected format
    return ExpandResult([s], True)

class Parser():
    """Basic parser for nested structures with opening and closing brackets."""

    # Default values.
    # These can be changed on a per instance by reassigning the attributes
    open_brackets = ['[', '{']
    close_brackets = [']', '}']  # Order must match that of open_brackets
        # Parentheses are not included because "*(…)" tends to appear in
        # mathematical expressions
    separators = [',']
    expanders = ['*']

    def get_closer(self, c):
        idx = self.open_brackets.index(c)
        if idx != -1:
            return self.close_brackets[idx]
    def get_opener(self, c):
        idx = self.close_brackets.index(c)
        if idx != -1:
            return self.open_brackets[idx]

    def extract_blocks(self, s):
        block_stack = deque()  # Unclosed blocks
        blocks = {}       # Closed blocks, keyed by their starting index
        for i, c in enumerate(s):
            if c in self.open_brackets:
                block = Block(start=i, opener=c, closer=self.get_closer(c))
                if len(block_stack) > 0:
                    block_stack[-1].blocks.append(block)
                block_stack.append(block)
            elif c in self.close_brackets:
                block = block_stack[-1]
                if len(block_stack) == 0:
                    raise ValueError("Unmatched closing bracket '{}' at position {}."
                                     .format(c, i))
                if c != block_stack[-1].closer:
                    raise ValueError("Closing bracket '{}' at position {} does not "
                                     "match opening bracket '{}' at position {}."
                                     .format(c, i, block.opener,
                                             block.start))
                block.stop = i+1

                # TODO: make this method of the Elements object or something
                el_start_i = list(block.elements.keys())[-1]
                el_stop_i = i
                block.elements[el_start_i] = s[el_start_i:el_stop_i]

                blocks[block.start] = block_stack.pop()
            elif c in self.separators:
                block = block_stack[-1]

                el_start_i = list(block.elements.keys())[-1]
                el_stop_i = i
                block.elements[el_start_i] = s[el_start_i:el_stop_i]

                block.elements[el_stop_i+1] = None

        if len(block_stack) > 0:
            raise ValueError("Unmatched opening bracket '{}' at position {}."
                             .format(block_stack[-1].opener,
                                     block_stack[-1].start))

        return blocks

            # A dictionary of the elements separated by one of the 'separators'
            # The key is the index of the first character

class Block(SimpleNamespace):
    def __init__(self, start, opener, closer):
        super().__init__()
        self.start = start
        self.stop = None
        self.opener = opener
        self.closer = closer
        self.elements = OrderedDict([(start+1, None)])
        self.blocks = []
