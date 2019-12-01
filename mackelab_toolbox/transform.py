# -*- coding: utf-8 -*-
"""
Provides the TransformedVar class. Features:
    - Single object for variable name, transformation and inverse transformation
    - Consistent interface for non-transformed variables with NonTransformedVar
Usage:
    >>> import theano.tensor as tt
    >>> from mackelab_toolbox.transform import TransformedVar, NonTransformedVar
    >>> x = tt.scalar('x')
    >>> T = TransformedVar()
"""
from collections import namedtuple
import numpy as np
import scipy as sp
import math as math

import simpleeval
import ast
import operator

from parameters.validators import Subclass
from .parameters import ParameterSpec

class Transform:
    class Parameters(ParameterSpec):
        schema = {'xname': Subclass(str), 'expr':Subclass(str)}
        def parse(self, desc):
            xname, expr = desc.split('->')
            self.xname = xname.strip()
            self.expr = expr.strip()
            # TODO: Better sanity check
            return '->' not in self.xname + self.expr

    # Replace the "safe" operators with their standard forms
    # (simpleeval implements safe_add, safe_mult, safe_exp, which test their
    #  input but this does not work with non-numerical types.)
    _operators = simpleeval.DEFAULT_OPERATORS
    _operators.update(
        {ast.Add: operator.add,
         ast.Mult: operator.mul,
         ast.Pow: operator.pow})
    # Allow evaluation to find operations in standard namespaces
    namespaces = {'np': np, 'sp': sp, 'math': math}

    def __new__(cls, *args, **kwargs):
        # Make calling Transform on a Transform instance just return the instance
        if len(args) > 0 and isinstance(args[0], Transform):
            # Don't return a new instance; just return this one
            return args[0]
        else:
            return super().__new__(cls)

    def __init__(self, desc):
        # No matter what we do in __new__, __init__ is called, so we need
        # to check this again.
        if not isinstance(desc, Transform):
            self.params = Transform.Parser(desc)
            self.xname = self.params.xname
            self.expr = self.params.expr

    def __str__(self):
        return self.desc
    def __repr__(self):
        return str(type(self)) + '(' + self.desc + ')'

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
                       "expression (only numpy and scipy, as 'np' and 'sp', are "
                       "available by default).\nIf '{}' is a module or class, you can "
                       "make it available by adding it to the transform namespace: "
                       "`Transform.namespaces.update({{'{}': {}}})`.\nSuch a line would "
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

class Bijection:
    """A pair of mutually inverse transformation."""
    class Parameters(ParameterSpec):
        schema = {'map': Transform.Parameters,
                  'inverse': Transform.Parameters}
        def parse(self, desc):
            if isinstance(desc, str):
                # Check there is exactly one ';' character
                try:
                    i = desc.index(';')
                except ValueError:
                    pass  # Raise error below
                else:
                    if ';' not in desc[i+1:]:
                        self.map = desc[:i].strip()
                        self.inverse_map = desc[i+1:].strip()
                        return True
            else:
                self.map = desc.get('map')
                self.inverse_map = desc.get('inverse')
                if None not in (self.map, self.inverse_map):
                    return True
            return False

    def __init__(self, *args, _inverse=None, test_value=1, **kwargs):
        """
        Set `test_value` to None to disable the bijection test.
        """
        self.desc = Bijection.Parameters(*args, **kwargs)
        self.map = Transform(self.desc.map)
        self.inverse_map = Transform(self.desc.inverse_map)
        self._inverse = _inverse
        if not (np.isclose(self.map(self.inverse_map(test_value)), test_value)
                and np.isclose(self.inverse_map(self.map(test_value)),
                               test_value)):
            raise ValueError("The specified `map` and `inverse_map` are not "
                             "inverses of one another. Test value: {}."
                             .format(test_value))


    def __call__(self, x):
        return self.map(x)

    @property
    def inverse(self):
        if self._inverse is None:
            self._inverse = Bijection(to=self.inverse_map, back=self.map,
                                      _inverse=self)
        return self._inverse


class MetaTransformedVar(type):
    """Implements subclass selection mechanism of TransformedVarBase."""
    # Based on https://stackoverflow.com/a/14756633
    def __call__(cls, desc, *args, orig=None, new=None):
        if len(args) > 0:
            raise TypeError("TransformedVar() takes only one positional argument.")
        if not( (orig is None) != (new is None) ):  #xor
            raise ValueError("Exactly one of `orig`, `new` must be specified.")

        if desc is not None and all(s in desc for s in ['name', 'to', 'back']):
            # Standard way to make transformed variable by providing transform description
            var = TransformedVar.__new__(TransformedVar, desc, *args, orig=orig, new=new)
            var.__init__(desc, *args, orig=orig, new=new)
        elif desc is not None and 'transform' in desc:
            # Alternative way to make transformed variable: pass entire
            # variable description, which includes transform description
            var = TransformedVar.__new__(TransformedVar, desc.transform, *args, orig=orig, new=new)
            var.__init__(desc.transform, *args, orig=orig, new=new)
        else:
            # Not a transformed variable
            if orig is None:
                orig = new  # orig == new for non transformed variables
            var = NonTransformedVar.__new__(NonTransformedVar, orig)
            var.__init__(desc, orig)

        return var

class TransformedVarBase(metaclass=MetaTransformedVar):
    """
    Base class for TransformedVar and NonTransformedVar.
    Can use its constructor to obtain either TransformedVar or NonTransformedVar,
    depending on whether `desc` provides a transform.
    """
    TransformName = namedtuple('TransformName', ['orig', 'new'])

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

class TransformedVar(TransformedVarBase):

    # See MetaTransformedVar for the code parsing arguments

    def __init__(self, desc, *args, orig=None, new=None):
        """
        Should pass exactly one of the parameters `orig` and `new`
        TODO: Allow non-symbolic variables. Possible implementation:
            Test to see if orig/new is a constant, callable or symbolic.
            Set the other orig/new to the same type.
        """
        if not all(s in desc for s in ['to', 'back', 'name']):
            raise ValueError("Incomplete transform description")
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

        # Set the variable names
        self.names = self.TransformName(*[nm.strip() for nm in desc.name.split('->')])
        if hasattr(self.orig, 'name'):
            if self.orig.name is None:
                self.orig.name = self.names.orig
            else:
                assert(self.orig.name == self.names.orig)
        if hasattr(self.new, 'name'):
            if self.new.name is None:
                self.new.name = self.names.new
            else:
                assert(self.new.name == self.names.new)

    def __str__(self):
        return self.names.orig + '->' + self.names.new + ' (' + self.to.desc + ')'

    def rename(self, orig, new):
        """
        Rename the variables

        Parameters
        ----------
        new: str
            Name to assign to self.new
        orig: str
            Name to assign to self.orig
        """
        self.names = self.TransformName(orig=orig, new=new)
        self.orig.name = orig
        self.new.name = new

class NonTransformedVar(TransformedVarBase):
    """Provides an interface consistent with TransformedVar."""

    # See MetaTransformedVar for the code parsing arguments

    def __init__(self, desc, orig):
        """
        `desc` only used to provide name attribute. If you don't need name,
        you can pass `desc=None`.

        Parameters
        ----------
        desc: ParameterSet
            Transform description
        orig: variable
            Theano or Python numeric variable.
        """
        self.orig = orig
        self.to = Transform('x -> x')
        self.back = Transform('x -> x')
        self.new = orig
        # Set name
        self.names = None
        if isinstance(desc, str):
            self.names = self.TransformName(desc, desc)
        elif desc is not None and 'name' in desc:
            nametup = desc.name.split('->')
            if len(nametup) == 1:
                self.names = self.TransformName(nametup[0], nametup[0])
            elif len(nametup) == 2:
                self.names = self.TransformName(*nametup)
                assert(self.names.new == self.names.orig)
            else:
                raise ValueError("Malformed transformation name description '{}'."
                                 .format(desc.name))
        if self.names is not None:
            if hasattr(self.orig, 'name'):
                if self.orig.name is not None:
                    assert(self.orig.name == self.names.orig)
            else:
                self.orig.name = self.names.orig

    def __str__(self):
        if self.names is not None:
            return self.names.orig + '(' + str(self.orig) + ')'
        else:
            return str(self.orig)

    def rename(self, orig, new=None):
        """
        Rename the variables

        Parameters
        ----------
        orig: str
            Name to assign to self.orig
        new: str
            Ignored; only provided to have consistent API with TransformedVar.
            If given, must be equal to `orig`.
        """
        if new is not None and new != orig:
            raise ValueError("For NonTransformedVar, the 'new' and 'orig' names must match.")
        self.names = self.TransformName(orig=orig, new=orig)
        if hasattr(self.orig, 'name') and self.orig.name is not None:
            self.orig.name = orig
