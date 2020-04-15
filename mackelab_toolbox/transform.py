# -*- coding: utf-8 -*-
"""
Provides the TransformedVar class. Features:
    - Single object for variable name, transformation and inverse transformation
    - Consistent interface for non-transformed variables with NonTransformedVar
Usage:
    >>> import mackelab_toolbox as mtb
    >>> from mackelab_toolbox.cgshim import shim
    >>> import mackelab_toolbox.transform as transform
    >>> shim.load('theano')              # Enable Theano types
    >>> mtb.typing.freeze_types()        # No more types will be added
    >>>                                  # TransformedVar is only defined after
    >>>                                  # types have been frozen
    >>> x = shim.tensor(np.array(1.), name='x')
    >>> T = transform.TransformedVar(
            bijection = "x -> x**2 ; y -> np.sqrt(y)",
            orig = x
        )
Note that starred imports are not supported, since importing them before
calling `freeze_types` will lead to TransformedVar being undefined.
Recommendation is to import `transform` as a module; alternatively the names
can be imported directly, but only after `freeze_types()` has been called.
"""
from __future__ import annotations  # Cleaner self-referencing models
from collections import namedtuple
import numpy as np
import math as math

import simpleeval
import ast
import operator

from typing import Optional, ClassVar, Union
import pydantic
import pydantic.generics
from pydantic import validator, root_validator, Field, ValidationError
import mackelab_toolbox as mtb
from mackelab_toolbox.pydantic import generic_pydantic_initializer

import mackelab_toolbox.utils as utils

# This module has dependencies on dynamic types (TransformedVar) which
# break if we allow `from mackelab_toolbox.transform import *`
# We _could_ allow Transform, and Bijection, but that would just be confusing,
# since TransformedVar is also part of the public API.
__ALL__ = []

@generic_pydantic_initializer
class Transform(pydantic.BaseModel):
    """
    Store a transformation (aka map, aka function) in text format.
    Two attributes are stored: `xname` and `expr`.
    `simpleeval`_ is used to evaluate `expr`, which provides some safety
    sanitization. (Note that we disable `simpleeval`'s safety checks for large
    values, since they prevent use with symbolic variables.)

    The standard way to initialize is to use a string of the form "x -> f(x)",
    although directly setting `xname` and `expr` is also possible.

    Parameters
    ----------
    desc: str
        Format: [x] -> [f(x)]
        Attributes `xname` and `expr` are extracted by splitting `desc` with
        '->'

    Alternative parameters
    ----------------------
    xname: str
    expr : str

    .. _simpleeval:
        https://pypi.org/project/simpleeval/
    """
    xname : str
    expr  : str
    # TODO: Use Config: json_dumps? to export desc instead of xname, expr

    # Class attributes
    # Replace the "safe" operators with their standard forms
    # (simpleeval implements safe_add, safe_mult, safe_exp, which test their
    #  input but this does not work with non-numerical types.)
    _operators = simpleeval.DEFAULT_OPERATORS
    _operators.update(
        {ast.Add: operator.add,
         ast.Mult: operator.mul,
         ast.Pow: operator.pow})
    # Allow evaluation to find operations in standard namespaces
    namespaces : ClassVar[dict] ={'np': np, 'math': math}

    class Config:
        schema_extra = {'single arg format': '[x] -> [f(x)]'}
        json_encoders = mtb.typing.json_encoders

    @classmethod  # Not a validator because desc not a pydantic attribute
    def check_desc(cls, desc):
        if not isinstance(desc, str) or desc.count('->') != 1:
            raise ValueError("Transform description must be a string with "
                             "format [x] -> [f(x)]. Provided description: "
                             f"{desc}")
        return desc

    def __init__(self, desc=None, **kwargs):
        if desc is not None:
            self.check_desc(desc)
            if any(x in kwargs for x in ('xname', 'expr')):
                raise ValueError("[Transform] Specify either `desc`, or "
                                 "`xname`+`expr`.")
            xname, expr = desc.split('->')
            kwargs['xname'] = xname.strip()
            kwargs['expr']  = expr.strip()
        super().__init__(**kwargs)

    @property
    def desc(self):
        return self.xname + " -> " + self.expr
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
            e.args = (
                (e.args[0] +
                 "\n\nThis may be due to a module function in the transform "
                 "expression (only numpy and math, as 'np' and "
                 f"'math', are available by default).\nIf '{e.name}' is a "
                 "module or class, you can make it available by adding it to "
                 "the transform namespace: `Transform.namespaces.update("
                 f"{{'{e.name}': {e.name}}})`.\n"
                 "Such a line would typically be included at the beginning of "
                 "the execution script (it does not need to be in the same "
                 "module as the one where the transform is defined, as long as "
                 "it is executed before).",)
                + e.args[1:]
                ) # ( (e.args[0]+"…",) + e.args[1:] )
            raise
        return res

# Transform.update_forward_refs()

def validator_parse_map(name, direction):
    splitidx = {'forward': 0, 'inverse': 1}[direction]
    def _parse_map(cls, m, values):
        if m is not None:
            return Transform(m)
        else:
            desc = values.get('desc', None)
            if desc is not None:
                mapdesc = desc.split(';')[splitidx]
                m = Transform(mapdesc)
            return m
    return validator(name, pre=True, allow_reuse=True)(_parse_map)

@generic_pydantic_initializer  # Allow initialization with json, instance, dict
class Bijection(pydantic.BaseModel):
    """A pair of mutually inverse transformation.

    Set `test_value` to None to disable the bijection test.
    """
    __slots__ = ('_inverse',)
    map         : Transform = Field(None, alias='to')
    inverse_map : Transform = Field(None, alias='back')
    test_value  : Union[mtb.typing.DType[np.number],mtb.typing.Array[np.number]] = 0.5

    class Config:
        schema_extra  = {'single arg format': '[x] -> [f(x)] ; [y] -> [f⁻¹(y)]'}
        json_encoders = mtb.typing.json_encoders

    # ------------
    # Validators and initializer

    @classmethod  # Not a validator because desc not a pydantic attribute
    def check_desc(cls, desc):
        if not isinstance(desc, str) or desc.count(';') != 1:
            raise ValueError("Bijection description must be a string with "
                             "format [x] -> [f(x)] ; [y] -> [f⁻¹(y)]. "
                             f"Provided description: {desc}")
        mapdesc, invdesc = desc.split(';')
        Transform.check_desc(mapdesc)
        Transform.check_desc(invdesc)
        return desc

    _parse_map    = validator_parse_map('map', 'forward')
    _parse_invmap = validator_parse_map('inverse_map', 'inverse')

    @root_validator
    def check_inverse(cls, values):
        forward_map, inverse_map, test_value = (values.get(x, None)
            for x in ('map', 'inverse_map', 'test_value'))
        if any(x is None for x in (forward_map, inverse_map, test_value)):
            pass
        elif not (np.all(np.isclose(forward_map(inverse_map(test_value)),
                                    test_value))
                  and np.all(np.isclose(inverse_map(forward_map(test_value)),
                                        test_value))):
            raise ValueError("The specified `map` and `inverse_map` are not "
                             "inverses of one another. "
                             f"Test value: {test_value}.")
        return values

    def __init__(self, desc=None, _inverse: Optional[Bijection]=None, **kwargs):
        object.__setattr__(self, '_inverse', _inverse)  # https://github.com/samuelcolvin/pydantic/issues/655
        if desc is not None:
            self.check_desc(desc)
            if any(x in kwargs for x in ('to', 'back')):
                raise ValueError(
                    "[Bijection] Specify either `desc`, or `to`+`back`.")
            to, back = (s.strip() for s in desc.split(';'))
            kwargs['to']   = to
            kwargs['back'] = back
        super().__init__(**kwargs)

    # ------------
    # Interface

    def __call__(self, x):
        return self.map(x)

    @property
    def to(self):
        return self.map
    @property
    def back(self):
        return self.inverse_map

    @property
    def inverse(self):
        if self._inverse is None:
            object.__setattr__(self, '_inverse', Bijection(
                to=self.inverse_map, back=self.map, _inverse=self))
        return self._inverse

    @property
    def desc(self):
        return f"{self.map.desc} ; {self.inverse_map.desc}"

# Bijection.update_forward_refs()


@generic_pydantic_initializer  # Allow initialization with dict, json, instance
class TransformNames(pydantic.BaseModel):
    """
    Store names for original and transformed variables. Essentially just
    an input normalizer.

    Examples
    --------
    These constructions are all equivalent.
    >>> tnames1 = TransformNames(orig='x', new='y')
    >>> tnames1
    TransformNames(orig='x', new='y')
    >>> tnames2 = TransformNames('x -> y')
    >>> tnames3 = TransformNames(tnames1.dict())
    >>> tnames4 = TransformNames(tnames1.json())
    >>> tnames5 = TransformNames(tnames1)   # Simply returns tnames1
    >>> tnames5 is tnames1
    """
    orig : str
    new  : str

    def __init__(self, desc=None, **kwargs):
        if isinstance(desc, str):
            if desc.count('->') != 1:
                raise ValueError(f"Wrong initializer: TransformNames({names}).")
            o, n = desc.split('->')
            for k,v in zip(('orig', 'new'), (o,n)):
                if k in kwargs:
                    raise ValueError(f"`{k}` provided by both string "
                                     "description and keyword argument.")
                kwargs[k] = v.strip()
        super().__init__(**kwargs)



Transform.update_forward_refs()
Bijection.update_forward_refs()

TransformedVar = mtb.typing.PostponedClass(
    'TransformedVar',
    'mackelab_toolbox.transform_postponed',
    'mackelab_toolbox.transform')
NonTransformedVar = mtb.typing.PostponedClass(
    'NonTransformedVar',
    'mackelab_toolbox.transform_postponed',
    'mackelab_toolbox.transform')
