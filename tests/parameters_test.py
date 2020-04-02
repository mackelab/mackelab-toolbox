# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:53:49 2017

@author: alex
"""

import numpy as np
from collections import OrderedDict
import hashlib
from parameters import ParameterSet
from parameters.validators import ValidationError
import mackelab_toolbox as mtb
import mackelab_toolbox.parameters

import pytest

def test_digest():
    params = ParameterSet({
        "seed": 100,
        "populations": ['α', 'β'],
        "N": [500, 100],
        "R": [1, 1],
        "u_rest": [20.123, 20.362],
        "p": [[0.1009, 0.1689],
                [0.1346, 0.1371]]
        })
    # The standardized string representation that should be created within
    # `digest` to compute the hash
    # You can retrieve this value after hashing with
    # `mtb.parameters.debug_store['digest']['hashed_string']`
    hashed_string = "OrderedDict([('N',array([500,100])),('R',array([1,1])),('p',array([[0.1009,0.1689],[0.1346,0.1371]])),('populations',['α','β']),('seed',100),('u_rest',array([20.123,20.362]))])"
    target_digest = "6d57b9e7cbae3c8692f305b0157f5cd7cb5a0ace"
    # if np.__version__ < '1.14':
    #     mtb.parameters._filename_printoptions['legacy'] = '1.13'
    #     target_name = 'be1c1866bcab0dfc86470cb90bce0d9c23adcf6e'
    # else:
    #     target_name = 'bb27d71cef9d07f15094ff5eeec5cf2d9659f68f'

    assert hashlib.sha1(bytes(hashed_string, 'utf-8')).hexdigest() == target_digest
    assert mtb.parameters.digest(params) == target_digest

    if False:
        # Can use the following to help debug
        printoptions = mtb.parameters._filename_printoptions
        np.set_printoptions(**printoptions)
        flat_params = mtb.parameters.params_to_arrays(params).flatten()
        sorted_params = OrderedDict( (key, flat_params[key]) for key in sorted(flat_params) )
        print("Sorted params")
        print(repr(sorted_params))
        print("Hashed string")
        print(mtb.parameters.debug_store['digest']['hashed_string'])
        # with open("tmp.log", 'w') as f:
        #     f.write(mtb.parameters.debug_store['digest']['hashed_string'])
        print("Digest")
        print(mtb.parameters.digest(params))
        #basename = hashlib.sha1(bytes(repr(sorted_params), 'utf-8')).hexdigest()


def parameterspec_test():
    class Foo:
        class Parameters(mtb.parameters.ParameterSpec):
            schema = {'x': 0., 'y': 1.}
    def __init__(self, *args, **kwargs):
        self.params = Foo.Parameters(*args, **kwargs)
        self.x = self.params.x
        self.y = self.params.y

    class Bar:
        class Parameters(mtb.parameters.ParameterSpec):
            schema = {'xy': Foo.Parameters,
                      'n': 1, 'name': Subclass(str)}
            def parse(self, desc, name='default_name'):
                if n in desc: self.n = desc['n']
                self.name = name
        def __init__(self, *args, **kwargs):
            self.params = Bar.Parameters(*args, **kwargs)
            self.foo = Foo(self.params.xy)
            self.n = self.params.n
            self.name = self.params.name

    # Standard construction of parameters
    bar1 = Bar(name='bar1', n=10, xy={'x': 0., 'y': 1.})
    bar2 = Bar(n=100, xy=Foo.Parameters(x=10., y=1.))
    try:
        # Fails because of type of name
        Bar(**{'name': 3, 'n': 10, 'xy': {'x': 0., 'y': 1.}})
    except ValidationError:
        pass
    else:
        assert False
    try:
        # Fails because of type of y
        Bar(**{'name': 'bar4', 'n': 10, 'xy': {'x': 0., 'y': 1}})
    except ValidationError:
        pass
    else:
        assert False
    # Re-using instantiated parameters
    bar5 = Bar(bar1.params)
    foo1 = Foo(bar5.foo.params)
    bar6 = Bar(bar1.params, bar2.params, name='bar6')
    bar7 = Bar(bar1.params, xy=bar2.foo.params, name='bar7')
    assert bar6.name == 'bar6' and bar6.n == 100 and bar6.xy.x == 10.
    assert bar7.name == 'bar7' and bar7.n == 10  and bar7.xy.x == 10.

def expansion_test():
    """TODO: Use ParameterRange instead of custom `expand_params`"""
    input_str = (
"""
'angles': [ [[*[5.655, 4, 3, 2, 1],
                  1.57],
              [1.57, 1.57]], # w
            [1.57, *{0, 3}],          # logτ_m
            [1.57]              # c
          ]
""")

    target_output = (
    ["\n'angles': [ [[5.655,\n                  1.57],\n              [1.57, 1.57]],\n            [1.57, 0],\n            [1.57]\n          ]",
     "\n'angles': [ [[5.655,\n                  1.57],\n              [1.57, 1.57]],\n            [1.57,  3],\n            [1.57]\n          ]",
     "\n'angles': [ [[ 4,\n                  1.57],\n              [1.57, 1.57]],\n            [1.57, 0],\n            [1.57]\n          ]",
     "\n'angles': [ [[ 4,\n                  1.57],\n              [1.57, 1.57]],\n            [1.57,  3],\n            [1.57]\n          ]",
     "\n'angles': [ [[ 3,\n                  1.57],\n              [1.57, 1.57]],\n            [1.57, 0],\n            [1.57]\n          ]",
     "\n'angles': [ [[ 3,\n                  1.57],\n              [1.57, 1.57]],\n            [1.57,  3],\n            [1.57]\n          ]",
     "\n'angles': [ [[ 2,\n                  1.57],\n              [1.57, 1.57]],\n            [1.57, 0],\n            [1.57]\n          ]",
     "\n'angles': [ [[ 2,\n                  1.57],\n              [1.57, 1.57]],\n            [1.57,  3],\n            [1.57]\n          ]",
     "\n'angles': [ [[ 1,\n                  1.57],\n              [1.57, 1.57]],\n            [1.57, 0],\n            [1.57]\n          ]",
     "\n'angles': [ [[ 1,\n                  1.57],\n              [1.57, 1.57]],\n            [1.57,  3],\n            [1.57]\n          ]"]
    )
    assert( mtb.parameters.expand_params(input_str) == target_output )


if __name__ == '__main__':
    test_digest()
