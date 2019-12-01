# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:53:49 2017

@author: alex
"""

import numpy as np
from collections import OrderedDict
from parameters import ParameterSet
from parameters.validators import ValidationError
import mackelab_toolbox as mtb
import mackelab_toolbox.parameters

def expansion_test():
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


def filename_test():
    params = ParameterSet({
        "seed": 100,
        "populations": ['α', 'β'],
        "N": [500, 100],
        "R": [1, 1],
        "u_rest": [20.123, 20.362],
        "p": [[0.1009, 0.1689],
                [0.1346, 0.1371]]
        })
    if np.__version__ < '1.14':
        mtb.parameters._filename_printoptions['legacy'] = '1.13'
        target_name = 'be1c1866bcab0dfc86470cb90bce0d9c23adcf6e'
    else:
        target_name = 'bb27d71cef9d07f15094ff5eeec5cf2d9659f68f'

    assert mtb.parameters.get_filename(params) == target_name

    if False:
        # Can use the following to help debug
        printoptions = mtb.parameters._filename_printoptions
        np.set_printoptions(**printoptions)
        flat_params = mtb.parameters.params_to_arrays(params).flatten()
        sorted_params = OrderedDict( (key, flat_params[key]) for key in sorted(flat_params) )
        print(repr(sorted_params))
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
                if n in desc: self.n = desc.['n']
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
    bar6 = Bar(name='bar6', bar1.params, bar2.params)
    bar7 = Bar(name='bar7', bar1.params, xy=bar2.foo.params)
    assert bar6.name == 'bar6' and bar6.n == 100 and bar6.xy.x == 10.
    assert bar7.name == 'bar7' and bar7.n == 10  and bar7.xy.x == 10.

if __name__ == '__main__':
    filename_test()
    parameterspec_test()
