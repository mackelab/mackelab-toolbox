# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:53:49 2017

@author: alex
"""

import numpy as np
from collections import OrderedDict
from parameters import ParameterSet
import mackelab as ml
import mackelab.parameters

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
    assert( ml.parameters.expand_params(input_str) == target_output )


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
        ml.parameters._filename_printoptions['legacy'] = '1.13'
        target_name = 'be1c1866bcab0dfc86470cb90bce0d9c23adcf6e'
    else:
        target_name = 'bb27d71cef9d07f15094ff5eeec5cf2d9659f68f'

    assert(ml.parameters.get_filename(params) == target_name)

    if False:
        # Can use the following to help debug
        printoptions = ml.parameters._filename_printoptions
        np.set_printoptions(**printoptions)
        flat_params = ml.parameters.params_to_arrays(params).flatten()
        sorted_params = OrderedDict( (key, flat_params[key]) for key in sorted(flat_params) )
        print(repr(sorted_params))
        #basename = hashlib.sha1(bytes(repr(sorted_params), 'utf-8')).hexdigest()

if __name__ == '__main__':
    filename_test()
