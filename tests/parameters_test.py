# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:53:49 2017

@author: alex
"""

import mackelab.parameters as parameters

def expansion_test():
    input_str = (
    """
    'angles': [ [[*[5.655, 4, 3, 2, 1],
                      1.57],
                  [1.57, 1.57]], # w
                [1.57, *(0, 3)],          # logτ_m
                [1.57]              # c
              ]
    """)
    
    target_output = (
    ["'angles': [ [[5.655,\n                  1.57],\n                 [1.57, 1.57]], # w\n                [1.57, 0],          # logτ_m\n                [1.57]              # c\n                ]\n",
     "'angles': [ [[5.655,\n                  1.57],\n                 [1.57, 1.57]], # w\n                [1.57,  3],          # logτ_m\n                [1.57]              # c\n                ]\n",
     "'angles': [ [[ 4,\n                  1.57],\n                 [1.57, 1.57]], # w\n                [1.57, 0],          # logτ_m\n                [1.57]              # c\n                ]\n",
     "'angles': [ [[ 4,\n                  1.57],\n                 [1.57, 1.57]], # w\n                [1.57,  3],          # logτ_m\n                [1.57]              # c\n                ]\n",
     "'angles': [ [[ 3,\n                  1.57],\n                 [1.57, 1.57]], # w\n                [1.57, 0],          # logτ_m\n                [1.57]              # c\n                ]\n",
     "'angles': [ [[ 3,\n                  1.57],\n                 [1.57, 1.57]], # w\n                [1.57,  3],          # logτ_m\n                [1.57]              # c\n                ]\n",
     "'angles': [ [[ 2,\n                  1.57],\n                 [1.57, 1.57]], # w\n                [1.57, 0],          # logτ_m\n                [1.57]              # c\n                ]\n",
     "'angles': [ [[ 2,\n                  1.57],\n                 [1.57, 1.57]], # w\n                [1.57,  3],          # logτ_m\n                [1.57]              # c\n                ]\n",
     "'angles': [ [[ 1,\n                  1.57],\n                 [1.57, 1.57]], # w\n                [1.57, 0],          # logτ_m\n                [1.57]              # c\n                ]\n",
     "'angles': [ [[ 1,\n                  1.57],\n                 [1.57, 1.57]], # w\n                [1.57,  3],          # logτ_m\n                [1.57]              # c\n                ]\n"]
    )

    assert( parameters.expand_params(input_str) == target_output )