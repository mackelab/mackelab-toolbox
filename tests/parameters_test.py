# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:53:49 2017

Copyright 2017, 2020 Alexandre René
"""

import numpy as np
from collections import OrderedDict
import hashlib
from parameters import ParameterSet
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


def test_ComputedParams():
    import numpy as np
    from dataclasses import dataclass
    from parameters import ParameterSet, ParameterRange, ParameterReference
    from mackelab_toolbox.parameters import ComputedParams

    import pytest

    # Example model: dynamical system with n x n random (Gaussian) connectivity
    @dataclass
    class Params(ComputedParams):
        n       : int   # dimension size
        σ       : float # scale of the Gaussian from which the J_ii are drawn
        J_seed  : int
        sim_seed: int

        # Computed params (need to assign a value so they aren't required as arguments)
        # Alternatively we could just not define them at all, and let
        # `compute_params` create them.
        J      : np.ndarray = None
        sim_rng: np.random.Generator = None

        non_param_fields = ('n', 'J_seed', 'sim_seed')

        def compute_params(self):
            J_rng = np.random.default_rng(self.J_seed)
            self.J = J_rng.normal(0, self.σ, size=(self.n, self.n))
            self.sim_rng = np.random.default_rng(self.sim_seed)

    # The use of ParameterReference is for illustration; there's not really
    # a need here to ensure the two RNGs are seeded differently.
    θspace = Params(
        n       =5,
        σ       =ParameterRange([0.02, 0.1, 0.5]),
        J_seed  =5,
        sim_seed=ParameterReference('J_seed')+1
    )

    θspace.describe()

    θ0 = next(iter(θspace))
    sim_rng_bytes = θ0.sim_rng.bytes(5)

    assert θspace.size == 3   # Total number of parameter sets; == to product of lengths of ParameterRange values
    for θ, σ in zip(θspace, [0.02, 0.1, 0.5]):
        # Each returned parameter set has a different σ
        assert θ.σ == σ
        # RNGs draw the same numbers
        assert sim_rng_bytes == θ.sim_rng.bytes(5)
        # J have the same size (although not the same value b/c σ is different)
        assert θ.J.shape == (5,5)
        assert np.any(θ.J != θ0.sim_rng)



    # Create a new parameter space with different parameter values
    θspace2 = θspace.copy().update(n=3, σ=0.5)
    assert θspace2.n == 3
    assert θspace2.σ == 0.5
    # The original parameter space is unmodified
    assert θspace.n == 5

    # The original parameter space can't be cast as a parameter set because
    # it defines an ensemble of parameter sets.
    with pytest.raises(ValueError):
        θspace.param_set
    # But the new one can
    pset = θspace2.param_set
    # 'n' and seeds are not included in the parameter set
    assert set(pset.keys()) == set(('σ', 'J', 'sim_rng'))

if __name__ == '__main__':
    test_digest()
