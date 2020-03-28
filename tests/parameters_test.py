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

def test_parameterized(caplog):

    from importlib import reload
    import mackelab_toolbox.parameters as ps
    from mackelab_toolbox.cgshim import shim
    from typing import List
    import numpy as np
    from nptyping import Array

    @ps.parameterized
    class Model:
        a: float
        # dt: float=0.01
        dt: float
        def integrate(self, x0, T, y0=0):
            x = x0; y=y0
            a = self.a; dt = self.dt
            for t in np.arange(0, T, self.dt):
                x += y*dt; y+=a*x*dt
            return x
    m = Model(a=-0.5, dt=0.01)
    assert str(m) == "Model_Parameterized(a=-0.5, dt=0.01)"
    assert np.isclose(m.integrate(1, 5.), -0.924757713688715)
    assert type(m.a) is float

    # Import after defining Model to test difference in mapping of `float` type
    import mackelab_toolbox.cgshim

    @ps.parameterized
    class Model2(Model):
        b: np.float32
        w: Array[np.float32]
        def integrate(self, x0, T, y0=0):
            x = w*x0; y = w*y0
            α = np.array([self.a, self.b]); dt = self.dt
            for t in np.arange(0, T, self.dt):
                x += y*dt; y+=α*x*dt
            return x.sum()

    # Basic type conversion
    w = np.array([0.3, 0.7], dtype='float64')
    w32 = w.astype('float32')
    w16 = w.astype('float16')
    m2_np = Model2(a=-0.5, b=-0.1, w=w, dt=0.01)
    assert str(m2_np) == "Model2_Parameterized(a=-0.5, dt=0.01, b=-0.1, w=array([0.3, 0.7], dtype=float32))"
    assert m2_np.b.dtype is np.dtype('float32')
    # Check that loading cgshim has overridden the meaning of the 'float' type
    assert type(m2_np.a) is not type(m.a)
    assert type(m2_np.a) is np.dtype(shim.config.floatX).type
    # Test conversion from ints and list
    wint = np.array([1, 2])
    m2_npint = Model2(a=-0.5, b=-0.1, w=wint, dt=0.01)
    assert m2_npint.w.dtype is np.dtype('float32')
    m2_nplist = Model2(a=-0.5, b=-0.1, w=[0.3, 0.7], dt=0.01)
    assert m2_nplist.w.dtype is np.dtype('float32')

    shim.load(True)
    m2_shared = Model2(a=-0.5, b=-0.1, w=shim.shared(w), dt=0.01)
    m2_tensor = Model2(a=-0.5, b=-0.1, w=shim.tensor(w16), dt=0.01)
    m2_shared = Model2(a=-0.5, b=-0.1, w=shim.shared(w32), dt=0.01)
    m2_tensor = Model2(a=-0.5, b=-0.1, w=shim.tensor(w32), dt=0.01)

    # Check that the calls with shared(w) and tensor(w) triggered warnings
    assert len(caplog.records) == 2
    caprec1 = caplog.records[0]
    assert caprec1.levelname == "ERROR"
    assert caprec1.msg.startswith("Attempted to cast a symbolic")
    caprec2 = caplog.records[1]
    assert caprec2.levelname == "WARNING"  # Warning because upcast less critical than downcast
    assert caprec2.msg.startswith("Attempted to cast a symbolic")


    # shim.shared(w).astype('floatX')
    #
    # m2_shared.w
    # m2_tensor.w

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
    # parameterspec_test()
