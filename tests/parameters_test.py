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

from pydantic import BaseModel
from pydantic import ValidationError
from pydantic.dataclasses import dataclass
import dataclasses
from mackelab_toolbox.typing import Computed, DType, Array
import mackelab_toolbox.cgshim as cgshim
from mackelab_toolbox.cgshim import shim
from typing import List, Tuple
import numpy as np

def test_parameterized(caplog):

    @dataclass
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
        @property
        def params(self):
            return ParameterSet(dataclasses.asdict(self))

    m = Model(a=-0.5, dt=0.01)
    assert str(m) == "test_parameterized.<locals>.Model(a=-0.5, dt=0.01)"
    assert np.isclose(m.integrate(1, 5.), -0.924757713688715)
    assert type(m.a) is float

    @dataclass
    class Model2(Model):
        b: DType[np.float64]
        w: Array[np.float32]
        def integrate(self, x0, T, y0=0):
            x = w*x0; y = w*y0
            α = np.array([self.a, self.b]); dt = self.dt
            for t in np.arange(0, T, self.dt):
                x += y*dt; y+=α*x*dt
            return x.sum()

    # Test subclassed List and Tuple
    # TODO: tuples with floats
    @dataclass
    class ModelB:
        l1: list
        l2: List[int]
        t1: tuple
        # t2: Tuple[np.float16, np.float32]
        pass

    mb = ModelB(['foo', 33], [3.5, 3], (1, 2, 3, 4, 5))
    assert mb.l1 == ['foo', 33]
    assert mb.l2 == [3, 3]
    assert mb.t1 == (1, 2, 3, 4, 5)

    # TODO: Use dataclass below once kwargs in dataclasses are fixed
    # NOTE: Some of the tests below do not work presently with with stdlib's
    #       dataclasses, due to the way keyword vs non-keyword parameters
    #       are ordered (see discussion https://bugs.python.org/issue36077)
    #       Pydantic dataclass however allows setting default/Computed
    #       override base class attributes, without having to use a default for
    #       new every attribute in the derived class
    @dataclass
    class Model3(Model):
        a: cgshim.types.FloatX # = 0.3      #  <<< Setting default does not work with dataclass (a comes before non-keyword b)
        β: DType[float] = Computed
        def __post_init__(self):
            # pydantic bug: calling super().__init_post__ not robust:
            # validation routine is run after first __post_init__ completes
            self._compute_params()
        def _compute_params(self):
            self.β = self.a*self.dt
        def integrate(self, x0, T, y0=0):
            x = x0; y=y0
            β = self.β; dt = self.dt
            for t in np.arange(0, T, self.dt):
                x += y*dt; y+=β*x
            return x

    # NOTE: Following previous note: with vanilla dataclasses Model4 would need
    #       to define defaults for every attribute.
    @dataclass
    class Model4(Model3):
        b: DType[np.float32] = np.float32(0.3)
        w: Array[np.float32] = (np.float32(1), np.float32(0.2))
        γ: Array[np.float32] = Computed
        def __post_init__(self):
            # pydantic bug: calling super().__init_post__ not robust:
            # validation routine is run after first __post_init__ completes
            super()._compute_params()
            self.γ = self.β * np.array([1, self.b/self.a], dtype='float32')
        def integrate(self, x0, T, y0=0):
            x = w*x0; y = w*y0
            γ = self.γ; dt = self.dt
            for t in np.arange(0, T, self.dt):
                x += y*dt; y+=γ*x
            return x.sum()

    # Basic type conversion
    w64 = np.array([0.25, 0.75], dtype='float64')
    w32 = w64.astype('float32')
    w16 = w64.astype('float16')
    with pytest.raises(ValidationError):   # With vanially dataclass: TypeError
        m2_np = Model2(a=-0.5, b=np.float128(-0.1), w=w16, dt=0.01)
    m2_np = Model2(a=-0.5, b=-0.1, w=w16, dt=0.01)
    assert str(m2_np) == "test_parameterized.<locals>.Model2(a=-0.5, dt=0.01, b=-0.1, w=array([0.25, 0.75], dtype=float32))"
    assert m2_np.b.dtype is np.dtype('float64')
    # Test conversion from ints and list
    m2_nplist = Model2(a=-0.5, b=np.float32(-0.1), w=[np.float32(0.25), np.float32(0.75)], dt=0.01)
    assert m2_nplist.w.dtype is np.dtype('float32')

    # Test computed fields and numpy type conversions
    m3 = Model3(a=.25, dt=0.02)
    assert type(m3.a) != type(m.a)
    assert type(m3.a) is np.dtype(shim.config.floatX).type
    assert type(m3.β) is np.dtype(shim.config.floatX).type
    assert m3.β == 0.005
    m4 = Model4(.25, 0.02, b=np.float32(0.7))
    assert isinstance(m4.w, np.ndarray) and m4.w.dtype.type is np.float32

    m4_16 = Model4(.25, 0.02, w=w16)
    m4_32 = Model4(.25, 0.02, w=w32)
    assert m4_16.w.dtype == 'float32'
    assert m4_32.w.dtype == 'float32'
    # Casts which raise TypeError
    with pytest.raises(ValidationError):   # With vanially dataclass: TypeError
        Model4(.3, 0.02, w=w64)
    with pytest.raises(ValidationError):   # With vanially dataclass: TypeError
        wint = np.array([1, 2])
        m2_npint = Model2(a=-0.5, b=-0.1, w=wint, dt=0.01)

    # Import after defining Model to test difference in mapping of `float` type

    shim.load(True)
    # If we define Model2symb before loading theano, Tensor, Shared & Symbolic
    # will not include shared types
    # Types are fixed when class is defined
    @dataclass
    class Model2symb(Model2):
        dt: cgshim.types.Tensor[float]
        b : cgshim.types.Shared[cgshim.types.FloatX]
        w : cgshim.types.Symbolic[np.float32]

    with pytest.raises(ValidationError):
        Model2(a=-0.5, b=-0.1, w=shim.shared(w32), dt=0.01)
    with pytest.raises(ValidationError):
        # Fails b/c w:shared
        Model2symb(a=-0.5, b=shim.shared(-0.1), w=shim.shared(w32), dt=shim.shared(0.01))
    with pytest.raises(ValidationError):
        # Fails b/c w:float
        Model2symb(a=-0.5, b=shim.shared(-0.1), w=w32, dt=shim.shared(0.01))
    with pytest.raises(ValidationError):
        # Fails b/c b:tensor
        Model2symb(a=-0.5, b=shim.tensor(-0.1), w=shim.tensor(w32), dt=shim.shared(0.01))
    with pytest.raises(ValidationError):
        # Fails b/c b:float
        Model2symb(a=-0.5, b=-0.1, w=shim.tensor(w32), dt=shim.shared(0.01))

    # Clear captured logs messages from failure tests
    caplog.clear()
    m2_sta32 = Model2symb(a=-0.5, b=shim.shared(-0.1), w=shim.tensor(w32), dt=0.01)
    m2_stt64 = Model2symb(a=-0.5, b=shim.shared(-0.1), w=shim.tensor(w64), dt=shim.tensor(0.01))
    m2_sts16 = Model2symb(a=-0.5, b=shim.shared(-0.1), w=shim.tensor(w16), dt=shim.shared(0.01))
    m2_stt32 = Model2symb(a=-0.5, b=shim.shared(-0.1), w=shim.tensor(w32), dt=shim.tensor(0.01))
    m2_sts32 = Model2symb(a=-0.5, b=shim.shared(-0.1), w=shim.tensor(w32), dt=shim.shared(0.01))

    # Check that the calls with shared(w) and tensor(w) triggered precision warnings
    assert len(caplog.records) == 2
    caprec1 = caplog.records[0]
    assert caprec1.levelname == "ERROR"
    assert caprec1.msg.startswith("w expects a variable of type <class 'theano.gof.utils.Symbolic'>.")
    caprec2 = caplog.records[1]
    assert caprec2.levelname == "WARNING"  # Warning because upcast less critical than downcast
    assert caprec2.msg.startswith("w expects a variable of type <class 'theano.gof.utils.Symbolic'>.")

    # Check method for recovering a ParameterSet
    from mackelab_toolbox.parameters import ParameterSet
    assert isinstance(m.params, ParameterSet)
    assert m.params == dict(a=-0.5, dt=0.01)

    params = m4.params
    assert len(params) == 6
    # Ideally `params` would be a view, and we would test with `is` instead `==`
    assert (params.a == m4.a
            and params.b == m4.b
            and params.dt == m4.dt
            and params.β == m4.β
            and np.all(params.w == m4.w)
            and np.all(params.γ == m4.γ))
    assert set(m2_stt32.params.keys()) == set(['a', 'dt', 'b', 'w'])
    assert shim.is_pure_symbolic(m2_stt32.params.w)

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
