import pytest
from pydantic import BaseModel, validator
from pydantic import ValidationError
from pydantic.dataclasses import dataclass
import dataclasses
import mackelab_toolbox as mtb
from mackelab_toolbox.typing import DType, Array
import mackelab_toolbox.cgshim as cgshim
from mackelab_toolbox.cgshim import shim
from mackelab_toolbox.dataclasses import retrieve_attributes
from typing import List, Tuple
import numpy as np

def test_pydantic(caplog):

    class Model(BaseModel):
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
            return ParameterSet(self.dict())

    m = Model(a=-0.5, dt=0.01)
    assert repr(m) == "Model(a=-0.5, dt=0.01)"
    assert np.isclose(m.integrate(1, 5.), -0.924757713688715)
    assert type(m.a) is float

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
    class ModelB(BaseModel):
        l1: list
        l2: List[int]
        t1: tuple
        # t2: Tuple[np.float16, np.float32]
        pass

    mb = ModelB(l1=['foo', 33], l2=[3.5, 3], t1=(1, 2, 3, 4, 5))
    assert mb.l1 == ['foo', 33]
    assert mb.l2 == [3, 3]
    assert mb.t1 == (1, 2, 3, 4, 5)

    # TODO: Use dataclass below once kwargs in dataclasses are fixed
    # NOTE: Some of the tests below do not work presently with with
    #       dataclasses, due to the way keyword vs non-keyword parameters
    #       are ordered (see discussion https://bugs.python.org/issue36077)
    #       Pydantic dataclass however allows setting default/computed
    #       override base class attributes, without having to use a default for
    #       new every attribute in the derived class
    class Model3(Model):
        a: cgshim.types.FloatX  = 0.3      #  <<< Setting default does not work with dataclass (a comes before non-keyword b)
        β: DType[float] = None
        @validator('β', pre=True, always=True)
        def set_β(cls, v, values):
            a, dt = (values.get(x, None) for x in ('a', 'dt'))
            # This test is important, because if `a` or `dt` are missing
            # (as when we assign to `m4_fail` below) or trigger a validation
            # error, they will be missing from `values` and raise KeyError here,
            # (See https://pydantic-docs.helpmanual.io/usage/validators)
            if all((a, dt)):
                return a*dt
        def integrate(self, x0, T, y0=0):
            x = x0; y=y0
            β = self.β; dt = self.dt
            for t in np.arange(0, T, self.dt):
                x += y*dt; y+=β*x
            return x

    # NOTE: Following previous note: with vanilla dataclasses Model4 would need
    #       to define defaults for every attribute.
    class Model4(Model3):
        b: DType[np.float32]
        w: Array[np.float32] = (np.float32(1), np.float32(0.2))
        γ: Array[np.float32] = None
        @validator('γ', pre=True, always=True)
        def set_γ(cls, v, values):
            a, b, β = (values.get(x, None) for x in ('a', 'b', 'β'))
            if all((a, b, β)):
                return β * np.array([1, b/a], dtype='float32')
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
    assert repr(m2_np) == "Model2(a=-0.5, dt=0.01, b=-0.1, w=array([0.25, 0.75], dtype=float32))"
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
    # These two constructors seem to specify the same model (default w is
    # simply reproduced), but because validators are not called on default
    # arguments in the first case the default w remains a tuple.
    # You can use the following validator to force its execution
    # @validator('w', always=True)
    #     def set_w(cls, v):
    #         return v
    m4_default = Model4(a=.25, dt=0.02, b=np.float32(0.7))
    m4         = Model4(a=.25, dt=0.02, b=np.float32(0.7),
                        w=(np.float32(1), np.float32(0.2)))
    assert isinstance(m4_default.w, tuple)
    assert isinstance(m4.w, np.ndarray) and m4.w.dtype.type is np.float32

    m4_16 = Model4(a=.25, dt=0.02, b=np.float32(0.3), w=w16)
    m4_32 = Model4(a=.25, dt=0.02, b=np.float32(0.3), w=w32)
    assert m4_16.w.dtype == 'float32'
    assert m4_32.w.dtype == 'float32'
    # Casts which raise TypeError
    with pytest.raises(ValidationError):   # With stdlib dataclass: TypeError
        m4_fail = Model4(a=.3, b=0.02, w=w64)
    with pytest.raises(ValidationError):   # With stdlib dataclass: TypeError
        wint = np.array([1, 2])
        m2_npint = Model2(a=-0.5, b=-0.1, w=wint, dt=0.01)

    # Import after defining Model to test difference in mapping of `float` type

    shim.load(True)
    # If we define Model2symb before loading theano, Tensor, Shared & Symbolic
    # will not include shared types
    # Types are fixed when class is defined
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

    params = ParameterSet(m4.dict())
    assert len(params) == 6
    assert (params.a is m4.a
            and params.b is m4.b
            and params.dt is m4.dt
            and params.β is m4.β
            and np.all(params.w is m4.w)
            and np.all(params.γ is m4.γ))
    assert set(m2_stt32.params.keys()) == set(['a', 'dt', 'b', 'w'])
    assert shim.is_pure_symbolic(m2_stt32.params.w)
