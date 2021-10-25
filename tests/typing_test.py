import pytest
from pydantic import BaseModel, validator
from pydantic import ValidationError
from pydantic.dataclasses import dataclass
import dataclasses
import mackelab_toolbox as mtb
# from mackelab_toolbox.typing import NPValue, Array
import mackelab_toolbox.typing as mtbT
import mackelab_toolbox.cgshim as cgshim
from mackelab_toolbox.cgshim import shim
from typing import List, Tuple
import numpy as np

# TODO: Systematically test casting of NPValue and Array, especially with
#       generic types

def test_IndexableNamespace():

    class Foo(BaseModel):
        bar: mtbT.IndexableNamespace
        class Config:
            json_encoders={mtbT.IndexableNamespace: mtbT.IndexableNamespace.json_encoder}

    foo = Foo(bar=mtbT.IndexableNamespace(a=[], b=3))
    foo2 = Foo.parse_raw(foo.json())

    assert foo == foo2
    assert foo.bar is not foo2.bar
    assert foo.bar.a is not foo2.bar.a
    assert foo.bar.b is foo2.bar.b  # Python reuses literals

def test_pydantic(caplog):

    class Model(BaseModel):
        a: float
        # dt: float=0.01
        dt: float
        class Config:
            json_encoders = mtbT.json_encoders
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
        b: mtbT.NPValue[np.float64]
        w: mtbT.Array[np.float32]
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

    shim.load('theano')
    mtbT.freeze_types()
    # TODO: Use dataclass below once kwargs in dataclasses are fixed
    # NOTE: Some of the tests below do not work presently with with
    #       dataclasses, due to the way keyword vs non-keyword parameters
    #       are ordered (see discussion https://bugs.python.org/issue36077)
    #       Pydantic dataclass however allows setting default/computed
    #       override base class attributes, without having to use a default for
    #       new every attribute in the derived class
    class Model3(Model):
        a: cgshim.typing.FloatX  = 0.3      #  <<< Setting default does not work with dataclass (a comes before non-keyword b)
        β: mtbT.NPValue[float] = None
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
        b: mtbT.NPValue[np.float32]
        w: mtbT.Array[np.float32] = (np.float32(1), np.float32(0.2))
        γ: mtbT.Array[np.float32] = None
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
        largeb = 57389780668102097303
            # Obtained with int(''.join((str(i) for i in np.random.randint(10, size=20))))
        m2_np = Model2(a=-0.5, b=largeb, w=w16, dt=0.01)
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

    shim.load('theano')
    # If we define Model2symb before loading theano, Tensor, Shared & Symbolic
    # will not include shared types
    # Types are fixed when class is defined
    class Model2symb(Model2):
        dt: cgshim.typing.Tensor[float]
        b : cgshim.typing.Shared[shim.config.floatX]
        w : cgshim.typing.Symbolic[np.float32]

    with pytest.raises(ValidationError):
        Model2(a=-0.5, b=-0.1, w=shim.shared(w32), dt=0.01)
    with pytest.raises(ValidationError):
        # Fails b/c w:float
        Model2symb(a=-0.5, b=shim.shared(-0.1), w=w32, dt=shim.shared(0.01))
    with pytest.raises(ValidationError):
        # Fails b/c b:tensor
        Model2symb(a=-0.5, b=shim.tensor(-0.1), w=shim.tensor(w32), dt=shim.shared(0.01))

    # Does not fail b/c w: shared <= Symbolic
    Model2symb(a=-0.5, b=shim.shared(-0.1), w=shim.shared(w32), dt=shim.shared(0.01))
    # Does not fail b/c b: casted as shared
    Model2symb(a=-0.5, b=-0.11111, w=shim.tensor(w32), dt=shim.shared(0.01))

    # Clear captured logs messages from failure tests
    caplog.clear()
    m2_sta32 = Model2symb(a=-0.5, b=shim.shared(-0.1), w=shim.tensor(w32), dt=0.01)
    m2_stt64 = Model2symb(a=-0.5, b=shim.shared(-0.1), w=shim.tensor(w64), dt=shim.tensor(0.01))
    m2_sts16 = Model2symb(a=-0.5, b=shim.shared(-0.1), w=shim.tensor(w16), dt=shim.shared(0.01))
    m2_stt32 = Model2symb(a=-0.5, b=shim.shared(-0.1), w=shim.tensor(w32), dt=shim.tensor(0.01))
    m2_sts32 = Model2symb(a=-0.5, b=shim.shared(-0.1), w=shim.tensor(w32), dt=shim.shared(0.01))

    # Check that the calls with shared(w) and tensor(w) triggered precision warnings
    assert len(caplog.records) == 2
    caprec1 = caplog.records[0]   # m2_stt64
    assert caprec1.levelname == "ERROR"
    assert caprec1.msg.startswith("w expects a symbolic variable with data type <class 'numpy.float32'>. The provided symbolic value has dtype float64")
    caprec2 = caplog.records[1]   # m2_sts16
    assert caprec2.levelname == "ERROR"
    assert caprec2.msg.startswith("w expects a symbolic variable with data type <class 'numpy.float32'>. The provided symbolic value has dtype float16")

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

    # Aggregate types, dimension testing
    class Foo(BaseModel):
        a : mtb.typing.AnyNumericalType   # arrays, symbolics ok
        b : mtb.typing.AnyScalarType      # symbolics ok, but not arrays
        c : mtb.typing.NPValue[np.number]
        d : mtb.typing.DType = 'int8'

    with pytest.raises(ValidationError):
        Foo(a=1, b=2, c=np.arange(2))
    with pytest.raises(ValidationError):
        Foo(a=1, b=np.arange(2), c=2)
    Foo(a=1, b=2, c=2)
    Foo(a=np.arange(4), b=2, c=np.int_(2))  # Underscore important, because np.int === int, which is not a numpy type
    Foo(a=1, b=np.array(2), c=np.array(2))
    # DType tests
    with pytest.raises(ValidationError):
        Foo(a=1, b=2, c=2, d=8)
    Foo(a=1, b=2, c=2)
    Foo(a=1, b=2, c=2, d=np.float)
    Foo(a=1, b=2, c=2, d='float32')
    Foo(a=1, b=2, c=2, d=shim.config.floatX)

def test_pydantic_rng():
    class RandomModel(BaseModel):
        rng: mtbT.RNGenerator
        class Config:
            json_encoders = mtbT.json_encoders

    RandomModel.schema()

    from numpy.random import Generator, PCG64, MT19937, SFC64, Philox
    seed = 953235987
    rm_pcg = RandomModel(rng=Generator(PCG64(seed)))
    rm_mt  = RandomModel(rng=Generator(MT19937(seed)))
    rm_sfc = RandomModel(rng=Generator(SFC64(seed)))
    rm_phi = RandomModel(rng=Generator(Philox(seed)))

    # Save models in their current initialized state
    pcg_json = rm_pcg.json()
    mt_json  = rm_mt.json()
    sfc_json = rm_sfc.json()
    phi_json = rm_phi.json()

    # Draw from models, advancing the bit generator
    pcg_draws = rm_pcg.rng.random(size=5)
    mt_draws  = rm_mt.rng.random(size=5)
    sfc_draws = rm_sfc.rng.random(size=5)
    phi_draws = rm_phi.rng.random(size=5)

    # Create new model copies, in their original initialized states
    rm_pcg2 = RandomModel.parse_raw(pcg_json)
    rm_mt2  = RandomModel.parse_raw(mt_json)
    rm_sfc2 = RandomModel.parse_raw(sfc_json)
    rm_phi2 = RandomModel.parse_raw(phi_json)

    # Draw again => same numbers as before
    assert np.all(pcg_draws == rm_pcg2.rng.random(size=5))
    assert np.all(mt_draws  == rm_mt2.rng.random(size=5))
    assert np.all(sfc_draws == rm_sfc2.rng.random(size=5))
    assert np.all(phi_draws == rm_phi2.rng.random(size=5))

    # Drawing _again_ produces different numbers => these really are random numbers
    assert np.all(pcg_draws != rm_pcg2.rng.random(size=5))
    assert np.all(mt_draws  != rm_mt2.rng.random(size=5))
    assert np.all(sfc_draws != rm_sfc2.rng.random(size=5))
    assert np.all(phi_draws != rm_phi2.rng.random(size=5))

def test_pydantic_legacy_rng():
    class RandomModel(BaseModel):
        rng: mtbT.RandomState
        class Config:
            json_encoders = mtbT.json_encoders

    RandomModel.schema()

    from numpy.random import RandomState
    seed = 953235987
    rm_leg = RandomModel(rng=RandomState(seed))

    # Save models in their current initialized state
    leg_json = rm_leg.json()

    # Draw from models, advancing the bit generator
    leg_draws = rm_leg.rng.random(size=5)

    # Create new model copies, in their original initialized states
    rm_leg2 = RandomModel.parse_raw(leg_json)

    # Draw again => same numbers as before
    assert np.all(leg_draws == rm_leg2.rng.random(size=5))

    # Drawing _again_ produces different numbers => these really are random numbers
    assert np.all(leg_draws != rm_leg2.rng.random(size=5))

def test_pydantic_theano_rngs():
    # TODO: We currently aren't testing every code path through .validate()
    shim.load('theano')
    mtbT.freeze_types()
    class RandomModel_Numpy(BaseModel):
        rng: mtbT.RandomStateStream
        class Config:
            json_encoders = mtbT.json_encoders
    class RandomModel_MRG(BaseModel):
        rng: mtbT.RNGStream
        class Config:
            json_encoders = mtbT.json_encoders

    rm_np = RandomModel_Numpy(rng=1)
    rm_mrg = RandomModel_MRG(rng=1)

    # Save models in their current initialized state
    rm_np_json = rm_np.json()
    rm_mrg_json = rm_mrg.json()

    # Draw from models, advancing the bit generator
    u1_np = rm_np.rng.uniform(size=(1,))
    draws_np = u1_np.eval()
    u1_mrg = rm_mrg.rng.uniform(size=(1,))
    draws_mrg = u1_mrg.eval()

    # Create new model copies, in their original initialized states
    rm2_np = RandomModel_Numpy.parse_raw(rm_np_json)
    rm2_mrg = RandomModel_MRG.parse_raw(rm_mrg_json)

    # Draw again => same numbers as before
    u2_np = rm2_np.rng.uniform(size=(1,))
    u2_mrg = rm2_mrg.rng.uniform(size=(1,))
    assert np.all(draws_np == u2_np.eval())
    assert np.all(draws_mrg == u2_mrg.eval())

    # Drawing _again_ produces different numbers => these really are random numbers
    assert np.all(draws_np  != u2_np.eval())
    assert np.all(draws_mrg != u2_mrg.eval())

def _test_pydantic_shim_rng(cgshim):
    # TODO: test that random state is saved and restored
    shim.load(cgshim)
    mtbT.freeze_types()
    class RandomModel(BaseModel):
        rng: mtbT.AnyRNG
        class Config:
            json_encoders = mtbT.json_encoders

    rm_shim = RandomModel(rng=shim.config.RandomStream())

    # Save models in their current initialized state
    rm_shim_json = rm_shim.json()

    # TODO: At present the UI is different between NumPy and Theano, because
    #       in the first case, rng.uniform() draws from a uniform, while in
    #       the second case, it returns a new RNG producing uniform values.
    #       We should have ShimmedRandomStream reproduce the latter, and then
    #       the rest of the test will work. (Have to check how this will work
    #       with models though first)
    #
    # # Draw from models, advancing the bit generator
    # u1 = rm_shim.rng.uniform(size=(1,))
    # rm_shim_draws = shim.eval(u1)

    # Create new model copies, in their original initialized states
    rm_shim2 = RandomModel.parse_raw(rm_shim_json)

    # # Draw again => same numbers as before
    # u2 = rm_shim2.rng.uniform(size=(1,))
    # # u2_draws = u2.eval()
    # # assert np.all(rm_shim_draws == u2_draws)
    # assert np.all(rm_shim_draws == shim.eval(u2))
    #
    # # Drawing _again_ produces different numbers => these really are random numbers
    # assert np.all(rm_shim_draws != shim.eval(u2))

def test_pydantic_shimtheano_rng():
    return _test_pydantic_shim_rng('theano')
def test_pydantic_shimnumpy_rng():
    return _test_pydantic_shim_rng('numpy')
