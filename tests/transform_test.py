import pytest
import numpy as np
from pydantic import ValidationError
import mackelab_toolbox as mtb
import mackelab_toolbox.typing
from mackelab_toolbox.cgshim import shim

def test_transform():
    shim.load('theano')
    mtb.typing.freeze_types()
    # Names can only be imported directly after types have been frozen
    from mackelab_toolbox.transform import Transform, Bijection

    # 'np' already in Transform namespaces
    φ = Transform(" x -> np.log10(x) ")
    assert φ.xname == "x"
    assert φ.expr  == "np.log10(x)"
    assert φ.desc  == "x -> np.log10(x)"
    # assert repr(φ) == "Transform(x -> np.log10(x)"
    assert φ(10)   == 1

    # TODO: Compare to expected values
    Transform.schema()
    φ.json()
    φ.dict()

    # 'math' also in namespaces
    φ2 = Transform("x -> math.log10(x)")
    assert φ(33) == φ2(33)

    ## Could split below into a test with 'theano' ##

    # `shim` was added by cgshim
    assert 'shim' in Transform.namespaces
    ψ = Transform("y -> shim.exp(y)")
    assert ψ(8.) == np.exp(8.)

    xval = np.array([1, 2])
    x = shim.tensor(xval, dtype=shim.config.floatX)
    assert shim.is_pure_symbolic(x)
    assert np.all(ψ(x).eval({x:xval}) == ψ(xval))

    # -------
    # Bijection
    Φ = Bijection("x -> x**2 ; y -> np.sqrt(y)", test_value=xval)
    assert Φ.desc == "x -> x**2 ; y -> np.sqrt(y)"

    with pytest.raises(ValidationError):
        # Incorrect inverse
        Bijection("x -> x**2 ; y -> y/2")

    assert Φ.to(8) == Φ(8) == 64
    assert np.all(Φ.to(x).eval({x:xval}) == Φ.to(xval))
    assert np.all(Φ.to(xval) == xval**2)
    assert np.all(Φ.back(x).eval({x:xval}) == Φ.back(xval))
    assert np.all(Φ.back(xval) == np.sqrt(xval))

    assert Φ.inverse.inverse is Φ
    assert Φ.inverse.back(8) == Φ.to(8)
    assert Φ.inverse.to(8)   == Φ.back(8)

    # TODO: Compare to expected values
    Bijection.schema()
    Φ.json()
    Φ.dict()

    # No-op on instances
    assert φ is Transform(φ)
    assert Φ is Bijection(Φ)

def test_transformed_var_numpy():

    mtb.typing.freeze_types()
    from mackelab_toolbox.transform import Bijection, TransformedVar, NonTransformedVar

    xval = np.array([1, 2])
    Φ = Bijection("x -> x**2 ; y -> np.sqrt(y)")

    xsq2   = TransformedVar(bijection=Φ, orig = xval)
    xsqrt2 = TransformedVar(bijection=Φ.inverse, new=xval)  # Inverse of xsq2

    assert np.all(xsq2.new == xval**2)
    assert np.all(xsq2.new  == xsqrt2.orig)
    assert np.all(xsq2.orig == xsqrt2.new)

    # Recogonizing a non-transformed var
    # […] see test_transformed_var_theano
    x_nt2  = TransformedVar(xval, names="y -> y")   # OK: xval doesn't have a name

    assert not isinstance(x_nt2, TransformedVar)
    assert isinstance(x_nt2, NonTransformedVar)

    x_nt2.orig is x_nt2.new

def test_transformed_var_theano():

    shim.load('theano')
    mtb.typing.freeze_types()
    from mackelab_toolbox.transform import Bijection, TransformedVar, NonTransformedVar

    # Type order is important
    assert str(mtb.typing.AnyNumericalType) == "typing.Union[abc.Shared, abc.Symbolic, mackelab_toolbox.typing.typing_module.Array[number], mackelab_toolbox.typing.typing_module.NPValue[number], mackelab_toolbox.typing.typing_module.Number, float, int]"
    assert TransformedVar.__fields__['orig'].type_.__args__ == mtb.typing.AnyNumericalType.__args__
        # Among postponed modules, TransformedVar needs to be imported last so
        # that AnyNumericalType has its final value.
        # This is super fragile and evidence that my current solution for
        # dynamic types is fundamentally flawed.

    xval = np.array([1, 2])
    x = shim.tensor(xval, dtype=shim.config.floatX)
    Φ = Bijection("x -> x**2 ; y -> np.sqrt(y)")
    xsq2   = TransformedVar(bijection=Φ, orig = xval)
    xsqrt2 = TransformedVar(bijection=Φ.inverse, new=xval)  # Inverse of xsq2
    with pytest.raises(ValidationError):
        x.name = 'wrong_name'
        xsq1 = TransformedVar(bijection=Φ, orig=x)
    x.name = 'x'
    xsq1   = TransformedVar(bijection=Φ, orig=x)
    xsqrt1 = TransformedVar(
        bijection = "y -> np.sqrt(y) ; x -> x**2",
        new       = x)

    # Reason why we don't have shared vars even though AnyNumericalType is a
    # Union where 'Shared' comes first: 'Shared' returns TypeError when
    # validating non-shared values.
    assert not shim.isshared(xsq2.new)
    assert not shim.isshared(xsq2.orig)
    
    assert xsq1.orig.name == 'x'
    assert xsq1.new.name  == 'y'
    assert xsqrt1.orig.name == 'y'
    assert xsqrt1.new.name  == 'x'
    assert xsq1.names.orig == xsq1.orig.name
    assert xsq1.names.new  == xsq1.new.name

    assert np.all(xsq2.new  == xval**2)
    assert np.all(xsq2.new  == xsqrt2.orig)
    assert np.all(xsq2.orig == xsqrt2.new)
    assert np.all(xsq1.orig.eval({x: xval}) == xsqrt1.new.eval({x:xval}))

    xsq1.rename(orig='a', new='b')
    assert xsq1.orig.name == 'a'
    assert xsq1.new.name  == 'b'
    assert x.name == 'a'


    # Recogonizing a non-transformed var
    with pytest.raises(ValueError):
        x_nt1  = TransformedVar(x)                  # Missing names
    with pytest.raises(ValidationError):
        x_nt1  = TransformedVar(x, names="a -> y")  # Names differ
    with pytest.raises(ValidationError):
        x_nt1  = TransformedVar(x, names="x -> x")  # Name differ from variable
    x.name = 'x'
    x_nt1  = TransformedVar(x, names="x -> x")
    x_nt2  = TransformedVar(xval, names="y -> y")   # OK: xval doesn't have a name

    assert not isinstance(x_nt1, TransformedVar)
    assert not isinstance(x_nt2, TransformedVar)
    assert isinstance(x_nt1, NonTransformedVar)
    assert isinstance(x_nt2, NonTransformedVar)

    x_nt1.orig is x_nt1.new
    x_nt2.orig is x_nt2.new

    assert x_nt1.orig.name == x_nt1.new.name  == 'x'
    x_nt1.rename(orig='foo')
    x_nt1.orig.name == x_nt1.new.name == x.name == 'foo'

    assert x_nt1.to(88) == x_nt1.back(88) == 88

    # No-op on instances
    assert xsq1 is TransformedVar(xsq1)
    assert x_nt1 is NonTransformedVar(x_nt1)
    assert x_nt1 is TransformedVar(x_nt1)
