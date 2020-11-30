
from mackelab_toolbox.utils import int_if_close
def int_if_close_test():
    xint = 5
    xfloat = 5.1
    xnpfloat = np.float32(5.1)
    xeps = 5 + np.finfo(float).eps * 50
    arrint = np.array([xint, xint])
    arrmixed = np.array([xint, xfloat, xeps])

    assert(ml.utils.int_if_close(xint) is xint)

    assert(np.rint(xint).astype(np.int) == xint)

    assert(ml.utils.int_if_close(xfloat) is xfloat)

    assert(ml.utils.int_if_close(xnpfloat) is xnpfloat)

    assert(ml.utils.int_if_close(xeps) == xint)

    assert(ml.utils.int_if_close(arrint) is arrint)

    assert(ml.utils.int_if_close(arrmixed) is arrmixed)

from mackelab_toolbox.utils import Singleton
def test_singleton():
    class FooType(metaclass=Singleton):
        pass
    class BarType(metaclass=Singleton):
        pass
    a = FooType()
    b = FooType()
    c = BarType()
    assert a is b
    assert a is not c

from mackelab_toolbox.utils import sentinel
def test_sentinel():
    a = sentinel('Foo')
    b = sentinel('Foo', "<Bar>")
    c = sentinel('Bar', "<Baz>")
    assert a is b
    assert a is not c
    assert str(a) == "<Foo>"
    assert str(b) == "<Foo>"
    assert str(c) == "<Baz>"

from mackelab_toolbox.utils import _tobytes
def test_hashing():
    # WIP
    assert len(_tobytes(-127)) == 1
    assert len(_tobytes(127)) == 1
    assert len(_tobytes(-128)) == 1
    assert len(_tobytes(128)) == 2
    assert len(_tobytes(-129)) == 2
    assert len(_tobytes(129)) == 2

from mackelab_toolbox.utils import prune_dict
def test_prune_dict():
    d = {'a': {'b': 3, 'c': {'b': 2}}, 'b': 1}
    assert prune_dict(d, {'c'}) == {'a': {'b': 3}, 'b': 1}
    assert prune_dict(d, {'b'}) == {'a': {'c': {}}}
