from pydantic import BaseModel, validator
from typing import Callable, Any

import pytest

def test_function_serialization():

    import mackelab_toolbox as mtb
    from mackelab_toolbox.cgshim import shim
    shim.load('numpy')
    import mackelab_toolbox.typing
    mackelab_toolbox.typing.freeze_types()

    import mackelab_toolbox.serialize
    mtb.serialize.config.trust_all_inputs = True

    class Foo(BaseModel):
        a: int
        # f: Callable[[float, int], float]
        f: Callable[[float, int], float]

        class Config:
            json_encoders = mtb.serialize.json_encoders

        @validator('f', pre=True)
        def set_f(cls, value):
            if isinstance(value, str):
                value = mtb.serialize.deserialize_function(
                    value, {}, {'do_nothing': do_nothing})
            return value

    def mypow(a, n):
        return a**n

    # We can have custom decorators, but they need to be passed to the
    # deserializer as the `locals` argument (see @validator above)
    def do_nothing(f):
        return f
    @do_nothing
    def mypow2(a, n):
        return a**n

    foo1 = Foo(a=1, f=mypow)
    foo2 = Foo(a=1, f=mypow2)
    fooλ = Foo(a=1, f=lambda a,n: a**n)

    foo1.json()
    boo1 = Foo.parse_raw(foo1.json())
    boo2 = Foo.parse_raw(foo2.json())
    with pytest.raises(ValueError):
        fooλ.json()

def test_numpy_serialization():
    # TODO: Validators should allow DOWNcasting instead of upcasting
    #       Logic: If a model requires 32-bit and downcasts a 64-bit, it will
    #              behave as expected. But if it upcasts a 16-bit, then it
    #              encounter unexpected numerical errors.
    #       TODO: Change typing_module accordingly, and then all validation
    #             tests should use `validate(x)`.

    import numpy as np
    import mackelab_toolbox as mtb
    import mackelab_toolbox.typing
    mtb.typing.freeze_types()
    from mackelab_toolbox.typing import NPValue, Array

    ## Test validators ##
    x = 3.1415
    i16  = np.int16(x)
    i32  = np.int32(x)
    i64  = np.int64(x)
    f16  = np.float16(x)
    f32  = np.float32(x)
    f64  = np.float64(x)
    c64  = np.complex64(x)
    c128 = np.complex128(x)

    assert NPValue['int16'].validate(i16).dtype is i16.dtype
    assert NPValue['int32'].validate(i32).dtype is i32.dtype
    assert NPValue['int64'].validate(i64).dtype is i64.dtype
    assert NPValue['float16'].validate(f16).dtype is f16.dtype
    assert NPValue['float32'].validate(f32).dtype is f32.dtype
    assert NPValue['float64'].validate(f64).dtype is f64.dtype
    assert NPValue['complex64'].validate(c64).dtype is c64.dtype
    assert NPValue['complex128'].validate(c128).dtype is c128.dtype


    arrx = [3, 0.14, 0.0015]
    arri16  = np.array(arrx, dtype=np.int16)
    arri32  = np.array(arrx, dtype=np.int32)
    arri64  = np.array(arrx, dtype=np.int64)
    arrf16  = np.array(arrx, dtype=np.float16)
    arrf32  = np.array(arrx, dtype=np.float32)
    arrf64  = np.array(arrx, dtype=np.float64)
    arrc64  = np.array(arrx, dtype=np.complex64)
    arrc128 = np.array(arrx, dtype=np.complex128)

    assert Array[np.int16].validate(arri16).dtype is i16.dtype
    assert Array[np.int32].validate(arri32).dtype is i32.dtype
    assert Array[np.int64].validate(arri64).dtype is i64.dtype
    assert Array[np.float16].validate(arrf16).dtype is f16.dtype
    assert Array[np.float32].validate(arrf32).dtype is f32.dtype
    assert Array[np.float64].validate(arrf64).dtype is f64.dtype
    assert Array[np.complex64].validate(arrc64).dtype is c64.dtype
    assert Array[np.complex128].validate(arrc128).dtype is c128.dtype

    ## Test serialization with Pydantic ##

    class Foo(BaseModel):
        i64: NPValue['int64']
        arrf64: Array['float64']

        class Config:
            json_encoders = mtb.typing.json_encoders

    # Foo(i64=x, arrf64=arrx)
    small = Foo(i64=int(x), arrf64=arrx)
    small.json()

    Foo.parse_raw(small.json())


    # Size of array chosen to be just above threshold to trigger base85 encoding
    largeA = np.tile(arrx, 34)
    len(str(largeA.tolist()))
    large = Foo(i64=int(x), arrf64=largeA)
    len(large.json())
    large.json()

    Foo.parse_raw(large.json())

    # Very large array, with clear benefit to compression
    verylargeA = np.random.random(size=(1000, 10)) * 10
    len(str(verylargeA.tolist()))
    verylarge = Foo(i64=int(x), arrf64=verylargeA)
    len(verylarge.json())

    Foo.parse_raw(verylarge.json())

    verylarge.json()

    # Compare b85 encoding with and without compression
    # (Compression is only 8% because the data are random)
    T = Array[np.float64]
    assert T.json_encoder is Array.json_encoder
    blosc_compressed = T.json_encoder(verylargeA)
    not_compressed = T.json_encoder(verylargeA, compression='none')
    len(blosc_compressed[1]['data'])
    len(not_compressed[1]['data'])
    # Both deserialize identically
    np.all(T.validate(blosc_compressed) == T.validate(not_compressed))
