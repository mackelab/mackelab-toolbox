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
