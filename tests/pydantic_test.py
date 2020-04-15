from pydantic import BaseModel
from mackelab_toolbox.pydantic import generic_pydantic_initializer

def test_generic_pydantic_initializer():
    @generic_pydantic_initializer
    class Foo(BaseModel):
        a : int
        b : int

    # These instantiations are all equivalent
    foo1 = Foo(a=1, b=2)
    foo2 = Foo(foo1)         # Simply returns foo1
    assert foo2 is foo1
    foo3 = Foo(foo1.dict())  # New instance with copied attributes
    foo4 = Foo(foo1.json())  # New instance with copied attributes
    assert foo3 is not foo1
    assert foo4 is not foo1
    assert foo3 == foo1
    assert foo4 == foo1
