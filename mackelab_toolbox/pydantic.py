from pydantic import ValidationError

def generic_pydantic_initializer(cls):
    """
    Add logic to a pydantic model which recognizes an initial positional
    argument as an unresolved initializer. If provided, this initializer can be:
      - A instance of `cls`, which is simply returned.
      - A dict to parse with `parse_obj`
      - A string to parse with `parse_json`
    If the initializer is not provided, the class is initialized as usual using
    keyword arguments (or additional args in __init__).
    To do this, this decorator does the following:
      - Adds the argument '__gendesc' to the class signature. This is removed
        before passing on to other __init__ or __new__ methods, but this
        argument name should be considered reserved.
      - Adds the attribute '_do_init' to the class' __slots__ (creates slots
        if required).
    Limitations:
      - Decorator should only be called on the lowest member of a class hierarchy
      - Decorator should not be used on a class for which you do not want to
        define __slots__.
      - Decorator does not work with models using Generic types

    Examples:
    ---------
    >>> from pydantic import BaseModel
    >>> from mackelab_toolbox.pydantic import generic_pydantic_initializer
    >>> @generic_pydantic_initializer
    >>> class Foo:
    >>>     a : int
    >>>     b : int
    These instantiations are all equivalent
    >>> foo1 = Foo(a=1, b=2)
    >>> foo2 = Foo(foo1)         # Simply returns foo1
    >>> assert foo2 is foo1
    >>> foo3 = Foo(foo1.dict())  # New instance with copied attributes
    >>> foo4 = Foo(foo1.json())  # New instance with copied attributes
    >>> assert foo3 is not foo1
    >>> assert foo4 is not foo1
    >>> assert foo3 == foo1
    >>> assert foo4 == foo1
    """

    # FIXME: Still shows up as 'WrappedClass' in ref

    if hasattr(cls, '_do_init'):
        raise RuntimeError("The `generic_pydantic_initializer` may be only be "
                           "applied to the lowest member of a class hierarchy.")
    class WrappedClass(cls):
        __slots__ = ('_do_init',)
        def __init__(self, __gendesc=None, *args, **kwargs):
            if self._do_init:
                if self._do_init == 'with desc':
                    args = (__gendesc,) + args
                return super(WrappedClass, self).__init__(*args, **kwargs)
        def __new__(clsarg, __gendesc=None, *args, **kwargs):
            desc = __gendesc  # Use name in signature less likely to clobber
            obj = None
            if isinstance(desc, clsarg):
                obj = desc
            elif isinstance(desc, dict):
                try:
                    obj = clsarg.parse_obj(desc)
                except ValidationError:
                    pass
            elif isinstance(desc, str):
                try:
                    obj = clsarg.parse_raw(desc)
                except ValidationError:
                    pass
            if obj is None:
                if desc is not None:
                    # __gendesc not recognized: it could be a positional
                    # argument expected by the wrapped class
                    args = (desc,) + args
                    do_init = 'with desc'
                else:
                    do_init = True
                # Only include *args and **kwargs when calling super().__new__()
                # if the wrapped class defines the method.
                # FIXME: What a parent to 'cls' is expecting these args ?
                if '__new__' in cls.__dict__:
                    obj = cls.__new__(clsarg, *args, **kwargs)
                else:
                    obj = super().__new__(clsarg)
                object.__setattr__(obj, '_do_init', do_init)
            else:
                if len(args) + len(kwargs) > 0:
                    raise TypeError(f"{clsarg} was passed unused arguments: "
                                    f"{','.join(str(a) for a in args)}"
                                    f"{', ' if len(a) > 0 else ''}"
                                    f"{','.join(str(k)+'='+str(v) for k,v in kwargs.items())}")
                object.__setattr__(obj, '_do_init', False)
            return obj
        class Config:
            try:
                title = cls.Config.title
            except AttributeError:
                pass
    WrappedClass.__name__ = cls.__name__
    return WrappedClass
