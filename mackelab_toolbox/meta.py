###################
# Introspection / Class-hacking / Metaprogramming

# To avoid import cycles (especially with hide_cwd_from_import), all imports
# are done within the respective functions

def fully_qualified_name(o):
    """
    Return fully qualified name for a class or function (i.e. including
    the module)

    Parameters
    ----------
    o: Type (class) or function

    Returns
    -------
    str

    Example
    -------
    >>> from mackelab_toolbox.utils import fully_qualified_name
    >>> from mackelab_toolbox.iotools import load
    >>> fully_qualified_name(load)
    'mackelab_toolbox.iotools.load'
    """
    import builtins

    # Based on https://stackoverflow.com/a/13653312
    name = getattr(o, '__qualname__', getattr(o, '__name__', None))
    if name is None:
        raise TypeError("Argument has no `__qualname__` or `__name__` "
                        "attribute. Are you certain it is a class or function?")
    module = o.__module__
    if module is None or module == builtins.__name__:
        return name
    else:
        return module + '.' + name

def argstr(args: tuple, kwargs: dict):
    """Reconstruct the string of arguments as it would have been passed to a
    function.

    Example
    -------
    >>> argstr((2, 4), {'a': 33})
    "2, 4, a=33"
    """
    s = ', '.join(f'{a} [{type(a)}]' for a in args)
    if len(s) > 0 and len(kwargs) > 0:
        s += ', '
    s += ', '.join(f'{k}:{v} [{type(v)}]' for k,v in kwargs.items())
    return s

class class_or_instance_method:
    """
    Method decorator which sets `self` to be either the class (when method
    is called on the class) or the instance (when method is called on an
    instance). Adapted from https://stackoverflow.com/a/48809254.

    .. Note:: This is clever, but it may hinder maintainability of your code.
    """
    def __init__(self, method, instance=None, owner=None):
        self.method = method
        self.instance = instance
        self.owner = owner

    def __get__(self, instance, owner=None):
        return type(self)(self.method, instance, owner)

    def __call__(self, *args, **kwargs):
        clsself = self.instance if self.instance is not None else self.owner
        return self.method(clsself, *args, **kwargs)

def print_api(obj_or_type,
              docstring_indent: int=4,
              show_class_docstring: bool=True,
              show_attributes: bool=False):
    """
    A quick method for printing the public API of an object.
    Attributes and methods beginning with an underscore are excluded.
    (I.e. neither private nor dunder methods are printed.)
    In IPython, a similar output can be achieved by calling `?` on each of the
    object's methods, although the output of this function is more compact
    and doesn't require knowing and typing each method name.

    The output begins with the object's own docstring, unless `show_class_docstring`
    is `False.
    By default only public methods are printed, along with their signature and
    docstring. Public attributes (i.e. those not starting with an underscore)
    will also be listed if `show_attributes` is `True`.

    Ordering of printed methods and attributes reflects that in which they
    appear in the class' definition.
    """
    import inspect

    indent_prefix = " "*docstring_indent
    if inspect.isfunction(obj_or_type) or inspect.ismethod(obj_or_type):
        print_function_api(obj_or_type)
        return   # EARLY EXIT
    elif not isinstance(obj_or_type, type):
        type_ = type(obj_or_type)
    else:
        type_ = obj_or_type
    if show_class_docstring:
        print(type_.__doc__)
    for attr, val in type_.__dict__.items():
        if not attr.startswith('_'):
            if inspect.isfunction(val) or inspect.ismethod(val):
                print_function_api(val, name=attr)
            elif show_attributes:
                print(attr)

def print_function_api(fn, name=None, docstring_indent: int=4):
    import textwrap
    import inspect

    indent_prefix = " "*docstring_indent
    if name is None:
        name = fn.__name__
    docstring = fn.__doc__
    if docstring is None:
        docstring = ""
    else:
        docstring = textwrap.dedent(docstring).strip("\n")
    print(f"{name}{inspect.signature(fn)}")
    print(textwrap.indent(docstring+"\n", indent_prefix))

# DEVNOTE: If you want to avoid the dependency on mackelab_toolbox by copying
#    this into your project, note that it is very easy to accidentally introduce
#    import cycles. For examples, the following won't work:
#
#        MyProject/__init__.py::
#          from analyses import analysis1, analysis2
#
#        MyProject/analyses.py::
#          from .utils import HideCWDFromImport
#          with HideCWDFromImport(__file__):
#            import typing
#          ...
#        MyProject/typing.py::
#          class MyType:
#            ...
#
#        MyProject/utils.py::
#          class HideCWDFromImport:
#            ...
#
#    The problem here is that `import .utils` is a package import, which
#    implicitely executes `import MyProject.__init__.py`.
#    The simplest way is to have a module containing only `HideCWDFromImport`,
#    place it in the same directory, and import without any dot in the path.
class HideCWDFromImport:
    """
    Context manager with prevents the current directory from being searched in
    an `import` statement. This is relevant if two conditions are satisfied:
    1. The most logical module name is the same as another package.
    2. Code will be executed from within the directory where the package resides.

    The motivating use case is the following: The project *MyProject* is
    organized as a pip package, with a combination of lower-level modules doing
    most of the heavy computations, and higher-level ones for analyzing results
    and plotting. The computations required specializing some builtin or
    third-party functionality – perhaps extra definitions to 'typing' were
    added, or the behaviour of 'parameters' was adjusted to account for these
    new types. These specializations _should_ be placed in a module 'typing' or
    'parameters', since they serve the same purpose as those base packages. They
    could like like this (the example of type hints is chosen here more for
    brevity than usefulness)::

        # MyProject.typing.py
        class Slice:
            ...

        # MyProject.analysis.py
        from typing import List
        from MyProject.typing Range

        def analyze_spectrum(data: List[float], window: Slice):
            ...

    This will work fine until the *analysis.py* script is executed from the root
    directory of *MyProject* (in which case `import typing` would point to
    *MyProject.typing* rather than the builtin *typing*). Since this isn't an
    unreasonable thing to do, we change *analysis.py* as follows::

        # MyProject.analysis.py – v2
        from mackelab_toolbox.meta import HideCWDFromImport
        with HideCWDFromImport(__file__):
            from typing import List
        from MyProject.typing Range

        def analyze_spectrum(data: List[float], window: Slice):
            ...

    This version now works no matter the current directory.

    .. Note:: Hiding the cwd is only required in packages that double as
       analysis directory. If instead the specialized types were defined is
       another package, say *lab_toolbox.typing*, no such special casing is
       necessary.

    .. Note:: If you would like to use this functionality without adding a
       dependency to *mackelab_toolbox* to your project (by copying its content
       into your own module), be careful not to introduce import cycles.
       For instance, the following would not work because
       `import MyProject.<module>` always imports the root package `MyProject`

            # MyProject.__init__.py
            from .analysis import analyze_spectrum, plot_spectrum

            # MyProject.typing.py
            class Slice:
                ...

            # MyProject.analysis.py – note that we import from *MyProject* now
            from typing import List
            from MyProject.utils import HideCWDFromImport
                # Import cycle: MyProject.__init__.py tries to import .analysis
            with HideCWDFromImport(__file__):
                from typing import List
            from MyProject.typing Range

            def analyze(data: List[float], window: Slice):
                ...

    .. Hint:: One can make this completely transparent and avoid the need for
       any special code in the analysis scripts by moving the “HideCWD” guard
       into the shadowing module::

            # MyProject.typing.py
            import sys
            from mackelab_toolbox.utils import hide_cwd_from_import

            if __name__ == "typing":
                # Imported as `import typing`
                # Remove from sys.modules and replace with builtin 'typing'
                del sys.modules['typing']
                with HideCWDFromImport(__file__):
                    import typing
            else:
                class Slice:
                    ...
    """

    def __init__(self, __file__: str=None):
        """
        :param:__file__: Path to any file in the directory we want to hide.
            Passing the value of `__file__` from the calling module should
            always be appropriate.
            Special case: if `__file__` points to a file named "__init__.py",
            then the _parent_ directory is excluded. This is to account for
            the use case where a subpackage shadows a third-party package,
            in which case the HideCWD guard needs to be placed in the
            *__init__.py* file within that subpackage.
        """
        # Remark: The reason we don't use os.getcwd() is because we want
        #         to avoid any unnecessary import. Any module name we import
        #         is a module name we can't shadow.

        directory, filename = __file__.rsplit('/', 1)
        if filename == "__init__.py":
            self.hidden_dir = directory.rsplit('/', 1)[0]
        else:
            self.hidden_dir = __file__.rsplit('/', 1)[0]

    def __enter__(self):
        # Python will search both in the script directory and the current directory
        # before looking at system locations, and this prevents us from loading
        # the base package 'parameters'
        # So we remove those locations from the path, noting their positions, and
        # after importing 'parameters' put them back in the correct position.
        import sys

        _script_dir = self.hidden_dir
        try:
            _script_dir_pos = sys.path.index(_script_dir)
        except ValueError:
            _script_dir_pos = None
        else:
            del sys.path[_script_dir_pos]
        try:
            _cur_dir_pos = sys.path.index('')
        except ValueError:
            _cur_dir_pos = None
        else:
            del sys.path[_cur_dir_pos]
        self._script_dir = _script_dir
        self._script_dir_pos = _script_dir_pos
        self._cur_dir_pos = _cur_dir_pos

    def __exit__(self, *args):
        import sys

        # Reinsert directories in inverse order, so positions are correct
        if self._cur_dir_pos is not None:
            sys.path.insert(self._cur_dir_pos, '')
        if self._script_dir_pos is not None:
            sys.path.insert(self._script_dir_pos, self._script_dir)
