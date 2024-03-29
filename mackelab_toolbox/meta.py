from __future__ import annotations

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

def print_subpackage_listing(__file__: str, namespace: Dict[str,object]):
    """
    Print the list of available objects, organized by the module where
    they are defined.

    This is intended to make collections of functions more discoverable in
    interactive sessions. For example, a set of analysis tools may use this to
    implement a `help()` function listing all functions.
    This doesn't replace well-written documentation, but it has the advantage
    of being immediately available during a session, and always being up to date.

    Typical usage in a __init__ file would look like:

        # mytools/__init__.py

        from .analysis import *
        from .post_analysis import *

        def help():
            from mackelab_toolbox.meta import print_subpackage_listing
            print_subpackage_listing(__file__, globals())

    Parameters
    ----------
    __file__: The file from which this function is called. In most cases
        ``__file__`` should be appropriate.

    namespace: The dictionary of identifiers available from within the module
        pointed to by `__file__`. Normally this would be ``globals()``; a
        subset of the contents of the ``globals()`` would also be appropriate.
        In any case, only entries which derive from the same parent as
        `__file__` will be shown, since those are generally the ones that
        are relevant.
    """
    from textwrap import dedent
    import inspect
    import sys
    prefix = "  "     # Indent prefix indicating sections
    outputwidth = 80  # Minimum expected width of the console window

    # Parse __file__. Deal with __init__.py specially, since in that case the parent is both name and root
    assert __file__ is not None
    for nm, mod in sys.modules.items():
        if getattr(mod, '__file__', None) == __file__:
            break
    assert mod.__file__ == __file__  # Loop actually found a match
    __name__ = mod.__name__
    directory, filename = __file__.rsplit('/', 1)
    if filename == "__init__.py":
        root = __name__
    else:
        root = __name__.rsplit('.', 1)[0]  # The subpackage containing this file
    # At this point, `__name__` and `__file__` are as if we were executing from
    # within the file pointed to by `__file__`.
    # `root` is the parent from which objects in `namespace` can be imported.

    objs = {}
    root = __name__.rsplit('.', 1)[0]  # The subpackage containing this file
    for nm, obj in namespace.items():
        if (nm.startswith('_')
              or not hasattr(obj, '__module__')  # Required for last test, but forces exclusion of variables
              or root not in obj.__module__      # Only show nephew submodules
              or obj is help):  # Don't show this help function
            continue
        if obj.__module__ not in objs:
            objs[obj.__module__] = {}
        objs[obj.__module__][nm] = obj

    print(dedent(f"""
        The following classes can be imported as

          from {__name__} import <C1>, <C2>, …

        where <C1>, <C2>, etc. are replaced by their names.

        Classes are grouped by the module in which they are defined.

        Additional information is displayed to the right of each class,
        currently its concrete (non-virtual) parents.
        The content of this information is field may change in the future.
        """))

    for module in sorted(objs):
        if module.startswith(f"{root}."):
            mod_nm = module[len(root)+1:]
        else:
            mod_nm = module
        print(f"From <…>.{mod_nm}:")
        max_len = max(len(nm) for nm in objs[module])
        info_indent = " "*(len(prefix) + max_len + 2)
        info_width = outputwidth - len(info_indent)

        for obj_nm in sorted(objs[module]):
            obj = objs[module][obj_nm]
            # Info is composed of all parent types in MRO which are
            # defined below `root`
            # We break lines so that in most cases they won't wrap
            # (which breaks the alignment used to denote sections)
            if isinstance(obj, type):
                parents = [T.__qualname__ for T in obj.mro()[1:]
                           if root in T.__module__]
                if parents:
                    infolines = [parents[0]]
                    for parent in parents[1:]:
                        if len(infolines[-1]) + len(parent) > info_width:
                            infolines[-1] += ","
                            infolines.append(info_indent + parent)
                        else:
                            infolines[-1] += ", " + parent
                else:
                    infolines = [""]
                info = '\n'.join(infolines)
            else:
                info = f"function, {inspect.signature(obj)}"
            print(f"{prefix}{obj_nm:<{max_len}}: {info}")
        print()  # Empty line between module sections

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
#    Or to package HideCWDFromImport in an external package, as we've done here.
class HideCWDFromImport:
    """
    Context manager with prevents the current directory from being searched in
    an `import` statement. This is relevant if two conditions are satisfied:
    1. The most logical module name is the same as another package.
    2. Code will be executed from within the directory where the package resides.

    .. Caution:: The following names cannot be hidden, as they are used by the
       context itself: *sys*, *os*.

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

    This version now works irrespective of the current directory.

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
            Special cases:
              -`__file__` is `None` => Infer the current directory with os.getcwd()
                  Note that this is only appropriate in interactive sessions;
                  for scripts, the cwd is not the same as the directory
                  containing the file.
              - `__file__` points to a file named "__init__.py".
                  In this case the _parent_ directory is excluded. This is to
                  account for the use case where a subpackage shadows a
                  third-party package, in which case the HideCWD guard needs to
                  be placed in the *__init__.py* file within that subpackage.
        """
        # Remark: Any module name we import is a module name we can't shadow;
        #         we can’t have a module 'os' in the current directory, because
        #         when we `import os` below, the current directory is still
        #         in the search path.
        #         Thus we limit ourselves to two modules: 'sys' and 'os'.
        import os  # Depending on `os` allows us to deal correctly with both *nix and Windows paths
        if __file__ is None:
            self.hidden_dir = os.getcwd()
        else:
            directory, filename = os.path.split(__file__)
            if filename == "__init__.py":
                self.hidden_dir = os.path.dirname(directory)
            else:
                self.hidden_dir = directory

    def __enter__(self):
        # Python will search both in the script directory and the current directory
        # before looking at system locations, and this prevents us from loading
        # the base package 'parameters'
        # So we remove those locations from the path, noting their positions, and
        # after importing 'parameters' put them back in the correct position.
        import sys

        _script_dir = self.hidden_dir
        _script_dir_pos = [i for i, path in enumerate(sys.path)
                           if path == _script_dir]
        for i in _script_dir_pos[::-1]:
            del sys.path[i]
        _cur_dir_pos = [i for i, path in enumerate(sys.path)
                        if path == ""]
        for i in _cur_dir_pos[::-1]:
            del sys.path[i]
        self._script_dir = _script_dir
        self._script_dir_pos = _script_dir_pos
        self._cur_dir_pos = _cur_dir_pos

    def __exit__(self, *args):
        import sys

        # Reinsert directories in inverse order, so positions are correct
        for i in self._cur_dir_pos:
            sys.path.insert(i, "")
        for i in self._script_dir_pos:
            sys.path.insert(i, self._script_dir)
            
class HideModule:
    """
    Accepts a list of module names, which will be removed from the list
    of already imported modules within the context.
    (Specifically, they are removed from `sys.modules` upon entering the
    context, and reinstated upon exit.)
    If modules are not in `sys.modules`, they are not removed, nor imported on
    exit.
    It is an error to import a hidden module within the context, since then
    it is ambiguous whether the previous or the new module should be kept.
    For consistency, this applies even if the module was not imported
    before entering the context.
    
    Rationale: This context messes in a pretty heavy-handed way with Python's
    import mechanism, and there should be few good reasons to use it.
    The anticipate usage, and the one which motivated its implementation, is
    to workaround incompatible dependencies. For example, as of this writing,
    the package `Sumatra` still requires an old version (1.8) of Django.
    If we also import `Bokeh`, we can run into problems, because `Bokeh`
    checks during import whether `Django` is already imported, and if it is,
    assumes that it is a more recent version. Since Bokeh's dependency on Django
    is optional, it can be imported as long as we remove Django from the list of
    imported modules. This can be done either by ensuring Bokeh is imported
    first, or by importing it within a `HideModule` context:
    
    >>> from mackelab_toolbox.meta import HideModule
    >>> with HideModule('django'):
    >>>     import bokeh

    Parameters
    ----------
    *module_list: Modules to hide, passed as separate arguments.
        Names must match exactly the keys in `sys.modules`.
    
    Raises
    ------
    ImportError (on exit): If a hidden module is imported within the context.
    
    """
    def __init__(self, *modules_to_hide: str):
        self.modules_to_hide = set(modules_to_hide)
        self._modules_hidden = None
        
    def __enter__(self):
        import sys
        self._modules_hidden = {}
        for nm in self.modules_to_hide:
            if nm in sys.modules:
                self._modules_hidden[nm] = sys.modules[nm]
                del sys.modules[nm]
                
    
    def __exit__(self, *args):
        import sys
        reimported_modules = self.modules_to_hide & set(sys.modules)
        if reimported_modules:
            raise ImportError("The following hidden modules were reimported "
                              f"within the `HideModule` context: {reimported_modules}")
        for nm, m in self._modules_hidden.items():
            sys.modules[nm] = m
