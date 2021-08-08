"""
In order to transparently provide dynamic types, this module loads the
`typing` namespace from `typing_module`, and replaces itself with it in
the loaded sys.modules.
"""
# TODO:
# - Use a module level __getattr__ and/or import hooks (https://dev.to/dangerontheranger/dependency-injection-with-import-hooks-in-python-3-5hap)
#   instead of the current hackery for dynamic types.
#   This would obviate the need for freeze_types, allow theano to be unloaded,
#   and just generally be a lot more intuitive.
# - Split parts of 'typing.py' into topical submodules (esp. NumPy), like PureFunction already is,
#   with everything into a global 'typing' namespace

# Add the typing namespace to mackelab_toolbox
# This is the same trick used in the official typing module:
#     https://github.com/python/typing/blob/d79eddef37cb09ca9a3d763364c4feb7b8473402/src/typing.py#L2435
# with the added hackiness that we are replacing the current module in
# sys.modules.
import sys
from .typing_module import typing
# As a temporary solution, to avoid premature refactoring, we manually inject everything
# from subpackages into the typing namespace
from . import typing_pure_function as tpf
for name in tpf.__all__:
    setattr(typing, name, getattr(tpf, name))
sys.modules[__name__] = typing
