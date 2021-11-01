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

# WIP: As a first step in splitting into topical submodules,
#      we leave typing_module.py untouched, but instead of replacing
#      `mackelab_toolbox.typing` with the `TypingContainer` defined therein,
#      we redirect attribute access to the TypingContainer with __getattr__.
from . import typing_module
from .typing_pure_function import *

def __getattr__(attr):
    return getattr(typing_module.typing, attr)
