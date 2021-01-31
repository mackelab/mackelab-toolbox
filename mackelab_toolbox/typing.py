"""
In order to transparently provide dynamic types, this module loads th
`typing` namespace from `typing_module`, and replaces itself with it in
the loaded sys.modules.
"""

# Add the typing namespace to mackelab_toolbox
# This is the same trick used in the official typing module:
#     https://github.com/python/typing/blob/d79eddef37cb09ca9a3d763364c4feb7b8473402/src/typing.py#L2435
# with the added hackiness that we are replacing the current module in
# sys.modules.
import sys
from mackelab_toolbox.typing_module import typing
sys.modules[__name__] = typing
