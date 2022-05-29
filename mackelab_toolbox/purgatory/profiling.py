def getfullsize(obj, _already_sized=None) -> int:
    """
    A recursive version of `sys.getsizeof`.
    For example, on mutable objects like `list` and `dict`, `sys.getsizeof`
    only depends on the number of elements (it gives the size of the container,
    but not of its contents).
    In contrast, `getfullsize` will recurse into the container and add
    the size of all elements.

    Implemented as set of recursion rules for different types.

    In order to avoid giving incorrect results, this function purposefully
    avoids duck typing: the type associated to a recursion rules must match
    exactly for it to be applied. So in contrast to `sys.getsizeof`, this
    function will only work on a small set of supported types.

    Also, the function keeps track of variables it has already sized,
    so that repeated references to the same object are not counted
    multiple times (although the size of the additional pointers *is*
    added to the total).

    **Extending support**
    To support additional types, there are two paths:

    - Add a function to the `size_functions` dictionary defined in this module.
      This is the only mechanism available to add support from built-in or
      3rd-party types.

    - Add a method `__fullsizeof__` to objects.
      In analogy with how `getsizeof` checks for a `__sizeof__` method,
      `getfullsize` will first check for a method of this name and use it if
      available. Typically such a method will consist of calling `getfullsize`
      on internal data structures of recognized types.
      Importantly, this method must take TWO arguments, and pass on the
      second to any recursive call to `getfullsize`.
    """
    if _already_sized is None:
        _already_sized = set()  # Can’t use frozenset as default arg, because we need a mutable
    elif id(obj) in _already_sized:
        # We are inside a recursive call, and the size of `obj` has already been accounted for
        return 0

    sizefn = getattr(obj, "__fullsizeof__", None)
    if sizefn is not None:
        _already_sized.add(id(obj))
        return sizefn(obj, _already_sized)

    typename = type(obj).__qualname__
    for substr, sizefn in size_functions.items():
        if substr in typename:
            res = sizefn(obj, _already_sized)
            if res is NotImplemented:
                continue
            else:
                _already_sized.add(id(obj))
                return res

    raise TypeError("`getfullsize` does not know how to measure the memory "
                    f"size of objects of type {typename}. See its "
                    "documentation for instructions on extending support.")


# %% [markdown]
# To avoid importing everything all the type, size checkers are stored in a dictionary, whose keys match substrings of `str(obj)`. Thus, if `type(obj)` includes `"array"`, the corresponding size checker is called. It then needs to import all types with that name, but at least it *only* imports types with that name.
#
# If there are multiple possible matches in the dictionary, the first one is used. If the checker returns `NotImplemented`, then the second is tried, etc.
#
# Note also that functions can be added to explicitely *disallow* measuring size, with a user-friendly error message. This can be useful for things like infinite iterators.
#
# **Caution to implementers** to ensure that memory is not counted twice:
# - Always use `getfullsize` in recursive calls (not `sys.getsizeof`).
# - Remember to pass `_already_sized` through to any recursive call.

# %%
def _builtin_size(obj, _already_sized):
    if type(obj) in {int, float, str}:
        return getsizeof(obj)
    else:
        return NotImplemented
def _sequence_size(obj, _already_sized):
    if type(obj) in {tuple, list, set, frozenset}:
        return getsizeof(obj) + sum(getfullsize(o, _already_sized) for o in obj)
    else:
        return NotImplemented
def _mapping_size(obj, _already_sized):
    from collections import OrderedDict
    if type(obj) in {dict, OrderedDict}:
        return (getsizeof(obj)
                + sum(getfullsize(k, _already_sized) + getfullsize(v, _already_sized) for k,v in obj.items()))
    else:
        return NotImplemented


# %% [markdown]
# Measuring sizes of NumPy arrays:
# - `.nbytes` returns the number of bytes consumed by the elements of the array, and *only*.
#   This excludes overhead (104 bytes) but includes the memory of elements in views.
# - `getsizeof` it returns only the size of the overhead (since the actual data is referenced).
# - Doing `.nbytes` + overhead is unsatisfactory, because it can lead to double counting.
#   For example, in the following case, it would count almost twice the actually used memory for array `A`:
#
#       [A[:-1], A[1:]]
#
# - When given a view, we return the size of the view itself, plus the *fullsize* of its underlying array.
#   The `_already_sized` tracking in `getfullsize` ensures that the underlying array is counted only once.
#
#   In the case where an object *only* uses a partial view of an array, this would produce a larger number
#   then the memory actually used. However, in such cases one could argue that the full array size should
#   still be counted (since it is somehow used), and since that leads to much simpler computer logic, it
#   is the interpretation we take.

# %%
def _numpy_size(obj, _already_sized):
    if type(obj) is np.ndarray:
        s = getsizeof(obj)
        if obj.base is not None:
            s += getfullsize(obj.base, _already_sized)
        return s
    else:
        return NotImplemented


# %%
size_functions = {
    "int"        : _builtin_size,
    "float"      : _builtin_size,
    "str"        : _builtin_size,
    "tuple"      : _sequence_size,
    "list"       : _sequence_size,
    "set"        : _sequence_size,
    "frozenset"  : _sequence_size,
    "dict"       : _mapping_size,
    "OrderedDict": _mapping_size,
    "array"      : _numpy_size
}
