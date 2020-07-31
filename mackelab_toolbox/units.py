"""
This module collects a few convenience functions for working with units.
These functions work with both `pint` and `quantities`, and attempt to do the
most sensible thing when called on plain (unitless) values.

Care is taken not to import either `pint` or `quantities` if it isn't already
loaded.
"""

from typing import Any
from .utils import Singleton

class UnitlessT(int, metaclass=Singleton):
    """
    This object allows to test for presence of units simply writing
    ``axis.units is unitless``.
    Instantiating with ``1`` allows us to use ``value * self.unit`` wherever
    we want.
    The use of the :class:`~utils.Singleton` metaclass ensures that ``is``
    checks are always valid. It is important to use ``is`` and not ``==``
    because a lot of things can be equal to 1.
    """
    # Pydantic-compatible validators
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v):
        # Raising an error signals to calling code e.g. a Pydantic parser,
        # to know it should try casting to another type.
        if v is not None and v != unitless:
            raise ValueError
        return cls(v)
unitless = UnitlessT(1)

def detect_unit_library(value: Any):
    """
    Return the unit libary on which `value` depends.
    We detect types with duck typing rather than testing
    against the actual types for two reasons:
      - Testing against types would force importing all quantities
        libraries, and therefore installing them.
      - In theory other libraries could implement these methods, and
        they would work as well.

    Parameters
    ----------
    value: Any scalar | tuple | list | set
        Any value is accepted; if the type isn't recognized, ``False`` is
        returned.
        `tuple`, `list` and `set` are iterated through, and return a list of
        same length where elements are either 'pint', 'quantities' or None.

    Returns
    -------
    - 'pint' | 'quantities' | 'none'
          Returns ``None`` if no unit library is recognized.
    - List of the [ 'pint' | 'quantities' | 'none' ]
    """
    if isinstance(value, (list, tuple, set)):
        return [detect_unit_library(v) for v in value]
    elif hasattr(value, 'compatible_units'):
        return 'pint'
    elif hasattr(value, 'simplified'):
        return 'quantities'
    else:
        return 'none'

def is_dimensionless(value: Any):
    if detect_unit_library(value) == 'none':
        return True
    else:
        return len(value.dimensionality) == 0  # Works with both pint & quantities

def unit_convert(value, to):
    """
    :param value: Value to convert.
    :param to: The unit to convert to.
        **NOTE**: This must be an actual unit, not e.g. the string 's'.
    """

    if to in [unitless, 'none', 1]:
        assert is_dimensionless(value)
        return value
    else:
        vallib = detect_unit_library(value)
        tolib = detect_unit_library(to)
        assert tolib != 'none'
        if vallib == 'none':
            # Cast values that had no units (but NOT explicitly dimensionless
            # quantities) to the expected unit
            return value * to
        else:
            vallib == tolib
            if vallib == 'pint':
                return value.to(to)
            elif vallib == 'quantities':
                return value.rescale(to)
            else:
                assert False  # Should not reach this point

def get_magnitude(value, in_units_of=None):
    vallib = detect_unit_library(value)
    if vallib == 'none':
        assert in_units_of is None
        return value
    elif in_units_of is None:
        return value.magnitude
    else:
        return unit_convert(value, in_units_of).magnitude
