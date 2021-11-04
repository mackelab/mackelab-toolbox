"""
Pydantic-aware classes for Pint objects.
Load these with `typing.load_pint()`
"""

import sys
from typing import Union
import pint as pint
from mackelab_toolbox import utils
from mackelab_toolbox.typing.typing_module import typing as mtbT
UnitRegistry = pint.UnitRegistry
ApplicationRegistry = getattr(pint, 'ApplicationRegistry',
                              getattr(pint, 'LazyRegistry', None))
# NB: Pint v0.18 changed to 'ApplicationRegistry'; before was 'LazyRegistry'
#     The second `getattr` is in case LazyRegistry is removed in the future

def _load_pint(typing):
    "Called by `typing.load_pint()` to make Pint objects available."
    typing.PintValue = PintValue
    typing.PintUnit  = PintUnit
    typing.add_json_encoder(pint.quantity.Quantity, PintValue.json_encoder)
    typing.add_json_encoder(pint.unit.Unit, PintUnit.json_encoder)
    typing.add_unit_type(PintUnit)
    utils._terminating_types |= {pint.quantity.Quantity, pint.unit.Unit}
    utils.terminating_types = tuple(utils._terminating_types)

class PintValue(pint.Quantity):
    """
    Before parsing a serialized PintValue, one needs to set the Pint unit
    registry. There should only be one unit registry within a project, so this
    is set as a *class* variable of `PintValue`.
    >>> import pint
    >>> ureg = pint.UnitRegistry()
    >>> from mackelab_toolbox.typing import PintUnit
    >>> PintUnit.ureg = ureg
    """
    @classmethod
    def __get_validators__(cls):
        # partial doesn't work because Pydantic expects a particular signature
        yield cls.validate

    @staticmethod
    def validate(v):
        if isinstance(v, pint.Quantity):
            return v
        elif mtbT.json_like(v, "PintValue"):
            return PintUnit.ureg.Quantity.from_tuple(v[1])
        else:
            raise ValueError(f"Value {v} is neither a Pint Quantity nor a "
                             "recognized serialization of a Pint Quantity.")

    @staticmethod
    def json_encoder(v):
        return ("PintValue", v.to_tuple())
# Pint calls its objects 'Unit' and 'Quantity', so it seems logical to provide
# the same naming scheme. (
PintQuantity = PintValue

class PintUnitMeta(type):

    @property
    def ureg(cls) -> Union[UnitRegistry, ApplicationRegistry]:
        return cls._ureg or cls._get_and_set_application_registry()

    @ureg.setter
    def ureg(cls, value: Union[UnitRegistry, ApplicationRegistry]):
        if not isinstance(value, (UnitRegistry, ApplicationRegistry)):
            raise TypeError(f"The registry assigned to `{cls.__name__}.ureg` must be a "
                            f"Pint UnitRegistry. Received {value} (type: {type(value)}).")
        cls._ureg = value
    def _get_and_set_application_registry(cls):
        cls._ureg = pint.get_application_registry()
        return cls._ureg

class PintUnit(pint.Unit, metaclass=PintUnitMeta):
    # TODO: Type check assignment to ureg
    _ureg: pint.UnitRegistry = None

    def apply(self, value) -> pint.Quantity:
        """
        Give `value` these units. In contrast to using multiplication by a unit
        to construct a `~Pint.Quantity`, `apply` is meant for situations where
        we are unsure whether `value` already has the desired units.

        - If `value` has no units: equivalent to multiplying by `self`.
        - If `value` has the same units as `self`: return `value` unchanged.
        - If `value` has different units to `self`: raise `ValueError`.

        Perhaps a more natural, but more verbose, function name would be
        "ensure_units".
        """
        if (not isinstance(value, pint.Quantity)
              or value.units == PintUnit.ureg.dimensionless):
            return value * self
        elif value.units == self:
            return value
        else:
            try:
                return value.to(self)
            except pint.DimensionalityError as e:
                raise ValueError(f"Cannot apply units `{self}` to value `{value}`: "
                                 f"it already has units `{value.units}`.") from e

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @staticmethod
    def validate(v):
        if isinstance(v, pint.Unit):
            return v
        elif isinstance(v, pint.Quantity):
            if v.magnitude != 1:
                raise ValueError("Quantities can only be converted to units "
                                 "if they have unit magnitude.")
            return v.units
        if mtbT.json_like(v, "PintUnit"):
            return PintUnit.ureg.Unit(v[1][0])
        else:
            raise ValueError(f"Value {v} is neither a Pint Unit, nor a Pint "
                             "Quantity with magnitude 1, nor a recognized "
                             "serialization of a Pint Quantity.")

    @staticmethod
    def json_encoder(v):
        return ("PintUnit", (str(v),))
