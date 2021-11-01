"""
Pydantic-aware classes for Quantities objects.
Load these with `typing.load_quantities()`
"""

import quantities
from mackelab_toolbox import utils

def _load_quantities(typing):
    "Called by `typing.load_quantities()` to make Quantities objects available."
    typing.QuantitiesValue = QuantitiesValue
    typing.QuantitiesUnit = QuantitiesUnit
    typing.QuantitiesDimension = QuantitiesDimension
    # Must have higher priority than numpy types
    typing.add_json_encoder(quantities.quantity.Quantity,
                            QuantitiesValue.json_encoder,
                            priority=5)
    typing.add_json_encoder(quantities.dimensionality.Dimensionality,
                            QuantitiesUnit.json_encoder,
                            priority=5)
    typing.add_unit_type(QuantitiesUnit)
    utils._terminating_types |= {quantities.quantity.Quantity, quantities.dimensionality.Dimensionality}
    utils.terminating_types = tuple(utils._terminating_types)

class QuantitiesValue():
    @classmethod
    def __get_validators__(cls):
        yield from (cls.json_noerror,  # For whatever reason, lambda doesn't work
                    cls.validate_value)

    @classmethod
    def json_noerror(cls, v):
        return cls.json_decoder(v, noerror=True)

    @classmethod
    def validate_value(cls, v):
        pq = QuantitiesUnit.pq()
        if isinstance(v, pq.Quantity):
            return v
        raise TypeError

    @staticmethod
    def json_encoder(v):
        return ("QuantitiesValue", (v.magnitude, str(v.dimensionality)))
    @staticmethod
    def json_decoder(v, noerror=False):
        pq = sys.modules.get('quantities', None)
        if pq is None:
            raise ValueError("'Quantities' module is not loaded.")
        elif (isinstance(v, (tuple,list))
              and len(v) > 0 and v[0] == "QuantitiesValue"):
            return pq.Quantity(v[1][0], units=v[1][1])
        elif noerror:
            # Let another validator try to parse the value
            return v
        else:
            raise ValueError("Input is incompatible with QuantitiesValue.json_decoder. "
                   f"Input value: {v} (type: {type(v)})")

class QuantitiesUnit(QuantitiesValue):
    """
    Exactly the same as QuantitiesValue, except that we enforce the magnitude
    to be 1. In contrast to Pint, Quantities doesn't seem to have a unit type.
    """
    @staticmethod
    def pq():
        pq = sys.modules.get('quantities', None)
        if pq is None:
            raise ValueError("'Quantities' module is not loaded.")
        return pq

    @classmethod
    def validate_value(cls, v, field=None):
        if field is None:
            field = SimpleNamespace(name="")
        pq = QuantitiesUnit.pq()
        if isinstance(v, pq.Quantity):
            if v.magnitude != 1:
                raise ValueError(f"Field {field.name}: Units must have "
                                 "magnitude of one.")
            return v
        raise TypeError

class QuantitiesDimension():

    @classmethod
    def __get_validators__(cls):
        yield from (lambda v: cls.json_decoder(v, noerror=True),
                    lambda v: cls.validate_value(v, noerror=True),
                    cls.validate_dim)

    @staticmethod
    def validate_dim(v):
        pq = QuantitiesUnit.pq()
        if isinstance(v, pq.dimensionality.Dimensionality):
            return v
        raise TypeError
    @staticmethod
    def validate_value(v, noerror=False):
        pq = QuantitiesUnit.pq()
        if isinstance(v, pq.Quantity):
            return v.dimensionality
        elif noerror:
            return v
        else:
            raise TypeError

    @staticmethod
    def json_encoder(v):
        return ("QuantitiesDimension", (str(v),))
    @staticmethod
    def json_decoder(v, noerror=False):
        pq = sys.modules.get('quantities', None)
        if pq is None:
            raise ValueError("'Quantities' module is not loaded.")
        if isinstance(v, (tuple,list)) and len(v) > 0 and v[0] == "QuantitiesDimension":
            return pq.quantity.validate_dimensionality(v[1][0])
        elif noerror:
            # Let another validator try to parse
            return v
        else:
            raise ValueError("Input is incompatible with QuantitiesDimension.json_decoder. "
                   f"Input value: {v} (type: {type(v)})")
