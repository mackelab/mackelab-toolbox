"""
Adds the following dynamic types to mackelab_toolbox.typing:

- FloatX
- Symbolic
- Shared
- Tensor
- AnyRNG
"""

from typing import Union
import numpy as np
import mackelab_toolbox as mtb
import mackelab_toolbox.typing
from mackelab_toolbox.cgshim import *

__validation_types = {}  # Serves to ensure get_type() doesn't define the same type twice
def get_type(baseT, Tname, namedesc, validators=()):
    """
    Parameters
    ----------
    baseT: type
        The type for which we want a type hint.
    Tname: str
        The name to assign to the type ("type name"). Normally capitalized.
    namedesc: str
        Name to use in error messages to describe the type. Normally not
        capitalized.
        E.g. setting `namedesc` to "symbolic", an error message may read
        "This field expects a symbolic variable."
    """
    # Structure:
    #   - Check __validation_types cache. If symbolic type is already there,
    #     skip everything and jump to the return.
    #   - Otherwise, create the typed base class TypedSymbolic, a metaclass
    #     SymbolicMeta which returns a subclass of TypedSymbolic, and a
    #     Symbolic class which has SymbolicMeta as metaclass
    #     + Store Symbolic in the cache
    #   - Return the value from the cache
    # The metaclass logic here, like the one in mtb.typing.Array, is based
    # on a suggestion by the pydantic developer:
    # https://github.com/samuelcolvin/pydantic/issues/380#issuecomment-594639970
    if baseT not in __validation_types:
        assert isinstance(validators, tuple)

        # For symbolic types, we just assume that val has already been properly casted.
        # We do this to keep the reference to the original variable
        class ValidatingType(baseT):
            @classmethod
            def __get_validators__(cls):
                yield from validators + (cls.validate_type,)

            @classmethod
            def validate_type(cls, value, field):
                if not isinstance(value, baseT):
                    raise TypeError(f"{field.name} expects a {namedesc} variable.")
                if cls._ndim  is not None and value.ndim != cls._ndim:
                    raise TypeError(f"{field.name} expects a variable with "
                                    f"{cls._ndim} dimensions.")
                if getattr(value, 'name', None) is None:
                    value.name = field.name
                if not np.can_cast(value.dtype, field.type_):
                    logger.error(
                        f"{field.name} expects a variable of type {field.type_}. "
                        f"Provided value has type {value.dtype}, which is not a "
                        "permitted cast. Note that for symbolic variables, we "
                        "perform no casting, so types must match exactly.")
                elif not np.dtype(value.dtype) is np.dtype(field.type_):
                    logger.warning(
                        f"{field.name} expects a variable of type {field.type_}. "
                        f"Provided value has type {value.dtype}. "
                        "Note that for symbolic variables, we perform "
                        "no casting, so types must match exactly.")
                return value

        class Meta(baseT.__class__):
            def __getitem__(self, args):
                if isinstance(args, tuple):
                    T = args[0]
                    ndim = args[1] if len(args) > 1 else None
                    extraargs = args[2:]  # For catching errors only
                else:
                    T = args
                    ndim = None
                    extraargs = []
                if (not isinstance(T, type) or len(extraargs) > 0
                    or not isinstance(ndim, (int, type(None)))):
                    raise TypeError(
                        f"`{Tname}` must be specified as either "
                        f"`{Tname}[T]` or `{Tname}[T, n], where `T` is a "
                        "type and `n` is an int. (received: "
                        f"[{', '.join((str(a) for a in args))}]).")
                return type('Symbolic', (ValidatingType,),
                            {'dtype': mtb.typing.convert_dtype(T),
                             '_ndim': ndim})

        __validation_types[baseT] = Meta(Tname, (baseT,), {})

    return __validation_types[baseT]

FloatX = mtb.typing.NPValue[shim.config.floatX]

"""
A type-hint compatible symbolic type.
The type returned depends on which symbolic library (if any) is loaded.
Types are cached to avoid recreating the same type more than once.
"""
if shim.config.library == 'numpy':
    Symbolic = mtb.typing.Array
else:
    Symbolic = get_type(shim.config.SymbolicType,
                        'Symbolic', 'symbolic')

"""
Return a type-hint compatible symbolic shared type.
The type returned depends on which symbolic library (if any) is loaded.
Types are cached to avoid recreating the same type more than once.
"""
if shim.config.library == 'numpy':
    Shared = mtb.typing.Array
else:
    def make_shared(cls, value, field):
        if not shim.isshared(value) and not shim.is_symbolic(value):
            value = shim.shared(value)
        return value
    Shared = get_type(shim.config.SymbolicSharedType,
                      'Shared', 'shared', validators=(make_shared,))

"""
Tensor
A Union type-hint compatible with any of Array, Symbolic or Shared.

NOTE: For shared variables does not check that data is actually a
      a tensor. Should we do this ?
"""
types = tuple(set((mtb.typing.Array, Symbolic, Shared)))
    # set to remove duplicates
if types not in __validation_types:
    class ShimTensorUnion:
        def __getitem__(self, args):
            return Union[tuple(T[args] for T in types)]
Tensor = ShimTensorUnion()

"""
RandomStreams
A Union of Numpy RandomState and the Theano RandomStreans
"""
if shim.config.library == 'numpy':
    AnyRNG = mtb.typing.RandomState
else:
    if 'TheanoRandomStreams' not in __validation_types:
        class TheanoRandomStreams:
            # Theano shared random streams store a np.random.RandomState
            # instance as `gen_seedgen`, and use that to seed further
            # generators for each Op.
            # Thus as long as we store the state of `gen_seedgen`, we
            # will recreate subsequent ops with the same seeds.
            @classmethod
            def __get_validators__(cls):
                yield cls.validate
            @classmethod
            def validate(cls, value, field):
                # WARNING: I haven't thoroughly tested that state is fully restored
                if isinstance(value, shim.config.SymbolicRNGType):
                    return value
                elif isinstance(value, shim.config.RandomStateType):
                    random_state = value
                else:
                    random_state = mtb.typing.RandomState.validate(value, field)
                random_streams = shim.config.RandomStreams()
                random_streams.gen_seedgen.set_state(random_state.get_state())
                return random_streams
            @classmethod
            def __modify_schema__(cls, field_schema):
                return mtb.typing.RandomState(field_schema)
            @classmethod
            def json_encoder(cls, v):
                return v.gen_seedgen.get_state()
        mtb.typing.add_json_encoder(
            shim.config.SymbolicRNGType,
            TheanoRandomStreams.json_encoder)
    AnyRNG = TheanoRandomStreams

mtb.typing.add_numerical_type(
    (Symbolic[np.number],   Shared[np.number]))
mtb.typing.add_scalar_type(
    (Symbolic[np.number,0], Shared[np.number,0]))
if isinstance(shim.config.SymbolicSharedType, type):
    mtb.typing.add_json_encoder(
        shim.config.SymbolicSharedType, lambda x: x.get_value())
