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
import logging
logger = logging.getLogger(__name__)

json_like = mtb.typing.json_like

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
            _ndim = None
            nptype = None

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
                    # Pydantic appends the type to each subfield name when a
                    # field uses Union[] to specify multiple types.
                    # I haven't found a way to work back to the original type,
                    # which arguably would be better than manipulating the string.
                    name = field.name
                    if name.endswith('_'+Tname):    # Python >=3.9: removesuffix()
                        name = name[:-len(Tname)-1]
                    value.name = name
                if cls.nptype is not None:
                    target_nptype = cls.nptype
                    # If nptype is an NPValue, we want _its_ nptype
                    target_nptype = getattr(target_nptype, 'nptype', target_nptype)
                    if not np.can_cast(value.dtype, target_nptype):
                        logger.error(
                            f"{field.name} expects a {namedesc} variable with "
                            f"data type {target_nptype}. Provided value has dtype "
                            f"{value.dtype}, which is not a permitted cast. "
                            "Note that for symbolic variables, we perform no "
                            "casting, so types must match exactly.")
                    elif not np.dtype(value.dtype) is np.dtype(target_nptype):
                        logger.warning(
                            f"{field.name} expects a {namedesc} variable with "
                            f"data type {target_nptype}. Provided value has dtype "
                            f"{value.dtype}, which is not a permitted cast. "
                            "Note that for symbolic variables, we perform no "
                            "casting, so types must match exactly.")
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
                if isinstance(T, np.dtype):
                    T = T.type
                elif isinstance(T, str):
                    T = np.dtype(T).type
                if (not isinstance(T, type) or len(extraargs) > 0
                    or not isinstance(ndim, (int, type(None)))):
                    raise TypeError(
                        f"`{Tname}` must be specified as either "
                        f"`{Tname}[T]` or `{Tname}[T, n], where `T` is a "
                        "type and `n` is an int. (received: "
                        f"[{', '.join((str(a) for a in args))}]).")
                return type(Tname, (ValidatingType,),
                            {'nptype': mtb.typing.convert_nptype(T),
                             '_ndim': ndim})

        __validation_types[baseT] = Meta(Tname, (ValidatingType,), {})
        # __validation_types[baseT] = Meta(Tname, (baseT,), {})

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
Random streams:
- RNGStream: The default random stream.
    + On Numpy : RNGenerator
    + On Theano: MRG_RandomStream  (theano.sandbox.rng_mrg)
- RandomStateStream: A random stream using the legacy RandomState object
    + On Numpy: RandomState
    + On Theano: RandomStream  (theano.tensor.random.utils)
- AnyRNG: Union of all Numpy and Theano types
    + If Theano is active, preference is given to Theano types.
"""
if shim.config.library == 'numpy':
    AnyRNG = Union[mtb.typing.RandomState]
    RNGStream = mtb.typing.RNGenerator
    RandomStateStream = mtb.typing.RandomState
else:
    if 'RandomStateStream' not in __validation_types:
        import theano
        class RandomStateStream:
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
                if isinstance(value, shim.config.SymbolicNumpyRNGType):
                    return value
                elif isinstance(value, shim.config.SymbolicMRGRNGType):
                    raise TypeError(
                        "Expected a NumPy RandomStream (theano.tensor.random.utils.RandomStream); "
                        "Received an MRG RandomStream (theano.sandbox.rng_mrg.MRG_RandomStream).")
                elif isinstance(value, shim.config.RandomStateType):
                    random_state = value
                elif json_like(value, "RandomStateStream"):
                    random_state = mtb.typing.RandomState.validate(value[1], field)
                else:
                    try:
                        random_state = mtb.typing.RandomState.validate(value, field)
                    except (TypeError, ValueError):
                        raise ValueError("Unable to deserialize RandomStateStream; "
                                         f"received: {value}.")
                random_stream = theano.tensor.random.utils.RandomStream()
                random_stream.gen_seedgen.set_state(random_state.get_state())
                return random_stream
            @classmethod
            def __modify_schema__(cls, field_schema):
                field_schema.update(
                    type="array",
                    description="('RandomStateStream', RandomState)",
                    items=[{'type': 'string'},
                           mtb.typing.RandomState__modify_schema__({})]
                )
            @classmethod
            def json_encoder(cls, v):
                return ("RandomStateStream", v.gen_seedgen.get_state())

        __validation_types['RandomStateStream'] = RandomStateStream

        mtb.typing.add_json_encoder(
            shim.config.SymbolicNumpyRNGType,
            RandomStateStream.json_encoder)

    if 'RNGStream' not in __validation_types:
        import theano
        class RNGStream:
            # The MRG RandomStream is a RNG which produces seeds for the RNGs
            # of each Op. The MRG RandomStream's own state is a plain
            # 6-d numpy array stored as the `rstate` attribute.
            # As long as we restore the MRG state, we
            # will recreate subsequent ops with the same seeds.
            @classmethod
            def __get_validators__(cls):
                yield cls.validate
            @classmethod
            def validate(cls, value, field):
                # WARNING: I haven't thoroughly tested that state is fully restored
                if isinstance(value, shim.config.SymbolicMRGRNGType):
                    return value
                elif isinstance(value, shim.config.SymbolicNumpyRNGType):
                    raise TypeError(
                        "Expected an MRG RandomStream (theano.sandbox.rng_mrg.MRG_RandomStream); "
                        "Received a NumPy RandomStream (theano.tensor.random.utils.RandomStream).")
                elif json_like(value, "RNGStream"):
                    mrg_state = mtb.typing.Array[np.integer,1].validate(value[1], field).astype(np.int32)
                elif isinstance(value, int):
                    mrg_state = value
                elif isinstance(value, np.ndarray):
                    mrg_state = mtb.typing.Array[np.integer,1].validate(value[1], field).astype(np.int32)
                else:
                    raise ValueError("Unable to deserialize RandomStateStream; "
                                     f"received: {value}.")
                return theano.sandbox.rng_mrg.MRG_RandomStream(mrg_state)
            @classmethod
            def __modify_schema__(cls, field_schema):
                field_schema.update(
                    type="array",
                    description="('RNGStream', array[6 x int])",
                    items=[{'type': 'string'},
                           mtb.typing.Array[np.int32,1].__modify_schema__({})]
                )
            @classmethod
            def json_encoder(cls, v):
                return ("RNGStream", v.rstate)

        __validation_types['RNGStream'] = RNGStream

        mtb.typing.add_json_encoder(
            shim.config.SymbolicMRGRNGType,
            RNGStream.json_encoder)

    AnyRNG = Union[__validation_types['RNGStream'],
                   __validation_types['RandomStateStream'],
                   mtb.typing.RandomState,
                   mtb.typing.RNGenerator]

mtb.typing.add_numerical_type(
    (Symbolic[np.number],   Shared[np.number]))
mtb.typing.add_scalar_type(
    (Symbolic[np.number,0], Shared[np.number,0]))
if isinstance(shim.config.SymbolicSharedType, type):
    mtb.typing.add_json_encoder(
        shim.config.SymbolicSharedType, lambda x: x.get_value())
