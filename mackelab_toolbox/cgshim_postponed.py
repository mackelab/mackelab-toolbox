"""
Adds the following dynamic types to mackelab_toolbox.typing:

- FloatX
- Symbolic
- Shared
- Tensor
- AnyRNG
"""

from typing import Union
from functools import lru_cache
import numpy as np
import mackelab_toolbox as mtb
import mackelab_toolbox.typing
from mackelab_toolbox.cgshim import *
import logging
logger = logging.getLogger(__name__)

json_like = mtb.typing.json_like

@lru_cache(maxsize=None)  # Ensure create_type() doesn't define the same type twice
def create_type(baseT: type, Tname: str, namedesc: str, validators: tuple=()):
    """
    Create a Pydantic-compatible type for base type `baseT`.
    This can be used as an annotation is a BaseModel, to parse arguments of
    type `baseT`.
    `baseT` is expected to similar to a NumPy value: values should define
    'ndim' and 'dtype', or expect to be casted to Array.
    
    The created type comes with a type validator, which does two things:
    
    - Check the value's dtype, and perform a cast when possible.
    - Ensures the value has a 'name' attribute (using the field name as default)
    
    Parameters
    ----------
    baseT:
        The type for which we want a type hint.
    Tname:
        The name to assign to the type ("type name"). Normally capitalized.
    namedesc:
        Name to use in error messages to describe the type. Normally not
        capitalized.
        E.g. setting `namedesc` to "symbolic", an error message may read
        "This field expects a symbolic variable."
    validators:
        Additional validators to apply to values. These are applied *after*
        the dtype validator, so they can assume that value types are correct.
    
    """
    # Structure:
    #   - Create the typed base class TypedSymbolic, a metaclass
    #     SymbolicMeta which returns a subclass of TypedSymbolic, and a
    #     Symbolic class which has SymbolicMeta as metaclass
    #     + Store Symbolic in the cache
    #   - Return the value from the cache
    # The metaclass logic here, like the one in mtb.typing.Array, is based
    # on a suggestion by the pydantic developer:
    # https://github.com/samuelcolvin/pydantic/issues/380#issuecomment-594639970
    assert isinstance(validators, tuple)

    # For symbolic types, we just assume that val has already been properly casted.
    # We do this to keep the reference to the original variable
    class ValidatingType(baseT):
        ndim_ = None
        nptype = None

        @classmethod
        def __get_validators__(cls):
            # First cast to nptype, so validators can assume dtype,
            # but only check that we have the expected type at the end,
            # so that e.g. validators can convert list to Shared
            yield from (cls.deserialize_array, cls.cast_nptype, *validators,
                        cls.ensure_name_if_possible, cls.validate_type_and_dim)
                        
        @classmethod
        def deserialize_array(cls, value):
            if json_like(value, 'Array'):
                value = mtb.typing.Array.validate(value)
            return value

        @classmethod
        def cast_nptype(cls, value, field):
            """
            - If `value` is list or tuple, convert to NumPy array
            - If `value` is symbolic, log an error if its dtype is not exactly
              as expected (we don't perform casts of symbolic values, to avoid
              breaking references).
            - If `value` is numeric and cannot be cast to the expected type,
              also log an error.
            - If `value` is numeric and can be cast to the expected type,
              perform the cast.
            """
            if isinstance(value, (list, tuple)):
                value = np.array(value)
            if cls.nptype is not None:
                target_nptype = cls.nptype
                # If nptype is an NPValue, we want _its_ nptype
                target_nptype = getattr(target_nptype, 'nptype', target_nptype)
                if (not hasattr(value, 'dtype')
                      or np.dtype(value.dtype) is not np.dtype(target_nptype)):
                    if shim.issymbolic(value):
                        logger.error(
                            f"{field.name} expects a {namedesc} variable with "
                            f"data type {target_nptype}. The "
                            f"provided symbolic value has dtype {value.dtype}. "
                            "Note that for symbolic variables, we perform no "
                            "casting, so types must match exactly.")
                    elif not np.can_cast(value, target_nptype):
                        logger.error(
                            f"{field.name} expects a {namedesc} variable with "
                            f"data type {target_nptype}. Provided value has dtype "
                            f"{value.dtype}, which is not a permitted cast. ")
                    else:
                        value = target_nptype(value)
            return value
            
        @classmethod
        def ensure_name_if_possible(cls, value, field):
            """
            If `value` does not already define a 'name' attribute (or it is
            'None'), assign the name of this field.
            
            .. Caution:: This function only makes a best effort to assign to
               (and possibly create) a 'name' attribute. However this is not
               always possible: types implemented in C (e.g. `int`, `float`,
               `numpy.ndarray`) do not accept arbitrary attributes.
               Types which define ``__slots__`` also do not accept arbitrary
               attributes.
               In these cases, the 'name' attribute is left undefined.
            """
            # NOTE: For types which don't allow setting 'name', `validate_type_and_dim`
            #       will typically raise TypeError
            if getattr(value, 'name', None) is None:
                # Pydantic appends the type to each subfield name when a
                # field uses Union[] to specify multiple types.
                # I haven't found a way to work back to the original type,
                # which arguably would be better than manipulating the string.
                name = field.name
                if name.endswith('_'+Tname):   # Python ≥3.9: removesuffix()
                    name = name[:-len(Tname)-1]
                try:
                    value.name = name
                except AttributeError:
                    # `value` does not accept arbitrary types
                    logger.debug("`ensure_name_if_possible` was enable to assign the "
                                 f"name '{name}' to variable of type {type(value)}.")
            return value
            
        @classmethod
        def validate_type_and_dim(cls, value, field):
            if not isinstance(value, baseT):
                raise TypeError(f"{field.name} expects a {namedesc} variable.")
            if cls.ndim_  is not None and value.ndim != cls.ndim_:
                raise TypeError(f"{field.name} expects a variable with "
                                f"{cls.ndim_} dimensions.")
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
                         'ndim_': ndim})

    return Meta(Tname, (ValidatingType,), {})

FloatX = mtb.typing.NPValue[shim.config.floatX]

"""
A type-hint compatible symbolic type.
The type returned depends on which symbolic library (if any) is loaded.
Types are cached to avoid recreating the same type more than once.
"""
if shim.config.library == 'numpy':
    Symbolic = mtb.typing.Array
else:
    Symbolic = create_type(shim.config.SymbolicType,
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
    Shared = create_type(shim.config.SymbolicSharedType,
                         'Shared', 'shared', validators=(make_shared,))

"""
Tensor
A Union type-hint compatible with any of Array, Symbolic or Shared.

NOTE: For shared variables does not check that data is actually a
      a tensor. Should we do this ?
"""
types = tuple(set((mtb.typing.Array, Symbolic, Shared)))
    # set to remove duplicates
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
            return ("RandomStateStream", v.gen_seedgen)

    mtb.typing.add_json_encoder(
        shim.config.SymbolicNumpyRNGType,
        RandomStateStream.json_encoder)

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

    mtb.typing.add_json_encoder(
        shim.config.SymbolicMRGRNGType,
        RNGStream.json_encoder)

    AnyRNG = Union[RNGStream,
                   RandomStateStream,
                   mtb.typing.RandomState,
                   mtb.typing.RNGenerator]

mtb.typing.add_numerical_type(
    (Symbolic[np.number],   Shared[np.number]))
mtb.typing.add_scalar_type(
    (Symbolic[np.number,0], Shared[np.number,0]))
if isinstance(shim.config.SymbolicSharedType, type):
    mtb.typing.add_json_encoder(
        shim.config.SymbolicSharedType, lambda x: x.get_value())
