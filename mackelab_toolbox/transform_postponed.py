from mackelab_toolbox.transform import *
AnyNumericalType = mtb.typing.AnyNumericalType  # Dynamic types have been frozen

@generic_pydantic_initializer  # Allow initialization with dict, json, instance
class TransformedVar(pydantic.BaseModel):
    """
    In order to serve as an input normalizer, the constructor attempts to
    recognize when arguments in fact describe a NonTransformedVar. This works
    in two cases:
        - When initializing with a single argument (json str | dict |instance)
        - When initializing without a 'bijection' keyword argument. (Which is
          required for TransformedVar, but redundant for NonTransformedVar.)

    Side-effects
    ----------
    If `orig` or `new` are symbolic, their `name` attribute is set to match
    the transform's names.

    Parameters
    ----------
    bijection: Bijection
        May also be a bijection descriptor (str | json str | dict)
    names: TransformNames (Optional)
        May also be a TransformNames descriptor (str | json str | dict)
        If omitted, names are inferred from `bijection`.
        Names are used to label the original and transformed variables.
        If these variables have a `name` attribute, it is set correspondingly.
        This parameter may be used to override the names defined by `bijection`.
    orig, new: (symbolic) scalar or array
        Exactly one of `orig`, `new` must be provided. It may be symbolic.
    """

    bijection : Bijection
    names     : Optional[TransformNames]
    orig      : AnyNumericalType
    new       : AnyNumericalType

    class Config:
        json_encoders = mtb.typing.json_encoders

    # ---------
    # Validators

    @classmethod  # Called in set_orig_new
    def construct_bijection(cls, bij):
        return Bijection(bij)

    @classmethod  # Called in set_orig_new
    def construct_names(cls, names, values):
        bij = values.get('bijection', None)
        if names is None and bij is not None:
            return TransformNames(orig=bij.map.xname,
                                  new=bij.inverse_map.xname)
        else:
            return TransformNames(names)  # No-op if v is a TransformNames

    @root_validator(pre=True)
    def set_orig_new(cls, values):
        orig, new, bij, names = (values.get(x, None) for x in
            ('orig', 'new', 'bijection', 'name'))
        if bij is None:
            return  # Abort
        # TODO: Any way to get the other validators to execute before root_validator?
        bij = cls.construct_bijection(bij)
        values.update(bijection=bij)
        names = cls.construct_names(names, values)
        values.update(names=names)

        # Check that we have exactly one of orig, new
        if not( (orig is None) != (new is None) ):  #xor
            raise ValueError("Exactly one of `orig`, `new` must be specified.")
        # Apply bijection to construct the other orig or new
        if orig is not None:
            new = bij.map(orig)
        elif new is not None:
            orig = bij.inverse_map(new)
        # Add `name` attribute to variables if possible
        if names is not None:
            if hasattr(orig, 'name'):
                if orig.name is None:
                    orig.name = names.orig
                elif orig.name != names.orig:
                    raise ValueError(f"`orig` variable name ({orig.name}) does "
                                     f"not match that of `names` ({names.orig}).")
            if hasattr(new, 'name'):
                if new.name is None:
                    new.name = names.new
                elif new.name != names.new:
                    raise ValueError(f"`new` variable name ({new.name}) does "
                                     f"not match that of `names` ({names.new}).")
        values['orig'] = orig
        values['new']  = new
        return values

    # -----------
    # Redirection to NonTransformedVar
    # There are two ways we try to recognize a NonTransformedVar signature:
    #   - A single argument initialization was attempted and failed
    #     -> If this happens, the `desc` is not None.
    #   - `bijection` is not present in keyword arguments

    def __new__(cls, desc=None, **kwargs):
        if desc is not None or 'bijection' not in kwargs:
            try:
                obj = NonTransformedVar(desc, **kwargs)
                # Don't need to call __new__: NonTransformedVar is a ≠ class,
                # so won't trigger __init__. Might as well use normal initializer
            except (ValidationError, ValueError, TypeError, AssertionError) as e:
                raise e
                raise ValueError(f"Arguments ({utils.argstr((desc,), kwargs)}) "
                                 "are invalid for both TransformedVar and "
                                 "NonTransformedVar.\n\n"
                                 f"Caught exception:\n-----------------\n{e}.")
        else:
            obj = super().__new__(cls)
        return obj

    # -----------
    # Interface
    # Some of this is older, but can still be more concise

    def __str__(self):
        return f"{self.names.orig} -> {self.names.new} ({self.to})"

    def rename(self, orig, new):
        """
        Rename the variables

        Parameters
        ----------
        new: str
            Name to assign to self.new
        orig: str
            Name to assign to self.orig
        """
        self.names = TransformNames(orig=orig, new=new)
        if hasattr(self.orig, 'name'):
            self.orig.name = orig
        if hasattr(self.new, 'name'):
            self.new.name = new

    @property
    def to(self):
        return self.bijection.map
    @property
    def back(self):
        return self.bijection.inverse_map


@generic_pydantic_initializer    # Allow init with json, instance, dict
class NonTransformedVar(pydantic.BaseModel):
    """Provides an interface consistent with TransformedVar.

    Note
    ----
        It should always be possible to instantiate with `TransformVar`; the
        constructor will recognize the invalid signature and redirect to
        `NonTransformedVar`. The only reason to instantiate `NonTransformedVar`
        are:
            1. To ensure/document that the returned object absolutely cannot
               be a transformed variable.
            2. To avoid the computational cost of a few validation failures.

    Parameters
    ----------
    bijection: Bijection or Transform  (Optional)
        Or valid descriptors (str | json str | dict) for Bijection or Transform.
        Used only to extract variable names; if provided, they must match
        `new`. (So a Bijection must have the same `xname` for both the map and
        inverse map.) \
        Explicitly passing this parameter is discouraged, since that is one
        mean by which NonTransformedVar signatures are distinguished from those
        of TransformedVar.
    names: TransformNames | str
        If TransformNames, `names.orig` and `names.new` must be the same.
    orig, new: (symbolic) scalar or array
        Exactly one of `orig`, `new` must be provided. It may be symbolic.
        They are exactly equivalent; `new` is provided only for consistency
        with TransformedVar.
    """
    bijection : ClassVar[Bijection] = Bijection("x -> x ; x -> x")
    names     : TransformNames
    orig      : AnyNumericalType
    new       : Optional[AnyNumericalType]
        # Only for consistency with TransformedVar

    class Config:
        json_encoders = mtb.typing.json_encoders

    # ----------
    # Validators and initializer

    @validator('names', pre=True)
    def construct_names(cls, names):
        names = TransformNames(names)
        if names.orig != names.new:
            raise ValueError("'orig' and 'new' names don't match (they are "
                             f"'{names.orig}' and '{names.new}'")
        return names

    @root_validator
    def set_orig_new(cls, values):
        orig, new, names = (values.get(x, None) for x in
            ('orig', 'new', 'names'))
        # Check that we have exactly one of orig, new
        if not( (orig is None) != (new is None) ):  #xor
            raise ValueError("Exactly one of `orig`, `new` must be specified.")
        if orig is None:
            orig = new
        else:
            new = orig
        if hasattr(orig, 'name'):
            if names is not None:
                if orig.name is None:
                    orig.name = names.orig
                elif orig.name != names.orig:
                    raise ValueError(f"`orig|new` variable name ({orig.name}) does "
                                     f"not match that of `names` ({names.orig}).")
                assert new.name == orig.name
        values['orig'] = orig
        values['new']  = new
        return values

    def __init__(self, desc=None, *, bijection=None, **kwargs):
        # We don't want `bijection` in the schema, but for compability with
        # TransformedVar it should be in the signature
        names = kwargs.get('names', None)
        if desc is not None:
            if 'orig' in kwargs or 'new' in kwargs:
                raise TypeError("When instantiating NonTransformedVar with "
                                "positional arguments, the `orig` and `new` "
                                "keyword arguments are redundant.")
            # If `desc` were a NonTransformedVar, it would already have been
            # caught by the @generic_pydantic_initializer decorator.
            # However it could also be a bare numeric type.
            kwargs['orig'] = desc
        if names is None and bijection is not None:
            try:
                φ = Transform(bijection)
                oname = φ.xname
                nname = φ.xname
            except (ValidationError, ValueError):
                bij = Bijection(bijection)
                oname = bij.map.xname
                nname = bij.inverse_map.xname
            kwargs['names'] = TransformNames(orig=oname, new=nname)
        super().__init__(**kwargs)

    # --------
    # Accessor methods

    @property
    def to(self):
        return self.bijection.map
    @property
    def back(self):
        return self.bijection.inverse_map

    def rename(self, orig, new=None):
        """
        Rename the variable

        Parameters
        ----------
        orig: str
            Name to assign to self.orig
        new: str
            Ignored; only provided to have consistent API with TransformedVar.
            If given, must be equal to `orig`.
        """
        if new is not None and new != orig:
            raise ValueError("For NonTransformedVar, the 'new' and 'orig' names must match.")
        self.names = TransformNames(orig=orig, new=orig)
        if hasattr(self.orig, 'name'):
            self.orig.name = orig
