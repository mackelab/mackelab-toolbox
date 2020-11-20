"""
Functionality that used to be included in `mackelab_toolbox.parameters`.
"""

def _param_diff(params1, params2, name1="", name2=""):
    KeyDiff = namedtuple('KeyDiff', ['name1', 'name2', 'keys'])
    NestingDiff = namedtuple('NestingDiff', ['key', 'name'])
    TypeDiff = namedtuple('TypeDiff', ['key', 'name1', 'name2'])
    ValueDiff = namedtuple('ValueDiff', ['key', 'name1', 'name2'])

    diff_types = {'keys': KeyDiff,
                  'nesting': NestingDiff,
                  'type': TypeDiff,
                  'value': ValueDiff}
    diffs = {key: set() for key in diff_types.keys()}
    keys1, keys2 = set(params1.keys()), set(params2.keys())
    if keys1 != keys2:
        diffs['keys'].add( KeyDiff(name1, name2, frozenset(keys1.symmetric_difference(keys2))) )
    def diff_vals(val1, val2):
        if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
            return (val1 != val2).any()
        else:
            return val1 != val2
    for key in keys1.intersection(keys2):
        if isinstance(params1[key], ParameterSetBase):
            if not isinstance(params2[key], ParameterSetBase):
                diffs['nesting'].add((key, name1))
            else:
                for diffkey, diffval in _param_diff(params1[key], params2[key],
                                                    name1 + "." + key, name2 + "." + key).items():
                    # Prepend key to all nested values
                    if hasattr(diff_types[diffkey], 'key'):
                        diffval = {val._replace(key = key+"."+val.key) for val in diffval}
                    if hasattr(diff_types[diffkey], 'keys') and len(diffval) > 0:
                        iter_type = type(next(iter(diffval)).keys)  # Assumes all key iterables have same type
                        diffval = {val._replace(keys = iter_type(key+"."+valkey for valkey in val.keys))
                                   for val in diffval}
                    # Update differences dictionary with the nested differences
                    diffs[diffkey].update(diffval)
        elif isinstance(params2[key], ParameterSetBase):
            diffs['nesting'].add(NestingDiff(key, name2))
        elif type(params1[key]) != type(params2[key]):
            diffs['type'].add(TypeDiff(key, name1, name2))
        elif diff_vals(params1[key], params2[key]):
            diffs['value'].add(ValueDiff(key, name1, name2))

    return diffs

def param_diff(params1, params2, name1="", name2=""):
    print("Bug warning: current implementation does not catch some differences "
          "in shape because of broadcasting (e.g. [[1 2], [1, 2]] vs [1, 2]).")
    if name1 == "":
        name1 = "params1" if name2 != "params1" else "params1_1"
    if name2 == "":
        name2 = "params2" if name1 != "params2" else "params2_2"
    diffs = _param_diff(params1, params2, name1, name2)
    for diff in diffs['keys']:
        name1, name2, keys = diff
        for key in keys:
            print("The parameter sets {} and {} do not share the following keys: {}"
                  .format(name1, name2, key))
    for diff in diffs['nesting']:
        print("Key '{}' is parameter set only for {}.".format(*diff))
    for diff in diffs['type']:
        key, name1, name2 = diff
        print("The following values have different type:\n"
              "  {}.{}: {} ({})\n  {}.{}: {} ({})"
              .format(name1, key, params1[key], type(params1[key]),
                      name2, key, params2[key], type(params2[key])))
    for diff in diffs['value']:
        key, name1, name2 = diff
        print("The parameter sets {} and {} differ on key {}.\n"
                  "  {}.{}: {}\n  {}.{}: {}"
              .format(name1, name2, key,
                      name1, key, params1[key],
                      name2, key, params2[key]))

###########################
# Parameter file expansion
# USE PARAMETERRANGE INSTEAD !
###########################

ExpandResult = namedtuple("ExpandResult", ['strs', 'done'])

def expand_params(param_str, fail_on_unexpanded=False, parser=None):
    """
    Expand a parameter description into multiple descriptions.
    The default parser expands contents of the form "*[a, b, ...]" into multiple
    files with the starred expression replaced by "a", "b", ….

    The default parser expands on '*' and recognizes '()', '[]' and '{}' as
    brackets. This can be changed by explicitly passing a custom parser.
    The easiest way to obtain a custom parser is to instantiate one with
    Parser, and change its 'open_brackets', 'close_brackets', 'separators' and
    'expanders' attributes.

    NOTE: The *parameters* package already provides `ParameterRange` and
    `ParameterSpace` objects, which provide the same functionality and should
    probably be used instead.

    Parameters
    ----------
    param_str: str
        The string description for the parameters.
    fail_on_unexpanded: bool (default False)
        (Optional) Specify whether to fail when an expansion character is found
        but unable to be expanded. By default such an error is ignored, but
        if your parameter format allows it, consider setting it to True to
        catch formatting errors earlier.
    parser: object
        (Optional) Only required if one wishes to replace the default parser.
        The passed object must provide an 'extract_blocks' method, which itself
        returns a dictionary of Parser.Block elements.

    Returns
    -------
    list of strings
        Each element is a complete string description for oneset of parameters.
    """

    if parser is None:
        parser = Parser()
    param_str = expand_urls(param_str)
        # TODO: Expanding urls as we expand blocks, rather than all
        #       at once at the beginning, would allow to apply expansion
        #       to parameter sets.
    param_strs = [strip_comments(param_str)]
    done = False
    while not done:
        res_lst = [_expand(s, fail_on_unexpanded, parser) for s in param_strs]
        assert(isinstance(res.strs, list) for res in res_lst)
        param_strs = [s for res in res_lst for s in res.strs]
        done = all(res.done for res in res_lst)

    return param_strs

def expand_param_file(param_path, output_path,
                      fail_on_unexpanded=False, parser=None,
                      max_files=1000):
    """
    Load the file located at 'param_path' and call expand_params on its contents.
    The resulting files are saved at the location specified by 'output_pathn',
    with a number appended to each's filename to make it unique.

    NOTE: The *parameters* package already provides `ParameterRange` and
    `ParameterSpace` objects, which provide the same functionality and should
    probably be used instead.

    Parameters
    ----------
    param_path: str
        Path to a parameter file.
    output_path: str
        Path to which the expanded parameter files will be saved. If this is
        'path/to/file.ext', then each will be saved as 'path/to/file_1.ext',
        'path/to/file_2.ext', etc.
    fail_on_unexpanded: bool (default False)
        (Optional) Specify whether to fail when an expansion character is found
        but unable to be expanded. By default such an error is ignored, but
        if your parameter format allows it, consider setting it to True to
        catch formatting errors earlier.
    parser: object
        (Optional) Only required if one wishes to replace the default parser.
        The passed object must provide an 'extract_blocks' method, which itself
        returns a dictionary of Parser.Block elements.
    max_files: int
        (Optional) Passed to iotools.get_free_file. Default is 1000.

    Returns
    -------
    None
    """
    with open(param_path, 'r') as f:
        src_str = f.read()

    pathnames = []
    for ps in expand_params(src_str, fail_on_unexpanded, parser):
        f, pathname = iotools.get_free_file(output_path, bytes=False,
                                            force_suffix=True,
                                            max_files=max_files)
        f.write(ps)
        f.close()
        pathnames.append(pathname)

    #print("Parameter files were written to " + ', '.join(pathnames))
        # TODO: Use logging
    return pathnames

def expand_urls(s):
    """
    Expand url specifications following the ParameterSet format.
    This operation is performed on the raw string, so the string does not
    need to be a valid ParameterSet definition.
    This also means that no validation is performed (e.g. urls in keys will
    be blindly expanded rather throw an error).
    """
    start = 0
    while True:
        # Remove comments at every iteration, since every time we expand a
        # url it may contain comments. In particular, we want to avoid expanding
        # commented out urls.
        s = strip_comments(s)
        start = s.find('url(', start)
        if start >= 0:
            stop = s.find(')', start)
            if stop is None:
                raise SyntaxError("`url(` has no matching closing parenthesis.")
            url = s[start+4:stop].strip("'\"")
            with open(url) as f:
                substr = strip_comments(f.read())
            s = s[:start] + substr + s[stop+1:]
        else:
            break
    return s

def _expand(s, fail_on_unexpanded, parser):
    # TODO: Allow multicharacter expansion tokens. Then we can use
    # this to exand `url()`
    blocks = parser.extract_blocks(s)
    for i, c in enumerate(s):
        if c in parser.expanders:
            if i+1 in blocks:
                block = blocks[i+1]
                expanded_str = [s[:i] + str(el) + s[block.stop:]
                                for el in block.elements.values()]
                # Return list of expanded strings and continuation flag
                return ExpandResult(expanded_str, False)
            elif fail_on_unexpanded:
                raise ValueError("Expansion identifier '*' at position {} "
                                 "must be followed by a bracketed expression.\n"
                                 "Context: '{}'."
                                 .format(i, s[max(i-10,0):i+10]))
    # Found nothing to expand; return the given string and termination flag
    # Wrap the string in a list to match the expected format
    return ExpandResult([s], True)

class Parser():
    """Basic parser for nested structures with opening and closing brackets."""

    # Default values.
    # These can be changed on a per instance by reassigning the attributes
    open_brackets = ['[', '{']
    close_brackets = [']', '}']  # Order must match that of open_brackets
        # Parentheses are not included because "*(…)" tends to appear in
        # mathematical expressions
    separators = [',']
    expanders = ['*']

    def get_closer(self, c):
        idx = self.open_brackets.index(c)
        if idx != -1:
            return self.close_brackets[idx]
    def get_opener(self, c):
        idx = self.close_brackets.index(c)
        if idx != -1:
            return self.open_brackets[idx]

    def extract_blocks(self, s):
        block_stack = deque()  # Unclosed blocks
        blocks = {}       # Closed blocks, keyed by their starting index
        for i, c in enumerate(s):
            if c in self.open_brackets:
                block = Block(start=i, opener=c, closer=self.get_closer(c))
                if len(block_stack) > 0:
                    block_stack[-1].blocks.append(block)
                block_stack.append(block)
            elif c in self.close_brackets:
                block = block_stack[-1]
                if len(block_stack) == 0:
                    raise ValueError("Unmatched closing bracket '{}' at position {}."
                                     .format(c, i))
                if c != block_stack[-1].closer:
                    raise ValueError("Closing bracket '{}' at position {} does not "
                                     "match opening bracket '{}' at position {}."
                                     .format(c, i, block.opener,
                                             block.start))
                block.stop = i+1

                # TODO: make this method of the Elements object or something
                el_start_i = list(block.elements.keys())[-1]
                el_stop_i = i
                block.elements[el_start_i] = s[el_start_i:el_stop_i]

                blocks[block.start] = block_stack.pop()
            elif c in self.separators:
                block = block_stack[-1]

                el_start_i = list(block.elements.keys())[-1]
                el_stop_i = i
                block.elements[el_start_i] = s[el_start_i:el_stop_i]

                block.elements[el_stop_i+1] = None

        if len(block_stack) > 0:
            raise ValueError("Unmatched opening bracket '{}' at position {}."
                             .format(block_stack[-1].opener,
                                     block_stack[-1].start))

        return blocks

            # A dictionary of the elements separated by one of the 'separators'
            # The key is the index of the first character

class Block(SimpleNamespace):
    def __init__(self, start, opener, closer):
        super().__init__()
        self.start = start
        self.stop = None
        self.opener = opener
        self.closer = closer
        self.elements = OrderedDict([(start+1, None)])
        self.blocks = []

#########################
# Parameter parsing / validation
# USE PYDANTIC INSTEAD !
#########################

# TODO: Make schema an abstractmethod
#       (currently fails because abc forces class to be hashed ?)
import abc
# from parameters import ParameterSet
from parameters.validators import SchemaBase, ParameterSchema, CongruencyValidator
from inspect import signature, Parameter

class Any(SchemaBase):
    """
    To be used as a value in a `ParameterSchema`. Validates the same-path
    `ParameterSet` value if it matches any of the provided value schemas.

    NOTE: Much of this functionality could probably be achieved by building of
    the `dataclass` class decorator introduced in Python 3.7.

    Examples
    --------
    >>> from parameters.validators import ParameterSchema, SubClass
    >>> from mackelab_toolbox.parameters import Any
    >>> schema = ParameterSchema({'a': Any(SubClass(int), SubClass(float))})
    """
    def __init__(self, *value_schemas):
        self.schemas = value_schemas

    def validate(self, leaf):
        return any(s.validate(leaf) for s in self.schemas)

    def __repr__(self):
        cls = self.__class__
        schemas = ', '.join([repr(s) for s in self.schemas])
        return ('.'.join([cls.__module__, cls.__name__])
                +'(schemas={})'.format(schemas))

    def __eq__(self, x):
        if isinstance(x, Any) and len(self.schemas) == len(x.schemas):
            equal = True
            for s in self.schemas:
                if not equal:
                    break
                for sx in x.schemas:
                    if s == sx:
                        break
                equal = False
            return equal
        else:
            return False

class Identity(SchemaBase):
    """
    To be used as a value in a `ParameterSchema`.  Validates the same-path
    `ParameterSet` is identical (an `is` comparison).
    Useful for sentinal values  like `None`.

    See also: `SchemaBase`
    """

    def __init__(self, cls=None):
        self.cls = cls

    def validate(self, leaf):
        return leaf is self.cls

    def __repr__(self):
        cls = self.__class__
        return '.'.join([cls.__module__, cls.__name__])+'(id=%s)' % (repr(self.cls),)

    def __eq__(self, x):
        if isinstance(x, Identity):
            return self.cls is x.cls
        else:
            return False

# TODO: subclass ParameterSchema
class ParameterSpec:
    """
    A ParameterSet constructor based on a ParameterSchema

    Schemas should be specified as a dictionary; they are are automatically cast
    to ParameterSchema at initialization. For nested schemas this is required.
    A nested schemas shoud specify the `ParameterSpec` subclass, not that
    spec's `schema` attribute. I.e. use
        schema = {'x': 1, 'y': Y.ParameterSpec}
    not
        schema = {'x': 1, 'y': Y.ParameterSpec.schema}


    Current limitation
    ------------------
    This is untested on recursive key names of the form
        schema = {'x.y' = 1}
    """
    parser = None  # Overwrite in derived class to implement parser
    class InvalidParser(RuntimeError):
        pass
    def __init__(self):
        """

        At the end of initialization, tests well-formedness by calling `self.validate(self)`.

        **Note**: `parse` will not be called if `args` and `kwargs` are empty,
        only ClassSpecs of this type (args) or only correspond to Spec
        attributes (kwargs).
        """
        # We need two versions of schema to deal with nested ParameterSpec.
        #   - The original `schema` needs to store ParameterSpec (not
        #     ParameterSpec.schema) in order to properly cast recursively.
        #   - The ParameterSchema used by the validator needs the `.schema`,
        #     otherwise it tests against `type(ParameterSpec)`.
        # Leaving `schema` untouched also avoids confusion for users
        self._schema = self.schema.copy()
        for k, v in self.schema.items():
            if isinstance(v, type) and issubclass(v, ParameterSpec):
                self._schema[k] = v.schema
        if not isinstance(self._schema, ParameterSchema):
            self._schema = ParameterSchema(self._schema)

        self._validator = CongruencyValidator()

    def __call__(self, *args, **kwargs):
        pset = self._parse(*args, **kwargs)
        if not self.validate(pset):
            raise self.InvalidDescription(
                "Provided descritor is incompatible with specification '{}'"
                .format(type(self).__qualname__))
        return pset

    def __str__(self):
        return str(self.schema)
    def __repr__(self):
        return repr(self.schema)
    def pretty(self, *args, **kwargs):
        return self.schema.pretty(*args, **kwargs)
    def _repr_html_(self):
        """
        Defaults:
            '-' indicates a mandatory argument.
            '?' indicates that we weren't able to compute a default, because
                the defaults() method requires some parameters to be set.
        """
        try:
            d = self.defaults()
        except TypeError:
            # Defaults invalid without args
            d = {θ: "?" for θ in self.keys()}
        s0 = "<table><tr><th>Parameter</th><th>Types</th><th>Default</th></tr>"
        s = [f"<tr><td>{θ}</td><td>{t}</td><td>{d.get(θ, '-')}</td></tr>"
             for θ, t in self.items()]
        sn = "</table>"
        return "\n".join([s0] + s + [sn])

    @staticmethod
    def defaults(**kwargs):
        """
        Redefine in derived class to provide defaults.

        Parameters
        ----------
        **kwargs:
            The provided parameter values. This is to allow
            defaults to depend on other parameters (e.g. one parameter
            may specify the number of dimensions).
            All positional arguments are converted to keywords before calling
            :def:defaults.
        """
        return {}

    # def __repr__(self):
    #     return {attr: getattr(self, attr) for attr in self.attrs}

    def flat(self):
        __doc__ = ParameterSet.flat.__doc__
        return ParameterSet(dict(self.items())).flat()

    def _parse(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs:
            Arguments as defined by `self.schema`

        Returns
        -------
        ParameterSet
        """
        pset = ParameterSet({})  # What we return at the end
        arglist = list(args)
        # Initialize attributes with those of passed Specs, if any
        for arg in args:
            if isinstance(arg, type(self)):
                for attr in self._schema:
                    setattr(pset, attr, getattr(arg, attr))
                arglist.pop(arg)
        # Kind of hacky way of avoid to overwrite parser keyword args
        if self.parse is None:
            parseargs = []
        else:
            # This builds a list of arguments which can be passed as keywords
            # to `parse`, to avoid removing them in the next step.
            # Any argument which can be set by keyword is removed; testing
            # `kinds` avoids adding *args and **kwargs to the list.
            parseargs = [name for name, param in signature(self.parse).parameters.items()
                         if param.kind in (Parameter.POSITIONAL_OR_KEYWORD,
                                           Parameter.KEYWORD_ONLY)]
        # Replace positional arguments by keyword ones
        allkwargs = {nm: arg for arg, nm in zip(arglist, self._schema)}
        if set(allkwargs).intersection(kwargs) != set():
            raise TypeError("Argument {} given by name and position".format(nm))
        allkwargs.update(kwargs)
        defaults = self.defaults(**allkwargs)
        # Replace arguments by their default values
        for k in self._schema:
            if k not in allkwargs:
                if k not in defaults:
                    raise TypeError("Required argument {} not found.".format(k))
                else:
                    allkwargs[k] = defaults[k]
        # Now extract any explicit keyword args. This allows simple parsers
        # to be implemented with no `parse` method at all.
        _allkwargs = allkwargs.copy()
        kwattrs = {}
        for k in _allkwargs:
            if k in self._schema and k not in parseargs:
                kwattrs[k] = allkwargs.pop(k)

        # If there are remaining args or kwargs, pass them to parse
        if len(allkwargs) > 0:
            if self.parse is None:
                raise ValueError(
                    "Parser {} does not implement a `parse` method and {} "
                    "are unrecognized keywords."
                    .format(type(self).__qualname__, list(allkwargs.keys())))
            else:
                pset.update([], **self.parse(**allkwargs))
        # Finally set the explicitly passed attributes
        # At the same time, recursively cast any attribute of type ParameterSpec
        for k, v in kwattrs.items():
            t = self._schema[k]
            # TODO?: Casts for other types ?
            if isinstance(t, type) and issubclass(t, ParameterSpec):
                v = t(v)
            setattr(pset, k, v)
        return pset

    def validate(self, pset):
        """Validate a parameter set against the parser's schema."""
        return self._validator.validate(pset, self._schema)

    @classmethod
    def castable(cls, *args, **kwargs):
        """
        Returns True if the arguments can be used to construct a valid
        ParameterSet.
        """
        pset = cls(*args, **kwargs)
        return pset.validate(pset)

    @classmethod
    def keys(cls):
        """
        Convenience method to access schema keys (i.e. variable names).
        """
        return cls.schema.keys()
    def items(self):
        # `keys()` makes sure to return only ParameterSet elements,
        # so use it to remove everything else
        # Not sure if schema or _schema would be better here
        for key in self.keys():
            yield key, self._schema[key]
    def values(self):
        for key in self.keys():
            yield self._schema[key]

    # If we use this abstractmethod, delete `parser = None` above
    # @abc.abstractmethod
    # def parse(self, desc):
    #     """Parse description. Returns True on success, False on failure."""
    #     return

    # @abc.abstractmethod
    # def schema(self):
    #     """
    #     Attribute names must be specified as a list of strings.
    #     Attributes must fully determine the specification.
    #     """
    #     return
