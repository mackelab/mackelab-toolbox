# -*- coding: utf-8 -*-
"""
Mackelab toolbox : Parameters
Created on Wed Sep 20 13:19:53 2017
Author: Alexandre René © 2017-2020

A collection of functions for working with `ParameterSet` objects defined by
the [parameters](https://parameters.readthedocs.io) package. Uses fall mostly
two categories:

  + Using ParameterSets as consistent identifiers
    This involves normalizing value types and computing a form of hash, which
    we call a "digest".
  + Querying ParameterSets, mostly to compare and find differences between
    two sets.
"""

from collections import deque, OrderedDict, namedtuple
from collections.abc import Iterable, Callable
import builtins
import itertools
from types import SimpleNamespace
import hashlib
from numbers import Number
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__file__)

# from parameters import ParameterSet

from . import iotools
try:
    from . import smttk
    smttk_loaded = True
    from sumatra.parameters import NTParameterSet as ParameterSet
    from parameters import ParameterSet as ParameterSetBase
except (NameError, ImportError):
    smttk_loaded = False
    from parameters import ParameterSet
    ParameterSetBase = ParameterSet
from .utils import isinstance, flatten, strip_comments

##########################
# Module variables
debug_store = {}
    # Functions can store values in here to help debugging

###########################
# Making file names from parameters
###########################

# We use the string representation of arrays to compute the hash,
# so we need to make sure it's standardized. The values below
# are the NumPy defaults.
_filename_printoptions = {
    'precision': 8,
    'edgeitems': 3,
    'formatter': None,
    'infstr': 'inf',
    'linewidth': 75,
    'nanstr': 'nan',
    'suppress': False,
    'threshold': 1000,
    'floatmode': 'maxprec',
    'sign': '-',
    'legacy': False}
_new_printoptions = {'1.14': ['floatmode', 'sign', 'legacy']}
    # Lists of printoptions keywords, keyed by the NumPy version where they were introduced
    # This allows removing keywords when using an older version
_remove_whitespace_for_filenames = True
_type_compress = OrderedDict((
    (np.floating, np.float64),
    (np.integer, np.int64)
))
    # When normalizing types (currently only in `digest`), numpy types
    # matching the key (left) are converted to the type on the right.
    # First matching entry is used, so more specific types should come first.

# # HACK: Remove this once we don't need to fix old filenames
# def subinput_hack(params):
#     for key, val in params.items():
#         if key == 'input' and ('type' in val and 'theano' in params):
#             # Only input entries for spike & activity generation were changed
#             params[key] = val.params
#         elif isinstance(val, ParameterSet):
#             subinput_hack(val)

def normalize_type(value):
    """
    Apply the type conversions given by `_type_compress`. This reduces the space
    of possible types, helping make filenames more consistent.
    """
    if isinstance(value, np.ndarray):
        for cmp_dtype, conv_dtype in _type_compress.items():
            if np.issubdtype(value.dtype, cmp_dtype):
                return value.astype(conv_dtype)
    # No conversion match was found: return value unchanged
    return value

def digest(params, suffix=None, convert_to_arrays=True):
    """
    Generate a unique name by hashing a parameter file.

    ..Note:
    Parameters whose names start with '_' are ignored. This means that two
    parameter sets A, B with `A['_x'] == 1` and `B['_x'] == 2` will be
    assigned the same name.

    ..Debugging:
    If two parameter sets should give the same filename but don't, check the
    value of `debug_store['digest']['hashed_string']`. This module-wide
    stores the most recently hashed string representation of a parameter set.
    Filename hashes will be the same if and only if these string represenations
    are the same.

    Parameters
    ----------
    params: ParameterSet or iterable of ParameterSets
        Filename will be based on these parameters. Parameter keys starting with
        an underscore are ignored.
        Can also be give a list of parameter sets; the name in this case will
        depend on the order of the list.
        Could also arbitrarily nested lists of parameter sets.

    suffix: str or None
        If not None, an underscore ('_') and then the value of `suffix` are
        appended to the calculated filename

    convert_to_arrays: bool
        If true, the parameters are normalized by using the result of
        `params_to_arrays(params)` to calculate the filename.
    """
    if isinstance(params, dict):
        # TODO: Any reason this implicit conversion should throw a warning ?
        params = ParameterSet(params)
    if (not isinstance(params, (ParameterSetBase, str))
        and isinstance(params, Iterable)):
        # Get a hash for each ParameterSet, and rehash them together
        basenames = [p.digest() if hasattr(p, 'digest')
                     else digest(p, None, convert_to_arrays)
                     for p in params]
        basename = hashlib.sha1(bytes(''.join(basenames), 'utf-8')).hexdigest()
        basename += '_'

    else:
        if not isinstance(params, ParameterSetBase):
            logger.warning("'digest()' requires an instance of ParameterSet. "
                           "Performing an implicit conversion.")
            params = ParameterSet(params)
        if convert_to_arrays:
            # Standardize the parameters by converting them all to arrays
            # -> `[1, 0]` and `np.array([1, 0])` should give same file name
            params = params_to_arrays(params)
        if params == '':
            basename = ""
        else:
            if (np.__version__ < '1.14'
                and _filename_printoptions['legacy'] != '1.13'):
                logger.warning(
                    "You are running Numpy v{}. Numpy's string representation "
                    "algorithm was changed in v.1.14, meaning that computed "
                    "filenames will not be consistent with those computed on "
                    "more up-to-date systems. To ensure consistent filenames, "
                    "either update to 1.14, or set  `mackelab_toolbox.parameters._filename_printoptions['legacy']` "
                    "to '1.13'. Note that setting the 'legacy' option may not "
                    "work in all cases.".format(np.__version__))
            # Remove printoptions that are not supported in this Numpy version
            printoptions = _filename_printoptions.copy()
            for version in (v for v in _new_printoptions if v > np.__version__):
                for key in _new_printoptions[version]:
                    del printoptions[key]

            # Standardize the numpy print options, which affect output from str()
            stored_printoptions = np.get_printoptions()
            np.set_printoptions(**printoptions)

            # HACK Force dereferencing of '->' in my ParameterSet
            #      Should be innocuous for normal ParameterSets
            def dereference(paramset):
                for key in paramset:
                    paramset[key] = paramset[key]
                    if isinstance(paramset[key], ParameterSetBase):
                        dereference(paramset[key])
            dereference(params)

            # Standardize the parameters by converting them all to arrays
            # -> `[1, 0]` and `np.array([1, 0])` should give same file name
            # params = params_to_arrays(params)

            # We need a sorted dictionary of parameters, so that the hash is consistent
            # Also remove keys starting with '_'
            # Types need to be normalized, because if we save values as Python
            # plain types, this can throw away some Numpy type information.
            # To make sure filenames are consistent when we read the parameters
            # back, we use one type per Python type (1 for floats, 1 for ints)
            flat_params = params.flatten()
                # flatten avoids need to sort recursively
            sorted_params = OrderedDict()
            for key in sorted(flat_params):
                if key[0] != '_':
                    val = flat_params[key]
                    if hasattr(val, 'digest'):
                        sorted_params[key] = \
                            val.digest() if hasattr(val.digest, '__call__') \
                            else val.digest
                    else:
                        sorted_params[key] = normalize_type(val)

            # Now that the parameterset is standardized, hash its string repr
            s = repr(sorted_params)
            debug_store['digest'] = {'hashed_string': s}
            if _remove_whitespace_for_filenames:
                # Removing whitespace makes the result more reliable; e.g. between
                # v1.13 and v1.14 Numpy changed the amount of spaces between some elements
                s = ''.join(s.split())
            basename = hashlib.sha1(bytes(s, 'utf-8')).hexdigest()
            basename += '_'
            # Reset the saved print options
            np.set_printoptions(**stored_printoptions)
    if isinstance(suffix, str):
        suffix = suffix.lstrip('_')
    if suffix is None or suffix == "":
        assert(len(basename) > 1 and basename[-1] == '_')
        return basename[:-1] # Remove underscore
    elif isinstance(suffix, str):
        return basename + suffix
    elif isinstance(suffix, Iterable):
        assert(len(suffix) > 0)
        return basename + '_'.join([str(s) for s in suffix])
    else:
        return basename + str(suffix)
get_filename = digest

def params_to_arrays(params):
    """
    Recursively apply `np.array()` to all values in a ParameterSet. This allows
    arrays to be specified in files as nested lists, which are more readable.
    Also converts dictionaries to parameter sets.
    """
    # TODO: Don't erase _url attribute
    ParamType = type(params)
        # Allows to work with types derived from ParameterSet, for example Sumatra's
        # NTParameterSet
    for name, val in params.items():
        if isinstance(val, (ParameterSetBase, dict)):
            params[name] = params_to_arrays(val)
        elif (not isinstance(val, str)
            and isinstance(val, Iterable)
            and all(isinstance(v, Number) for v in flatten(val))):
                # The last condition leaves objects like ('lin', 0, 1) as-is;
                # otherwise they would be casted to a single type
            params[name] = np.array(val)
    return ParamType(params)

def params_to_lists(params):
    """
    Recursively call `tolist()` on all NumPy array values in a ParameterSet.
    This allows exporting arrays as nested lists, which are more readable
    and properly imported (array string representations drop the comma
    separating elements, which causes import to fail).
    """
    # TODO: Don't erase _url attribute
    ParamType = type(params)
        # Allows to work with types derived from ParameterSet, for example Sumatra's
        # NTParameterSet
    for name, val in params.items():
        if isinstance(val, (ParameterSetBase, dict)):
            params[name] = params_to_lists(val)
        elif isinstance(val, np.ndarray):
            params[name] = val.tolist()
    return ParamType(params)
params_to_nonarrays = params_to_lists

###########################
# Manipulating ParameterSets
###########################

def prune(params, keep, exclude=None):
    """
    Filter `params`, keeping only the names (aka keys) indicated in `keep`.
    E.g. if `keep` is 'model.N', then the returned object is a ParameterSet
    with only the contents of `params.model.N`. The root is unchanged, so
    parameters are still accessed as `params.model.N.[attr]`.

    Parameters
    ----------
    params: ParameterSet

    keep: str, or list of str
        Parameters to keep. Should correspond to keys  in `params`.

    exclude: str, or list of str
        (Optional) Same format as `keep`. Exclude these parameters, if they would have
        otherwise been kept.

    Returns
    -------
    ParameterSet
        Copy of `params`, keeping only the attributes given in `keep`.
    """
    # Normalize filters
    if isinstance(keep, str) or not isinstance(keep, Iterable):
        keepfilters = [keep]
    else:
        keepfilters = keep
    if exclude is None:
        excludefilters = []
    elif isinstance(exclude, str) or not isinstance(exclude, Iterable):
        excludefilters = [exclude]
    else:
        excludefilters = exclude

    # Create a new ParameterSet, and fill it with the elements of `params`
    newparams = ParameterSet({})
    for filter in keepfilters:
        if '.' in filter:
            filter, subfilter = filter.split('.', maxsplit=1)
        else:
            subfilter = None
        if filter not in params:
            logger.debug("Tried to filter a ParameterSet with '{}', but it "
                         "contains no such key. Filter was ignored.")
        else:
            if subfilter is None:
                if filter in newparams:
                    # This parameter name was already added – almost certainly an error
                    raise ValueError("Filter parameter '{}' overlaps with another"
                                     .format(filter))
                newparams[filter] = params[filter].copy()  # Don't touch the original data
            else:
                if filter not in newparams:
                    newparams[filter] = ParameterSet({})
                newparams[filter][subfilter] = prune(params[filter], subfilter)[subfilter]

    # Remove the excluded parameters
    for filter in excludefilters:
        if '.' in filter:
            # del does not work with nested keys
            psetkey, el = filter.rsplit('.', maxsplit=1)
            del newparams[psetkey][el]
        else:
            del newparams[filter]

    return newparams


#########################
# Parameter parsing / validation
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


###########################
# Comparing ParameterSets
###########################

# Three main functions, which should be merged at some point.
#   + param_diff
#         Older function. A lot of manipulations, apparently buggy.
#         Simply prints the differences, which is often not ideal.
#         Should probably be scrapped.
#   + ParameterComparison
#         Actually a class.
#         Only method which can take records, or more than two sets
#         I always forget what its custom methods are.
#   + dfdiff
#         Newer, more straightforward
#         Easier to use because it returns a standard Pandas DataFrame

def dfdiff(pset1, pset2, name1='pset 1', name2='pset 2'):
    """
    Uses `pset1`'s `diff` method to compare `pset1` and `pset2`.
    Falls back to `pset2`'s method if `pset1` doesn't have one, and to
    `NTParameterSet.diff` if neither pset does.
    """

    if not hasattr(ParameterSet, 'diff'):
        raise RuntimeError(
            "`dfdiff` requires Sumatra's NTParameterSet. Make sure "
            "you can load `import mackelab_toolbox.smttk`.")

    pset1 = params_to_lists(pset1)  # params_to_lists returns same type as pset
    pset2 = params_to_lists(pset2)
    if not hasattr(pset1, 'diff'):
        if hasattr(pset2, 'diff'):
            # Use pset2's diff method, since pset1 doesn't have one
            pset1, pset2 = pset2, pset1
        else:
            # Cast to a type which has a diff method
            pset1 = ParameterSet(pset1)
    diff = pset1.diff(pset2)
    return psets_to_dataframe(**{name1:diff[0], name2:diff[1]})

def psets_to_dataframe(*args, **psets):
    from itertools import count, chain

    # Add unlabelled parameter sets to psets with default names
    if len(args) > 0:
        psets = psets.copy()
        newpsets = {'pset'+str(i): p for i, p in zip(count(1), args)}
        if len(set(newpsets).intersection(psets)) != 0:
            raise ValueError("Don't use 'pset' to label a parameter set, or pass "
                             "all parameter sets as keywords to avoid label "
                             "clashes.")
        psets.update(newpsets)

    d = {lbl: {tuple(k.split('.')) : v for k, v in ParameterSet(pi).flatten().items()}
     for lbl, pi in psets.items()}
    # Make sure all keys have same length, otherwise we get all NaNs (ind -> 'inner dict')
    indlens = (len(ik) for ik in
               chain.from_iterable(ind.keys() for ind in d.values()))
    klen = max(chain([0], indlens))
    if klen == 0:
        d = {}
    else:
        d = {ok: {k + ('–',)*(klen-len(k)): v for k, v in od.items()}
             for ok, od in d.items()}

    return pd.DataFrame(d)

ParamRec = namedtuple('ParamRec', ['label', 'parameters'])
    # Data structure for associating a name to a parameter set

class ParameterComparison:
    """
    Example usage:
        testparams = ParameterSet("path/to/file")
        records = mackelab_toolbox.smttk.get_records('project').list
        cmp = ParameterComparison([testparams] + records, ['test params'])
        cmp.dataframe(depth=3)
    """
    def __init__(self, params, labels=None):
        """
        Parameters
        ----------
        params: iterable of ParameterSet's or Sumatra records
        labels: list or tuple of strings
            Names for the elements of `params` which are parameter sets. Records
            don't need a specified name since we use their label.
        """
        self.records = make_paramrecs(params, labels)
        self.comparison = structure_keys(get_differing_keys(self.records))

    def _get_colnames(self, depth=1, param_summary=None):
        if param_summary is None:
            param_summary = self.comparison
        if depth == 1:
            colnames = list(param_summary.keys())
        else:
            nonnested_colnames = [key for key, subkeys in param_summary.items() if subkeys is None]
            nested_colnames = itertools.chain(*[ [key+"."+colname for colname in self._get_colnames(depth-1, param_summary[key])]
                                                 for key, subkeys in param_summary.items() if subkeys is not None])
            colnames = nonnested_colnames + list(nested_colnames)
        return colnames

    def _display_param(self, record, name):
        if self.comparison[name] is None:
            try:
                display_value = record.parameters[name]
            except KeyError:
                display_value = "–"
        else:
            display_value = "<+>"
        return display_value

    def dataframe(self, depth=1, maxcols=50):
        """
        Remark
        ------
        Changes the value of pandas.options.display.max_columns
        (to ensure all parameter keys are shown)
        """
        colnames = self._get_colnames(depth)
        columns = [ [self._display_param(rec, name) for name in colnames]
                    for rec in self.records ]
        pd.options.display.max_columns = max(len(colnames), maxcols)
        return pd.DataFrame(data=columns,
             index=[rec.label for rec in self.records],
             columns=colnames)

def make_paramrecs(params, labels=None):
    """
    Parameters
    ----------
    params: iterable of ParameterSet's or sumatra Records
    labels: list or tuple of strings
        Names for the elements of `params` which are parameter sets. Records
        don't need a specified name since we use the label.
    """
    if labels is None:
        labels = []
    i = 0
    recs = []
    for p in params:
        if smttk_loaded and isinstance(p, (smttk.sumatra.records.Record, smttk.RecordView)):
            recs.append(ParamRec(p.label, p.parameters))
        else:
            assert(isinstance(p, ParameterSet))
            if i >= len(labels):
                raise ValueError("A label must be given for each element of "
                                 "`params` which is not a Sumatra record.")
            recs.append(ParamRec(labels[i], p))
            i += 1
    assert(i == len(labels)) # Check that we used all names
    return recs

def _isndarray(a):
    """
    Test if a is an Numpy array, without having to import Numpy.
    Will also return True if `a` looks like an array. (Specifically,
    if it implements the `all` and `any` methods.)
    """
    # A selection of attributes sufficiently specific
    # for us to treat `a` as a Numpy array.
    array_attrs = {'all', 'any'}
    return array_attrs.issubset(dir(a))

def _dict_diff(a, b):
    """
    Ported from sumatra.parameters to allow comparing arrays and lists.
    """
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    intersection = a_keys.intersection(b_keys)
    difference1 = a_keys.difference(b_keys)
    difference2 = b_keys.difference(a_keys)
    result1 = dict([(key, a[key]) for key in difference1])
    result2 = dict([(key, b[key]) for key in difference2])
    # Now need to check values for intersection....
    for item in intersection:
        if isinstance(a[item], dict):
            if not isinstance(b[item], dict):
                result1[item] = a[item]
                result2[item] = b[item]
            else:
                d1, d2 = _dict_diff(a[item], b[item])
                if d1:
                    result1[item] = d1
                if d2:
                    result2[item] = d2
        else:
            if _isndarray(a[item]) or _isndarray(b[item]):
                equal = (a[item] == b[item]).all()
            elif isinstance(a[item], Iterable):
                equal = ( isinstance(b[item], Iterable)
                          and all(x == y for x, y in zip(a[item], b[item]))
                          and len(list(a[item])) == len(list(b[item])) )
                    # len() == len() tests for different length generators
            else:
                equal = (a[item] == b[item])
            if not equal:
                result1[item] = a[item]
                result2[item] = b[item]
    if len(result1) + len(result2) == 0:
        assert a == b, "Error in _dict_diff()"
    return result1, result2

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

def get_differing_keys(records):
    """
    Parameters
    ----------
    records: ParamRec instances
    """
    assert(isinstance(records, Iterable))
    assert(all(isinstance(rec, ParamRec) for rec in records))

    diffpairs = {(i,j) : _param_diff(records[i].parameters, records[j].parameters,
                                     records[i].label, records[j].label)
                  for i, j in itertools.combinations(range(len(records)), 2)}
    def get_keys(diff):
        assert(bool(hasattr(diff, 'key')) != bool(hasattr(diff, 'keys'))) # xor
        if hasattr(diff, 'key'):
            return set([diff.key])
        elif hasattr(diff, 'keys'):
            return set(diff.keys)
    differing_keys = set().union( *[ get_keys(diff)
                                     for diffpair in diffpairs.values() # all difference types between two pairs
                                     for difftype in diffpair.values()  # keys, nesting, type, value
                                     for diff in difftype ] )

    return differing_keys

def structure_keys(keys):
    #keys = sorted(keys)
    roots = set([key.split('.')[0] for key in keys])
    tree = {root: [] for root in roots}
    for root in roots:
        for key in keys:
            if '.' in key and key.startswith(root):
                tree[root].append('.'.join(key.split(".")[1:]))

    return ParameterSet({root: None if subkeys == [] else structure_keys(subkeys)
                         for root, subkeys in tree.items()})

###########################
# Parameter file expansion
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


###################
# ParameterSet sampler
###################

class NoPops:
    # Sentinel class
    pass

class ParameterSetSampler:
    """
    This class mainly serves two purposes:
      - Convert a distribution definition into a sampler for that parameter
      - Maintain a cache of the state of the RNG, so that draws
        a) are consistent across runs and code changes
           (only changes to the parameter file itself will change the chosen parameters)
        b) do not affect random draws from outside this module

    Each parameter description may contain a transform description. When that is
    the case, the defined distribution is on the transformed variable. The
    sampler will apply the inverse transformation before returning the variable,
    such that sampled variables are always on the original space.

    Usage
    -----
    >>> from parameters import ParameterSet
    >>> # Define two variables:
        # - `x` for which `log10(x)` is normally distributed with
        #   with mean 0, standard deviation 1 and shape (2,).
        #   This is equivalent to using a 'lognormal' distribution.
        # - `y` which is gamma distributed.
        dist_desc = ParameterSet({'seed': 100,
                                  'x': {
                                    'dist'     : 'normal',
                                    'shape'    : (2,),
                                    'loc'      : 0.,
                                    'scale'    : 1.,
                                    'transform': {
                                      'name': 'x -> logx',
                                      'to'  : 'x -> np.log10(x)',
                                      'back': 'logx -> 10**logx' }
                                    },
                                  'y': {
                                    'dist'     : 'gamma',
                                    'shape'    : (1,),
                                    'a'        : 2.,
                                    'scale'    : 1.
                                  }
                                }
    >>> sampler = ParameterSetSampler(dist_desc)
    >>> x = sampler.sample('x')
    >>> y = sampler.sample('y')
    >>> # Other ways to call sample():
    >>> xy = sampler.sample(['x', 'y'])  # ParameterSet
    >>> xy2 = sampler()  # ParameterSet
    >>> x == xy['x']  # False
    >>> x != xy2['x'] and xy['x'] != xy2['x']  # True
    >>> # Order in which we sample parameters does not matter
    >>> sampler2 = ParameterSetSampler(dist_desc)  # Uses same seed as `sampler`
    >>> y2 = sampler2.sample('y')
    >>> x2 = sampler2.sample('x')
    >>> x == x2  # True
    >>> y == y2  # True

    TODO: Usage with populations

    Note
    ----
    To achieve its goals, this class effectively maintains its own separate
    random number generator. This means that samples produced within this class may not
    not be independent from samples produced outside of it. This shouldn't be a problem if
    e.g. the 'other' samples are those used to induce noise in a simulation. However,
    generating other parameters with a separate RNG should be avoided.

    Parameters
    ----------
    dists: ParameterSet (param_desc format)
        For priors in `dists` which define a transform, the returned sampler
        will be for the *transformed* variable.
    seed: int
        If given, overrides any seed present in `dists`.

    TODO: Cast parameters with subpopulations as BroadcastableBlockArray ?
    """
    population_attrs = ['population', 'populations', 'mixture', 'mixtures', 'label', 'labels']
    def __init__(self, dists, seed=None):
        """
        """
        # Implementation:
        # In order to always sample the same way, we set an order for parameters.
        # We can then sample them sequentially (i.e. each is sampled once, before
        # any one is sampled twice).
        # At any time, we can save the state of the RNG and reload it later to continue sampling.

        dists = ParameterSet(dists)  # Normalize the input (allows e.g. urls)
        self._iter_idx = None        # Internal index for the iterator
        orig_state = np.random.get_state()   # Store the current RNG state so it can be reset later

        # Get population / mixture labels
        #popstrs = [ attr for attr in [getattr(dists, attr, None) for attr in self.population_strs]
                         #if attr is not None ]
        popattrs = [ attr for attr in self.population_attrs if attr in dists ]
        if len(popattrs) > 1:
            raise ValueError("Multiple populations specifications. Only one of {} is needed."
                             .format(population_strs))
        elif len(popattrs) == 1:
            popnames = dists[popattrs[0]]
        else:
            popnames = None

        # Set seed
        if seed is None:
            seed = getattr(dists, 'seed', None)
        if seed is not None:
            np.random.seed(seed)

        # Get all the variable names and fix their order.
        # If we didn't fix their order here, changing the order in the parameter file
        # would change the sampled numbers.
        self.varnames = sorted([name for name in dists if name not in ['seed'] + popattrs])

        # Create the samplers
        self._samplers = {
            varname: ParameterSampler(varname, dists[varname], popnames)
            for varname in self.varnames }

        self._samplers[self.varnames[0]].set_previous(
            self._samplers[self.varnames[-1]], -1)
        for i in range(1, len(self.varnames)):
            self._samplers[self.varnames[i]].set_previous(
                self._samplers[self.varnames[i-1]], 0)

        # Reset the RNG to its external state
        self.rng_state = np.random.get_state()
        np.random.set_state(orig_state)

    # At the moment we shouldn't access samplers directly, because they don't
    # set the RNG state. Eventually we should change this, and then providing
    # this iterator might become a good idea
    # #######
    # # Define iterator
    # def __iter__(self):
    #     self._iter_idx = -1  # Indicates index of last returned sampler
    #     return self

    # def __next__(self):
    #     if len(self.varnames) <= self._iter_idx + 1:
    #         self._iter_idx = None
    #         return StopIteration
    #     else:
    #         self._iter_idx += 1
    #         return self._samplers[self.varnames[self._iter_idx]]
    # # End iterator definition
    # #######

    @property
    def sampled_varnames(self):
        """Return the names of the variables which we are sampling."""
        return [name for name, sampler in self._samplers.items()
                if sampler.sampled_idx is not None]

    def sample(self, varname=None):
        """
        Return a sample for the variable identified with 'varname'.
        'Varname' can be a list of names, in which case a ParameterSet
        instance is returned, with each entry keyed by a variable name.
        If no variable is specified, the full set is sampled and
        returned as a ParameterSet.
        """
        orig_state = np.random.get_state()
        np.random.set_state(self.rng_state)

        if varname is None:
            varname = self.varnames

        if isinstance(varname, str) or not isinstance(varname, Iterable):
            res = self._samplers[varname]()
        else:
            res = ParameterSet(
                {name: self._samplers[name]() for name in varname})

        self.rng_state = np.random.get_state()
        np.random.set_state(orig_state)

        return res

class ParameterSampler:
    """
    Implements one of the samplers in ParameterSetSampler.
    Samplers are set as a circular chain: before computing a new sample,
    each checks the previous sampler to see if it has been computed up to
    the same index, plus an offset (offsets should be 0 or negative).
    This is done to ensure that the same parameter set (if it specifies
    a seed) always returns the same draws.

    Sampling happens in the __call__() method.
    """
    # TODO: See if some code can be shared with pymc3.PyMCPrior.get_dist()
    def __init__(self, name, desc, popnames=None):
        if not isinstance(desc, ParameterSetBase):
            # It's a fixed value: no need for sampling
            self.sampled_idx = None   # This indicates that we aren't sampling
            def get_sample():
                logger.debug("Getting {} sample.".format(self.name))
                return np.array(desc)
        else:
            if 'dist' not in desc:
                raise ValueError("Unrecognized distribution type '{}'."
                                .format(desc.dist))
            # if popnames is None:
            #     # Provide a default population name, in case there is only one
            #     # population (in which case no name is necessary)
            #     popnames = ["pop1"]
            self.sampled_idx = 0
            shapes = [()]
            pop_pattern = ()
                # pop_pattern indicates which dimensions are sampled with
                # different parameters for each population
            for s in desc.shape:
                if not isinstance(s, str):
                    shapes = [ r + (s,) for r in shapes ]
                    pop_pattern += (False,)
                else:
                    pop_pattern += (True,)
                    pop_sizes = s.split('+')
                    if len(pop_sizes) != len(popnames):
                        raise ValueError("The parameter '{}' has a shape with {} "
                                         "components, but we have {} populations."
                                         .format(name, len(pop_sizes), len(popnames)))
                    shapes = [ r + (int(psize),)
                               for r in shapes
                               for psize in pop_sizes ]

            pop_samplers = type(self).PopSampler(desc, popnames)
            self._popnames = popnames
            self._pop_samplers = pop_samplers
                # Used as a hacky handle to recover the sampler params

            def key(*poplabels):
                return ','.join(poplabels)

            # TODO: Remove special cases/pop_pattern and make a generic
            #       function that works with any shape
            if all(b is False for b in pop_pattern):
                # There is no population-based sampling
                assert(len(shapes) == 1)
                def get_sample():
                    logger.debug("Getting {} sample.".format(self.name))
                    return pop_samplers[NoPops](shapes[0])
            elif pop_pattern == (True,):
                def get_sample():
                    logger.debug("Getting {} sample.".format(self.name))
                    return np.block(
                        [pop_samplers[key(pop)](shape)
                         for pop, shape in zip(popnames, shapes)])
            elif pop_pattern == (False, True):
                def get_sample():
                    logger.debug("Getting {} sample.".format(self.name))
                    return np.block(
                        [ [ pop_samplers[key(pop)](shape)
                            for pop, shape in zip(popnames, shapes)] ] )
            elif pop_pattern == (True, False):
                def get_sample():
                    logger.debug("Getting {} sample.".format(self.name))
                    return np.block(
                        [ [pop_samplers[key(pop)](shape)]
                          for pop, shape in zip(popnames, shapes) ] )
            elif pop_pattern == (True, True):
                n = len(popnames)
                def get_sample():
                    logger.debug("Getting {} sample.".format(self.name))
                    return np.block(
                        [ [ pop_samplers[key(pop1, pop2)](shapes[i + j])
                            for pop2, j in zip(popnames, range(0, n**1, n**0)) ]
                          for pop1, i in zip(popnames, range(0, n**2, n**1)) ] )
            else:
                raise NotImplementedError("Population samplers for block-broadcastable "
                                          "pattern '{}' are not yet implemented."
                                          .format(pop_pattern))

        if isinstance(desc, ParameterSetBase) and 'transform' in desc:
            inverse = Transform(desc.transform.back)
            self._get_sample = lambda : inverse(get_sample())
        else:
            self._get_sample = get_sample
        self._cache = deque()
        self.name = name # Not actually used, but useful e.g. for debugging

    # =======
    class PopSampler:
        """Retrieval interface for the different block samplers in ParameterSampler"""
        def __init__(self, distparams, popnames):
            self.distparams = distparams
            self.key = None
            self.popnames = popnames

        def __getitem__(self, key):
            self.key = key    # Used in __getattr__
            if self.dist == 'normal':
                def _sample_pop(size):
                    return np.random.normal(self.loc, self.scale, size=size)
            elif self.dist == 'expnormal':
                def _sample_pop(size):
                    return np.exp(np.random.normal(self.loc, self.scale,
                                                   size=size))
            elif self.dist in ['exponential', 'exp']:
                def _sample_pop(size):
                    return np.random.exponential(self.scale, size=size)
            elif self.dist == 'gamma':
                def _sample_pop(size):
                    return np.random.gamma(shape=self.a, scale=self.scale,
                                           size=size)
            else:
                raise ValueError("Unrecognized distribution type '{}'."
                                .format(self.distparams.dist))

            def sample_pop(size):
                self.key = key    # Used in __getattr__
                res = _sample_pop(size)
                factor = self.factor
                if factor is not None:
                    res *= factor
                self.key = None
                return res

            self.key = None
            return sample_pop

        # Retrieve the population-specific
        # parameter, or fall back to the global one if the first
        # isn't given
        def __getattr__(self, attr):
            if self.key is None:
                # Not called from within __getitem__: return a value
                # for each population
                if self.popnames is not None:
                    return [self.distparams
                              .get(α, self.distparams)
                              .get(attr, getattr(self.distparams, attr, None))
                             for α in self.popnames]
                else:
                    return getattr(self.distparams, attr)

            elif self.key is not NoPops and attr in self.distparams[self.key]:
                return getattr(self.distparams[self.key], attr)
            else:
                return getattr(self.distparams, attr, None)
    # =======

    def __call__(self):
        if len(self._cache) == 0:
            self._sample()
        return self._cache.popleft()

    def _sample(self, sample_i=None):
        if self.sampled_idx is None:
            self._cache.append(self._get_sample())
        else:
            if sample_i is None:
                sample_i = self.sampled_idx + 1
            if sample_i > self.sampled_idx:
                while self.previous.sampled_idx < sample_i + self.previous_offset:
                    self.previous._sample(sample_i + self.previous_offset)
                self.sampled_idx += 1
                self._cache.append(self._get_sample())
            else:
                pass
                #assert(len(self._cache) > 0)

    @property
    def previous(self):
        if self._previous.sampled_idx is None:
            return self._previous.previous
        else:
            return self._previous

    @property
    def previous_offset(self):
        if self._previous.sampled_idx is None:
            # Add the previous sampler's offset, since it's skipped over
            return self._previous_offset + self._previous.previous_offset
        else:
            return self._previous_offset

    def set_previous(self, previous_sampler, offset):
        """Set the previous ParameterSampler in the chain."""
        if offset > 0:
            raise ValueError("Offset cannot be positive.")
        if offset not in [0, -1]:
            logger.warning("ParameterSampler index offsets are usually either 0 or -1. "
                           "You specified {}.".format(offset))
        self._previous = previous_sampler
        self._previous_offset = offset
