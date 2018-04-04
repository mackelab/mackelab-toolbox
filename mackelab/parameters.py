# -*- coding: utf-8 -*-
"""
Provides the 'expand_params' method for expanding a parameter description strings.
This allows to use one descriptor to specify multiple parameter sets.



Created on Wed Sep 20 13:19:53 2017

@author: alex
"""

from collections import deque, OrderedDict, namedtuple, Iterable
import itertools
from types import SimpleNamespace
import hashlib
from numbers import Number
import numpy as np
import scipy as sp
import pandas as pd
import logging
logger = logging.getLogger(__file__)

from parameters import ParameterSet

from . import iotools
try:
    from . import smttk
    smttk_loaded = True
except (NameError, ImportError):
    smttk_loaded = False
from .utils import flatten

##########################
# Transformed parameters
##########################

import simpleeval
import ast
import operator

class Transform:
    # Replace the "safe" operators with their standard forms
    # (simpleeval implements safe_add, safe_mult, safe_exp, which test their
    #  input but this does not work with non-numerical types.)
    _operators = simpleeval.DEFAULT_OPERATORS
    _operators.update(
        {ast.Add: operator.add,
         ast.Mult: operator.mul,
         ast.Pow: operator.pow})
    # Allow evaluation to find operations in standard namespaces
    namespaces = {'np': np,
                  'sp': sp}

    def __new__(cls, *args, **kwargs):
        # Make calling Transform on a Transform instance just return the instance
        if len(args) > 0 and isinstance(args[0], Transform):
            # Don't return a new instance; just return this one
            return args[0]
        else:
            return super().__new__(cls)

    def __init__(self, transform_desc):
        # No matter what we do in __new__, __init__ is called, so we need
        # to check this again.
        if not isinstance(transform_desc, Transform):
            xname, expr = transform_desc.split('->')
            self.xname = xname.strip()
            self.expr = expr.strip()


    def __call__(self, x):
        names = {self.xname: x}
        names.update(self.namespaces)
        try:
            res = simpleeval.simple_eval(
                self.expr,
                operators=Transform._operators,
                names=names)
        except simpleeval.NameNotDefined as e:
            e.args = ((e.args[0] +
                       "\n\nThis may be due to a module function in the transform "
                       "expression (only numpy and scipy, as 'np' and 'sp', are "
                       "available by default).\nIf '{}' is a module or class, you can "
                       "make it available by adding it to the transform namespace: "
                       "`Transform.namespaces.update({{'{}': {}}})`.\nSuch a line would "
                       "typically be included at the beginning of the execution script "
                       "(it does not need to be in the same module as the one where "
                       "the transform is defined, as long as it is executed before)."
                       .format(e.name, e.name, e.name),)
                      + e.args[1:])
            raise
        return res

    @property
    def desc(self):
        return self.xname + " -> " + self.expr

class MetaTransformedVar(type):
    """Implements subclass selection mechanism of TransformedVarBase."""
    # Based on https://stackoverflow.com/a/14756633
    def __call__(cls, desc, *args, orig=None, new=None):
        if len(args) > 0:
            raise TypeError("TransformedVar() takes only one positional argument.")
        if not( (orig is None) != (new is None) ):  #xor
            raise ValueError("Exactly one of `orig`, `new` must be specified.")

        if desc is not None and all(s in desc for s in ['name', 'to', 'back']):
            # Standard way to make transformed variable by providing transform description
            var = TransformedVar.__new__(TransformedVar, desc, *args, orig=orig, new=new)
            var.__init__(desc, *args, orig=orig, new=new)
        elif desc is not None and 'transform' in desc:
            # Alternative way to make transformed variable: pass entire
            # variable description, which includes transform description
            var = TransformedVar.__new__(TransformedVar, desc.transform, *args, orig=orig, new=new)
            var.__init__(desc.transform, *args, orig=orig, new=new)
        else:
            # Not a transformed variable
            if orig is None:
                orig = new  # orig == new for non transformed variables
            var = NonTransformedVar.__new__(NonTransformedVar, orig)
            var.__init__(desc, orig)

        return var

class TransformedVarBase(metaclass=MetaTransformedVar):
    """
    Base class for TransformedVar and NonTransformedVar.
    Can use it's constructor to obtain either TransformedVar or NonTransformedVar,
    depending on whether `desc` provides a transform.
    """
    TransformName = namedtuple('TransformName', ['orig', 'new'])

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

class TransformedVar(TransformedVarBase):

    # def __new__(cls, desc, *args, orig=None, new=None):
    #     # Required because of MetaTransformedVar
    #     return super().__new__(cls)

    def __init__(self, desc, *args, orig=None, new=None):
        """
        Should pass exactly one of the parameters `orig` and `new`
        TODO: Allow non-symbolic variables. Possible implementation:
            Test to see if orig/new is a constant, callable or symbolic.
            Set the other orig/new to the same type.
        """
        if not all(s in desc for s in ['to', 'back', 'name']):
            raise ValueError("Incomplete transform description")
        self.to = Transform(desc.to)
        self.back = Transform(desc.back)
        if orig is not None:
            #assert(shim.issymbolic(orig))
            self.orig = orig
            self.new = self.to(self.orig)
        elif new is not None:
            #assert(shim.issymbolic(new))
            self.new = new
            self.orig = self.back(new)

        # Set the variable names
        self.names = self.TransformName(*[nm.strip() for nm in desc.name.split('->')])
        #assert(len(names) == 2)
        if hasattr(self.orig, 'name'):
            if self.orig.name is None:
                self.orig.name = self.names.orig
            else:
                assert(self.orig.name == self.names.orig)
        if hasattr(self.new, 'name'):
            if self.new.name is None:
                self.new.name = self.names.new
            else:
                assert(self.new.name == self.names.new)

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
        self.names = self.TransformName(orig=orig, new=new)
        self.orig.name = orig
        self.new.name = new

class NonTransformedVar(TransformedVarBase):
    """Provides an interface consistent with TransformedVar."""

    # def __new__(cls, orig):
    #     # Required because of MetaTransformedVar
    #     return super().__new__(cls)

    def __init__(self, desc, orig):
        """
        `desc` only used to provide name attribute. If you don't need name,
        you can pass `desc=None`.

        Parameters
        ----------
        desc: ParameterSet
            Transform description
        orig: variable
            Theano or Python numeric variable.
        """
        self.orig = orig
        self.to = lambda x: x
        self.back = lambda x: x
        self.new = orig
        # Set name
        if isinstance(desc, str):
            self.names = self.TransformName(desc, desc)
        elif desc is not None and 'name' in desc:
            nametup = desc.name.split('->')
            if len(nametup) == 1:
                self.names = self.TransformName(nametup[0], nametup[0])
            elif len(nametup) == 2:
                self.names = self.TransformName(*nametup)
                assert(self.names.new == self.names.orig)
            else:
                raise ValueError("Malformed transformation name description '{}'."
                                 .format(desc.name))
        if hasattr(self, 'names'):
            if hasattr(self.orig, 'name'):
                if self.orig.name is not None:
                    assert(self.orig.name == self.names.orig)
            else:
                self.orig.name = self.names.orig

    def rename(self, orig, new=None):
        """
        Rename the variables

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
        self.names = self.TransformName(orig=orig, new=orig)
        if hasattr(self.orig, 'name') and self.orig.name is not None:
            self.orig.name = orig

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

def get_filename(params, suffix=None, convert_to_arrays=True):
    """
    Generate a unique filename by hashing a parameter file.

    Parameters
    ----------
    params: ParameterSet
        Filename will be based on these parameters

    suffix: str or None
        If not None, an underscore ('_') and then the value of `suffix` are
        appended to the calculated filename

    convert_to_arrays: bool
        If true, the parameters are normalized by using the result of
        `params_to_arrays(params)` to calculate the filename.
    """
    if not isinstance(params, ParameterSet):
        logger.warning("'get_filename()' requires an instance of ParameterSet. "
                       "Performing an implicit conversion.")
        params = ParameterSet(params)
    if convert_to_arrays:
        params = params_to_arrays(params)
    if params == '':
        basename = ""
    else:
        if np.__version__ < '1.14' and _filename_printoptions['legacy'] != '1.13':
            logger.warning("You are running Numpy v{}. Numpy's string representation "
                           "algorithm was changed in v.1.14, meaning that computed "
                           "filenames will not be consistent with those computed on "
                           "more up-to-date systems. To ensure consistent filenames, "
                           "either update to 1.14, or set `mackelab.parameters._filename_printoptions['legacy']` "
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
        # We need a sorted dictionary of parameters, so that the hash is consistent
        flat_params = params_to_arrays(params).flatten()
            # flatten avoids need to sort recursively
            # _params_to_arrays normalizes the data
        sorted_params = OrderedDict( (key, flat_params[key]) for key in sorted(flat_params) )
        s = repr(sorted_params)
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
        if isinstance(val, (ParameterSet, dict)):
            params[name] = params_to_arrays(val)
        elif (not isinstance(val, str)
            and isinstance(val, Iterable)
            and all(isinstance(v, Number) for v in flatten(val))):
                # The last condition leaves objects like ('lin', 0, 1) as-is;
                # otherwise they would be casted to a single type
            params[name] = np.array(val)
    return ParamType(params)

def params_to_nonarrays(params):
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
        if isinstance(val, (ParameterSet, dict)):
            params[name] = params_to_nonarrays(val)
        elif isinstance(val, np.ndarray):
            params[name] = val.tolist()
    return ParamType(params)

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

###########################
# Comparing ParameterSets
###########################

ParamRec = namedtuple('ParamRec', ['label', 'parameters'])
    # Data structure for associating a name to a parameter set

class ParameterComparison:
    """
    Example usage:
        testparams = ParameterSet("path/to/file")
        records = mackelab.smttk.get_records('project').list
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
        if isinstance(params1[key], ParameterSet):
            if not isinstance(params2[key], ParameterSet):
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
        elif isinstance(params2[key], ParameterSet):
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

    Parameters
    ----------
    param_str: str
        The string descriptor for the parameters.
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

def strip_comments(s):
    return '\n'.join(line.partition('#')[0].rstrip()
                     for line in s.splitlines())

def _expand(s, fail_on_unexpanded, parser):
    #param_strs = [s]  # Start with a single parameter string
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


class ParameterSetSampler:
    """
    This class mainly serves two purposes:
      - Convert a distribution definition into a sampler for that parameter
      - Maintain a cache of the state of the RNG, so that draws
        a) are consistent across runs and code changes
           (only changes to the parameter file itself will change the chosen parameters)
        b) do not affect random draws from outside this module
    NOTE: To achieve its goals, this class effectively maintains its own separate
    random number generator. This means that samples produced within this class may not
    be independent from samples produced outside of it. This shouldn't be a problem if
    e.g. the 'other' samples are those used to induce noise in a simulation. However,
    generating other parameters with a separate RNG should be avoided.

    # TODO: Cast parameters with subpopulations as BroadcastableBlockArray ?
    """
    population_attrs = ['population', 'populations', 'mixture', 'mixtures', 'label', 'labels']
    def __init__(self, dists):
        """
        Parameters
        ----------
        dists: ParameterSet
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
        if 'seed' in dists:
            np.random.seed(dists.seed)

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
        if not isinstance(desc, ParameterSet):
            # It's a fixed value: no need for sampling
            self.sampled_idx = None   # This indicates that we aren't sampling
            def get_sample():
                logger.debug("Getting {} sample.".format(self.name))
                return np.array(desc)
        else:
            if 'dist' not in desc:
                raise ValueError("Unrecognized distribution type '{}'."
                                .format(desc.dist))
            if popnames is None:
                # Provide a default population name, in case there is only one
                # population (in which case no name is necessary)
                popnames = ["pop1"]
            self.sampled_idx = 0
            shapes = [()]
            pop_pattern = ()
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

            pop_samplers = type(self).PopSampler(desc)

            def key(*poplabels):
                return ','.join(poplabels)
            n = len(popnames)

            # TODO: Remove special cases/pop_pattern and make a generic
            #       function that works with any shape
            if pop_pattern == (True,):
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
                def get_sample():
                    logger.debug("Getting {} sample.".format(self.name))
                    return np.block(
                        [ [ pop_samplers[key(pop1, pop2)](shapes[i + j])
                            for pop2, j in zip(popnames, range(0, n**1, n**0)) ]
                          for pop1, i in zip(popnames, range(0, n**2, n**1)) ] )

        if isinstance(desc, ParameterSet) and 'transform' in desc:
            inverse = Transform(desc.transform.back)
            self._get_sample = lambda : inverse(get_sample())
        else:
            self._get_sample = get_sample
        self._cache = deque()
        self.name = name # Not actually used, but useful e.g. for debugging

    # =======
    class PopSampler:
        """Retrieval interface for the different block samplers in ParameterSampler"""
        def __init__(self, distparams):
            self.distparams = distparams
            self.key = None

        def __getitem__(self, key):
            self.key = key    # Used in __getattr__
            if self.dist == 'normal':
                def sample_pop(size):
                    self.key = key    # Used in __getattr__
                    res = np.random.normal(self.loc,
                                           self.scale, size=size)
                    self.key = None
                    return res
            else:
                raise ValueError("Unrecognized distribution type '{}'."
                                .format(distparams.dist))
            self.key = None
            return sample_pop

        # Retrieve the population-specific
        # parameter, or fall back to the global one if the first
        # isn't given
        def __getattr__(self, attr):
            if attr in self.distparams[self.key]:
                return getattr(self.distparams[self.key], attr)
            else:
                return getattr(self.distparams, attr)
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
