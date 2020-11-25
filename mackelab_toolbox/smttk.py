import os
import re
import glob
import operator
import itertools
import multiprocessing
from collections import namedtuple, deque, OrderedDict
from collections.abc import Iterable, Sequence, Callable
from datetime import datetime
import logging
import numpy as np
logger = logging.getLogger(__file__)
try:
    import pandas as pd
    pandas_loaded = True
    pd.set_option('display.max_rows', 15)  # Truncate summary dataframes
except ImportError:
    pandas_loaded = False

from parameters import ParameterSet
import sumatra.commands
from sumatra.recordstore import DjangoRecordStore as RecordStore
    # The usual RecordStore, provided as a convenience
from sumatra.records import Record

import mackelab_toolbox as mtb
import mackelab_toolbox.parameters


##################################
# Custom errors
class RecordNotFound(Exception):
    pass

##################################
#
# Accessing the record store
#
##################################

# TODO: Make these methods of a "RecordStoreView class"
# TODO: Make `get_records` return a lazy, filterable query, to avoid loading
#       all records every time.
#       This might be possible by using Django's LazyQuery(?).

def get_record(recordstore, project, label):
    """
    Retrieve a singe record. In contrast to `get_records()`, the result is not wrapped in a list.
    """
    assert( isinstance(label, str) )
        # Don't accept lists as input
    res = get_records(recordstore, project, label)
    if len(res) > 1:
        raise RuntimeError("More than one record was found. Specify a unique label.")
    return res[0]

def get_records(recordstore, project, labels=None,
                script=None,
                before=None, after=None,
                min_data=1
                ):
    """
    Return the records whose labels match `label`.
    The filters may be partial, i.e. the parameter sets of all records matching
    '*label*', '*script*',... are returned.

    min_data: int
        Minimum number of output files that should be associated with a record.
        Default value of 1 excludes all records that have no associated data.
    """
    # TODO: Use database backend so that not all records need to be loaded into memory just
    #       to filter them.
    if labels is not None:
        # RecordStore has builtin functions for searching on labels
        if isinstance(labels, str):
            labels = [labels]
        lbl_gen = (fulllabel for fulllabel in recordstore.labels(project)
                   if any(label in fulllabel for label in labels))
        record_list = [recordstore.get(project, fulllabel) for fulllabel in lbl_gen]
    else:
        record_list = recordstore.list(project)

    reclist = RecordList(record_list)

    if script is not None:
        reclist = reclist.filter.script(script)
        #record_list = [record for record in record_list if script in record.main_file]

    if before is not None:
        reclist = reclist.filter.before(before)
        #if isinstance(before, tuple):
            #before = datetime(*before)
        #if not isinstance(before, datetime):
            #tnorm = lambda tstamp: tstamp.date()
        #else:
            #tnorm = lambda tstamp: tstamp
        #record_list = [rec for rec in record_list if tnorm(rec.timestamp) < before]
    if after is not None:
        reclist = reclist.filter.after(after)
        #if isinstance(after, tuple):
            #after = datetime(*after)
        #if not isinstance(after, datetime):
            #tnorm = lambda tstamp: tstamp.date()
        #else:
            #tnorm = lambda tstamp: tstamp
        #record_list = [rec for rec in record_list if tnorm(rec.timestamp) >= after]


    if min_data > 0:
        reclist = reclist.filter.output(minimum=min_data)

    return reclist.list

class RecordView:
    """
    A read-only interface to Sumatra records with extra convenience methods.
    In contrast to Sumatra.Record, RecordView is hashable and thus can be used in sets
    or as a dictionary key.
    """

    def __new__(cls, record, *args, **kwargs):
        if isinstance(record, RecordView):
            return self
        else:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, record):
        if not isinstance(record, Record):
            raise ValueError("'record' must be an instance of sumatra.records.Record.")
        self._record = record
        # Setting read-only attributes with a loop as below /seems/ to work, but
        # causes trouble when filtering
        #for attr in […]
            # def getter(self):
            #     return getattr(self._record, attr)
            # setattr(self, attr, property(getter))

    def __hash__(self):
        # Hash must return an int
        # The following returns the label, converting non-numeric characters to their
        # ASCII value, so '20180319-225426' becomes 2018031945225426.
        return int(''.join(c if c.isnumeric() else str(ord(c)) for c in self.label))

    # Set all the Record attributes as read-only properties
    @property
    def timestamp(self):
        return self._record.timestamp
    @property
    def label(self):
        return self._record.label
    @property
    def reason(self):
        return self._record.reason,
    @property
    def duration(self):
        return self._record.duration
    @property
    def executable(self):
        return self._record.executable
    @property
    def repostitory(self):
        return self._record.repository
    @property
    def main_file(self):
        return self._record.main_file
    @property
    def version(self):
        return self._record.version
    @property
    def parameters(self):
        return self._record.parameters
    @property
    def input_data(self):
        return input_data
    @property
    def script_arguments(self):
        return self._record.script_arguments
    @property
    def launch_model(self):
        return launch_mode
    @property
    def datastore(self):
        return self._record.datastore
    @property
    def input_datastore(self):
        return self._record.input_datastore
    @property
    def outcome(self):
        return self._record.outcome
    @property
    def output_data(self):
        return self._record.output_data
    @property
    def tags(self):
        return self._record.tags,
    @property
    def diff(self):
        return self._record.diff
    @property
    def user(self):
        return self._record.user
    @property
    def on_changed(self):
        return self._record.on_changed
    @property
    def stdout_stderr(self):
        return self._record.stdout_stderr
    @property
    def repeats(self):
        return self._record.repeats

    # Reproduce the Record interface; database writing functions are deactivated.
    def __nowrite(self):
        raise AttributeError("RecordView is read-only – Operations associated with "
                             "running or writingg to the database are disabled.")
    def register(self, *args, **kwargs):
        self.__nowrite()
    def run(self, *args, **kwargs):
        self.__nowrite()
    def __repr__(self):
        return repr(self._record)
    def describe(self, *arg, **kwargs):
        return self._record.describe(*args, **kwargs)
    def __ne__(self, other):
        return self._record != other
    def __eq__(self, other):
        return self._record == other
    def difference(self, *args, **kwargs):
        return self._record.difference(*args, **kwargs)
    def delete_data(self):
        self.__nowrite()
    @property
    def command_line(self):
        return self._record.command_line
    @property
    def script_content(self):
        return self._record.script_content

    # New functionality
    @property
    def outputpath(self):
        return [ os.path.join(self.datastore.root, output_data.path)
                 for output_data in self.output_data ]

    def get_outputpath(self, include=None, exclude=None, filter=None, keepdim=False):
        """
        Parameters
        ----------
        include: str, or list/tuple of str
            Only return paths that include any one of the given strings

        exclude: str, or list/tuple of str
            Only return paths that do not include any of the given strings

        filter: function
            Arbitrary function taking a single string (path) as input. If given,
            only paths for which this function returns True are returned.

        keepdim: bool
            By default, if only one path is found, it is returned without
            being wrapped in a list. This is convenient in interactive use,
            but for scripts it can be better to have a consistent return type.
            Specifying `keepdim=True` indicates to always return a list.

        Returns
        -------
        list of str, or str
            List of output paths, or the bare path if there is only one and keepdim=False.
        """
        # Construct the filter function
        if filter is None:
            filter = lambda x: True
        if include is None:
            include = ("",) # any() on an empty tuple returns False
        elif isinstance(include, str) or not isinstance(include, Iterable):
            include = (include,)
        if exclude is None:
            exclude = ()
        elif isinstance(exclude, str) or not isinstance(exclude, Iterable):
            exclude = (exclude,)
        def filter_fn(path):
            return ( any(s in path for s in include)
                    and not any(s in path for s in exclude)
                    and filter(path) )

        # Get output data path(s)
        paths = [path for path in self.outputpath if filter_fn(path)]

        if not keepdim and len(paths) == 1:
            paths = paths[0]

        return paths


    def get_datapath(self, data_idx=None):
        logger.warning("Deprecation warning: use `outputpath` property.")
        if data_idx is None:
            if len(self.output_data) > 1:
                raise ValueError("Multile output files : \n"
                                 + '\n'.join(self.output_data)
                                 + "\nYou must specify an output index")
            else:
                data_idx = 0
        path = os.path.join(self.datastore.root, self.output_data[data_idx].path)
        return path

    def extract(self, field):
        """
        Retrieve record value corresponding to the field keyword.
        'Field' can specify any defined extraction rule, although generally
        it simply refers to a particular field in the record.
        TODO: Provide an extension mechanism so users can define their own
              extraction rules.
        """
        def splitarg(arg, default):
            """Extract the value following '=' in `arg`."""
            if '=' in arg:
                res = arg.split('=')
                if len(res) != 2:
                    raise ValueError("Malformed extraction argument "
                                        "'{}'".format(arg))
                else:
                    return res[1]
            else:
                return default
        if 'parameters' == field:
            return self.parameters
        elif 'outputpath' in field:
            idx = splitarg(field, None)
            outputpath = self.outputpath
            if len(outputpath) == 0:
                raise IndexError("No output data is associated to this record.")
            if idx is not None:
                return outputpath[idx]
            else:
                if len(outputpath) == 1:
                    return outputpath[0]
                else:
                    raise IndexError("This record has more than one output; "
                                     "you must provide an index.")
        else:
            extract_rules = ['parameters', 'outputpath']
            # TODO: Add custom extraction rules here
            extract_rules = ["'" + rule + "'" for rule in extract_rules]
            raise ValueError("You tried extracting data from a record using rule '{}', "
                             "which is undefined. Currently defined rules for are {} and {}."
                             .format(field, ', '.join(extract_rules[:-1]), extract_rules))

_cmpop_ops = ['lt', 'le', 'eq', 'ne', 'gt', 'ge']
    # Ops which are implemented in the `operator` module
_cmpop_strings = _cmpop_ops + ['isin']
    # Other ops with custom implementations

def _get_compare_op(cmpop):
    """
    Parameters
    ----------
    cmpop: str or function
        'Compare Op'
        If a string, should be one of 'lt', 'le', 'eq', 'ne', 'gt', 'ge', 'isin'.
        If a function, should take two arguments and return a bool
    """
    # Custom filters
    def isin(a, b):
        """Return `a in b`"""
        return a in b

    # FIXME: No way to set a custom function with current interface
    if isinstance(cmpop, str):
        if cmpop in _cmpop_ops:
            return getattr(operator, cmpop)
        elif cmpop in _cmpop_strings:
            if cmpop == 'isin':
                return isin
        else:
            raise ValueError("Unrecognized comparison operation '{}'.".format(cmpop))
    elif isinstance(cmpop, Callable):
        return cmpop
    else:
        raise ValueError("'cmpop' must be either a string corresponding to a "
                            "comparison operation (like 'lt' or 'eq'), or a callable "
                            "implementing a comparison operation. The passed value is "
                            "of type '{}', which is not compatible with either of these "
                            "forms.".format(type(cmpop)))

class ParameterSetFilter:
    """
    Specialized filter, meant to be used as [record list].filter.parameters
    """

    def __init__(self, reclst, cmpop='eq', key=None):
        """
        Parameters
        ----------
        reclist: RecordList
            Instance of RecordList.
        cmpop: str or function
            'Compare Op'
            If a string, should be one of 'lt', 'le', 'eq', 'ne', 'gt', 'ge', 'isin'.
            If a function, should take two arguments and return a bool
        """
        self.reclst = reclst
        self._cmp = _get_compare_op(cmpop)
        self.key = key

    def __getattr__(self, attr):
        if attr in _cmpop_strings:
            return type(self)(self.reclst, attr)
        else:
            key = self.join_keys(self.key, attr)
            return ParameterSetFilter(self.reclst, self._cmp, key=key)

    def __call__(self, paramset):
        """
        Parameters
        ----------
        paramset: ParameterSet
        """
        def test(paramset, key, value):
            if not isinstance(paramset, ParameterSet):
                logger.debug("ParameterSetFilter: skipping record whose "
                             "`parameters` attribute is of type '{}'."
                             .format(type(paramset)))
                return False
            try:
                fullkey = self.join_keys(self.key, key)
                if fullkey is None:
                    paramset_value = paramset
                else:
                    paramset_value = paramset[fullkey]
            except KeyError:
                return False
            else:
                return self.cmp(paramset_value, value)
        if isinstance(paramset, dict):
            paramset = ParameterSet(paramset)

        if isinstance(paramset, ParameterSet):
            return RecordList(rec for rec in self.reclst
                              if all( test(rec.parameters, key, paramset[key])
                                      for key in paramset.flatten().keys()))
                # Currently paramset.flatten() doesn't deal with '->' referencing, so can't use .items()
        else:
            return RecordList(rec for rec in self.reclst
                              if test(rec.parameters, None, paramset))

    def _get_paramset(record):
        if isinstance(record, ParameterSet):
            return record
        else:
            return record.parameters

    def cmp(self, a, b):
        return np.all(self._cmp(a, b))

    def join_keys(self, key1, key2):
        if all(key in ('', None) for key in (key1, key2)):
            return None
        elif key1 in ('', None):
            return key2
        elif key2 in ('', None):
            return key1
        else:
            return '.'.join((key1, key2))

class _AnyRecordFilter:
    class wrapped_function:
        def __init__(self, f, reclst):
            self.f = f
            self.reclst = reclst
        def __call__(self, arglist):
            recordlists = [self.f(arg).list for arg in arglist]
                # We use .list to make sure we have separate iterators for each arg
                # TODO: Use itertools.tee instead ? That would avoid allocating the lists
            return RecordList(set(itertools.chain.from_iterable(recordlists)))
    def __init__(self, recordfilter):
        self.recordfilter = recordfilter
    def __getattr__(self, attr):
        filterfn = getattr(self.recordfilter, attr)
        self.recordfilter.reclst.list
        if isinstance(filterfn, Callable):
            return self.wrapped_function(getattr(self.recordfilter, attr), self.recordfilter.reclst)
        else:
            raise AttributeError("RecordFilter.{} is not a callable method."
                                 .format(attr))

class _AllRecordFilter:
    def __init__(self, recordfilter):
        self.recordfilter = recordfilter
    def __getattr__(self, attr):
        raise NotImplementedError

# TODO: Not (or None ?) record filter

class RecordFilter:
    """
    Can overwrite RecordFilter.on_error_defaults to change default behaviour for
    all filters.
    Filters can have three defined behaviours when they catch an error. A common
    need for example is to have the filter catch AttributeError, either to reject
    all elements that don't have a particular attribute (False), or to avoid filtering
    elements that don't have a particular attribute (True). By default the
    `AttributeError` error in `on_error_defaults` is set to False.
      - False: the condition returns False
      - True: the condition returns True
      - 'raise': the error is reraised. The same happens if there is no entry
        corresponding to the error in `on_error_defaults`.
    """
    on_error_defaults = {
        AttributeError: False
    }

    def __init__(self, record_list):
        self.reclst = record_list
        self.parameters = ParameterSetFilter(record_list)
        # Multi-condition filter wrappers
        self.any = _AnyRecordFilter(self)
        self.all = _AllRecordFilter(self)

    # Default filter
    def __call__(self, cond, errors=None):
        on_error = self.on_error_defaults
        if errors is not None:
            on_error.update(errors)
        def test(rec):
            """Wraps 'cond' with the error handlers specified by 'on_error'"""
            try:
                return cond(rec)
            except tuple(on_error.keys()) as e:
                if on_error[type(e)] == 'raise':
                    raise
                else:
                    logger.debug("Filtering raised {}. Ignored. (RecordFilter."
                                 "on_error_defaults)".format(str(type(e))))
                    return on_error[type(e)]
        iterable = filter(test, self.reclst)
        return RecordList(iterable)

    # Custom filters
    def output(self, minimum=1, maximum=None):
        # TODO: Use iterable that doesn't need to allocate all the data
        iterable = [rec for rec in self.reclst
                    if ((minimum is None or len(rec.output_data) >= minimum)
                        and (maximum is None or len(rec.output_data) <= maximum))]
        return RecordList(iterable)

    def before(self, date, *args):
        """
        Keep only records which occured before the given date. Date is exclusive
        Can provide date either as a single tuple, or multiple arguments as for datetime.datetime()
        """
        if isinstance(date, datetime):
            if len(args) > 0:
                raise ValueError("Too many arguments for `filter.before()`")
        elif isinstance(date, tuple):
            date = datetime(*(date+args))
        elif isinstance(date, int) and len(str(date)) <= 8:
            # Convenience interface to allow dropping the commas
            # Date can be an integer of length 4, 5, 6, 7 or 8; if less than 8
            # digits, well be extended with the earliest date (so 2018 -> 20180101)
            datestr = str(date)
            if len(datestr) < 4:
                raise ValueError("Date integer must give at least the year.")
            elif len(datestr) < 8:
                Δi = 8 - len(datestr)
                datestr = datestr + "0101"[-Δi:]
                date = int(datestr)
            year, month, day = date//10000, date%10000//100, date%100
            date = datetime(year, month, day)
        else:
            date = datetime(date, *args)
        if not isinstance(date, datetime):
            tnorm = lambda tstamp: tstamp.date()
        else:
            tnorm = lambda tstamp: tstamp
        return RecordList(rec for rec in self.reclst if tnorm(rec.timestamp) < date)

    def after(self, date, *args):
        """
        Keep only records which occurred after the given date. Date is inclusive.
        Can provide date either as a single tuple, or multiple arguments as for datetime.datetime()
        """
        if isinstance(date, datetime):
            if len(args) > 0:
                raise ValueError("Too many arguments for `filter.after()`")
        elif isinstance(date, tuple):
            date = datetime(*(date+args))
        elif isinstance(date, int) and len(str(date)) <= 8:
            # Convenience interface to allow dropping the commas
            # Date can be an integer of length 4, 5, 6, 7 or 8; if less than 8
            # digits, well be extended with the earliest date (so 2018 -> 20180101)
            datestr = str(date)
            if len(datestr) < 4:
                raise ValueError("Date integer must give at least the year.")
            elif len(datestr) < 8:
                Δi = 8 - len(datestr)
                datestr = datestr + "0101"[-Δi:]
                date = int(datestr)
            year, month, day = date//10000, date%10000//100, date%100
            date = datetime(year, month, day)
        else:
            date = datetime(date, *args)
        if not isinstance(date, datetime):
            tnorm = lambda tstamp: tstamp.date()
        else:
            tnorm = lambda tstamp: tstamp
        return RecordList(rec for rec in self.reclst if tnorm(rec.timestamp) >= date)

    def on(self, date, *args):
        """
        Keep only records which occurred on the given date.
        Can provide date either as a single tuple, or multiple arguments as for datetime.datetime()
        """
        if isinstance(date, datetime):
            if len(args) > 0:
                raise ValueError("Too many arguments for `filter.on()`")
        elif isinstance(date, tuple):
            date = datetime(*(date+args))
        elif isinstance(date, int) and len(str(date)) <= 8:
            # Convenience interface to allow dropping the commas
            datestr = str(date)
            if len(datestr) < 4:
                raise ValueError("Date integer must give at least the year.")
            elif len(datestr) < 8:
                Δi = 8 - len(datestr)
                datestr = datestr + "0101"[-Δi:]
                date = int(datestr)
            year, month, day = date//10000, date%10000//100, date%100
            date = datetime(year, month, day)
        else:
            date = datetime(date, *args)
        after = date
        before = date.replace(day=date.day+1)

        if not isinstance(date, datetime):
            tnorm = lambda tstamp: tstamp.date()
        else:
            tnorm = lambda tstamp: tstamp
        return RecordList(rec for rec in self.reclst if tnorm(rec.timestamp) >= after and tnorm(rec.timestamp) < before)

    def script(self, script):
        return RecordList(record for record in self.reclst if script in record.main_file)

    def label(self, label):
        return RecordList(rec for rec in self.reclst if label in rec.label)

    def reason(self, reason):
        return RecordList(rec for rec in self.reclst
                              if any(reason in line for line in rec.reason))

    def outputpath(self, output):
        return RecordList(rec for rec in self.reclst
                          if any(output in path for path in rec.outputpath))

class RecordList:
    """
    This class ensures that all elements of an iterable are RecordViews; it
    can be underlined by any iterable. It will not automatically cast the iterable
    as a list – a version underlined by a list (which thus supports len(), indexing,
    etc.) can be obtained through the .list attribute.
    It also provides a filter interface. If `reclist` is a RecordList, then
        - `reclist.filter(cond)` is the same as `filter(reclist, cond)`
        - `reclist.filter.output_data()` filters the list based on output data
    We expect to add more filters as time goes on.

    FIXME: If constructed from a generator, can only iterate once. Should do
           something to protect the user from deleting data by e.g. calling `list()`.
    """
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = None
        self.filter = RecordFilter(self)

    def __len__(self):
        return len(self.iterable)

    def __getitem__(self, key):
        try:
            res = self.iterable[key]
        except TypeError:
            # For convenience, cast recordlist to an indexable list rather than throwing an error
            # Once we've done this and allocated the memory, we might as well replace the internal iterable
            self.iterable = list(self.iterable)
            res = self.iterable[key]

        if isinstance(res, Iterable):
            res = RecordList(res)
        return res

    def __iter__(self):
        self.iterator = iter(self.iterable)
        return self

    def __next__(self):
        rec = next(self.iterator)
        if isinstance(rec, RecordView):
            # Skip the unecessary casting step
            return rec
        elif isinstance(rec, Record):
            return RecordView(rec)
        else:
            raise ValueError("A RecordList may only be composed of sumatra "
                             "records, but this one contains element(s) of "
                             "type '{}'".format(type(rec)))

    @property
    def latest(self):
        """
        Return the record with the latest timestamp.
        """
        latest = None
        for rec in self:
            if latest is None:
                latest = rec
            elif latest.timestamp < rec.timestamp:
                latest = rec
        return latest

    @property
    def earliest(self):
        """
        Return the record with the earliest timestamp.
        """
        earliest = None
        for rec in self:
            if earliest is None:
                earliest = rec
            elif earliest.timestamp >= rec.timestamp:
                # > would also work, but >= makes this a better opposite
                # operation to `earliest`
                earliest = rec
        return earliest

    @property
    def list(self):
        """
        Convert the internal iterable to a list. This is useful to avoid consuming
        the iterable, if we need to make multiple passes through it.

        Returns
        -------
        self
        """
        self.iterable = list(self.iterable)
        return self

    @property
    def summary(self):
        """
        Return a RecordListSummary.
        """
        if not isinstance(self.iterable, Sequence):
            # Should catch most cases where iterating consumes the iterable
            # Once we've done this and allocated the memory, we might as well replace the internal iterable
            self.iterable = list(self.iterable)
        return RecordListSummary(self)

    def get(self, label):
        """
        Retrieve the record corresponding to the given label.

        Parameters
        ----------
        label: str | list of str
            Label of the record we want to retrieve.
            Can also be an iterable of labels. In this case the record corresponding
            to each is retrieved, and the result returned as a RecordList.

        Returns
        -------
        Record or RecordList
            Returns a Record if `label` is a `str`, a RecordList otherwise.
        """
        if not isinstance(label, str):
            if not isinstance(label, Iterable):
                raise ValueError("`label` must either be a single string label, "
                                 "or a list of labels")
            # TODO: Use 'or' label filter once that is implemented
            return RecordList([self.get(lbl) for lbl in label])

        found = self.filter(lambda rec: rec.label == label).list
            # Don't use the label filter because it uses 'in' instead of '==' comparison
        if len(found) == 0:
            raise RecordNotFound("No record has a label corresponding to '{}'."
                                 .format(label))
        elif len(found) == 1:
            return found[0]
        else:
            raise RecordNotFound("{} records have a label corresponding to '{}'."
                                 .format(len(found), label))

    def extract(self, *args ):
        """
        Typical usage: `record_list.extract('parameters', 'datapath')`
        """
        fields = [ field.split("=")[0] for field in args ]
        RecAttributes = namedtuple("RecAttributes", fields)
        for rec in self:
            yield RecAttributes(*(rec.extract(field) for field in args))

    def get_datapaths(self, common_params=None,
                  include=None, exclude=None, filter=None,
                  return_parameters=False):
        """
        Parameters
        ----------
        common_params: str, or list of str
            Unless None, `mackelab_toolbox.parameters.prune()` will be applied to each record with
            this filter specification. An error is raised if any of the returned
            ParameterSets differs from the others.

        include, exclude, filter: same as RecordView.get_outputpath()

        return_parameters: bool
            If True, also return the common parameters. `common_params` must also
            be set. Default value is False.

        Returns
        -------
        List of str (paths).
        Optionally also returns a ParameterSet. The returned ParameterSet is a copy
        of `common_params`, such that it can be modified with affecting the original.
        """
        # if not isinstance(records, Iterable):
        #     raise ValueError("`records` must be a list of records.")
        # if len(records) == 0:
        #     raise ValueError("`records` list is empty.")
        records = list(self)

        # Check to make sure that all records have the same parameters
        if common_params is not None:
            test_params = mtb.parameters.prune(records[0].parameters, common_params)
            for rec in records:
                # FIXME: Following does not work with numpy arrays
                #        Need to specialize comparisons on ParameterSets to apply
                #        .any() / .all() comparisons between numpy components.
                if mtb.parameters.prune(rec.parameters, common_params) != test_params:
                    raise ValueError("Parameters differ between records")

        # Get the data paths
        data_paths = []
        for rec in records:
            rec_paths = rec.get_outputpath(include, exclude, filter, keepdim=True)
            data_paths.extend(rec_paths)

        # Return the paths, possibly along with the parameters
        if return_parameters:
            if common_params is None:
                raise ValueError("Must specify a filter for common parameters "
                                "in order to return a parameter set.")
            return data_paths, test_params.copy()
        else:
            return data_paths

import re
import pandas as pd
class RecordListSummary(OrderedDict):
    """
    **Class attributes**

    - `re_merge_labels`: Records are merged if the first match group this regex
      results in the same string.
      Default matches the standard timestamp label format with alphanumerical
      suffix: YYYYMMDD-HHMMSS_[aa]

      Timestamps must match but the suffix may differ, as long as it is composed
      only of alphanumerical characters. So records with the following labels
      would be merged:

      - 20200604-123060
      - 20200604-123060_1
      - 20200604-123060_10
      - 20200604-123060_a4ef

      But the following wouldn't

      - 20200604-123060_a4ef_
      - 20200604-123060_a4-ef
      - 20200604-123160
      - 20200604-123260

      Default value: ``r'^(\d{8,8}-\d{6,6})_([a-zA-Z0-9]+)?$'``

    """
    re_merge_labels = r'^(\d{8,8}-\d{6,6})_([a-zA-Z0-9]+)?$'
    def __init__(self, recordlist, base=None, merge=False):
        """
        Parameters
        ----------
        recordlist: RecordList | None
            Record list to summarizes. Records whose labels differ by only a
            suffix are combined.
            Set to `None` creates an empty summary, unless `base` is not None.
        base: None | OrderedDict
            Initialize the summary with this dictionary. If `None`, summary
            is initialized as an empty dictionary to which the entries of
            `recordlist` are then added.
        merge: bool
            Whether to merge similar labels according to `re_merge_labels`.
            Default is ``False``.
        """
        if base is None: base = ()  # Empty initialization
        elif not isinstance(base, OrderedDict):
            raise ValueError("`base` argument to `RecordListSummary` most be "
                             "an OrderedDict or a derived class, like "
                             "RecordListSummary.")
        super().__init__(base)

        if recordlist is None:
            return  # Skip iteration over `recordlist`
        if not merge:
            for r in recordlist:
                assert r.label not in self
                self[r.label] = [r]
            return
        assert merge
        for r in recordlist:
            # For labels following the standard format, merge records whose
            # labels differ only by a suffix
            # Scripts for these records were started within the same second,
            # thus almost assuredly at the same time with a dispatch script
            # such as smttk's `run`.
            m = re.match(self.re_merge_labels, r.label)
            if m is None or not m.groups():
                # Not a standard label format -- no merging
                assert(r.label not in self)
                self[r.label] = [r]
            else:
                # Standard label format
                shared_lbl = m[1]
                if shared_lbl in self:
                    self[shared_lbl].append(r)
                else:
                    self[shared_lbl] = [r]

    @property
    def merged(self):
        """Return a copy of the summary where similar labels are merged."""
        return RecordListSummary(mtb.utils.flatten(self.values()), merge=True)
    @property
    def unmerged(self):
        """Return a copy of the summary where similar labels are unmerged."""
        return RecordListSummary(mtb.utils.flatten(self.values()), merge=False)

    def __call__(self, *args, **kwargs):
        """
        Call `.dataframe()` with the given arguments
        """
        return self.dataframe(*args, **kwargs)

    def __str__(self):
        return str(self.dataframe())
    def __repr__(self):
        """Used to display the variable in text-based interpreters."""
        return repr(self.dataframe())
    def _repr_html_(self):
        """Used by Jupyter Notebook to display a nicely formatted table."""
        df = self.dataframe()
        # Instead of letting `_repr_html` truncate long lines, add hard line
        # breaks to the data (`self.dataframe()` returns a copy)
        colwidth = pd.get_option('display.max_colwidth')
        df.transform(self._add_newlines, colwidth=colwidth)
        pd.set_option('display.max_colwidth', None)
            # Deactivate line truncation for call to `_repr_html_`
        df_html = df._repr_html_()
        # Print newlines in text fields correctly
        df_html = df_html.replace("\\n", "<br>")
        pd.set_option('display.max_colwidth', colwidth)
            # Return `max_colwidth` to previous value
        return df_html

    def head(self, nrows):
        headkeys = itertools.islice(self.keys(), nrows)
        headrecs = OrderedDict( (key, self[key]) for key in headkeys )
        return RecordListSummary(None, headrecs)
    def tail(self, nrows):
        nkeys = len(self.keys())
        tailkeys = itertools.islice(self.keys(), nkeys-nrows, None)
        tailrecs = OrderedDict( (key, self[key]) for key in tailkeys )
        return RecordListSummary(None, tailrecs)

    @staticmethod
    def _truncate_value(attr, value, max_chars, max_lines):
        if attr == 'tags':
            # There's ordering of more relevant information with tags, and they
            # are usually short anyway.
            return value
        if isinstance(value, tuple):
            return tuple(RecordListSummary._truncate_value(attr, v, max_chars, max_lines)
                         for v in value)
        value = '\n'.join(value.splitlines()[:max_lines])
        if len(value) > max_chars:
            if attr == 'main_file':
                value = '…'+value[-max_chars+1:]
            else:
                value = value[:max_chars-1]+'…'
        return value

    def dataframe(self, fields=('reason', 'outcome', 'tags', 'main_file', 'duration'),
                  parameters=(), hash_parameter_sets=True,
                  max_chars=50, max_lines=1):
        """
        Parameters
        ----------
        ...
        max_chars: int
            Truncate record value after this many characters.
            Special case:
                'main_file' is truncated from the end instead of the beginning.
                'tags' are never truncated
        max_lines: int
            Keep only this number of lines, even if more lines would fit
            within the character limit.
        """
        def combine(recs, attr):
            # Combine the values from different records in a single entry
            def get(rec, attr):
                # Retrieve possibly nested attributes
                if '.' in attr:
                    attr, nested = attr.split('.', 1)
                    return get(getattr(rec, attr), nested)
                else:
                    value = getattr(rec, attr)
                    if hash_parameter_sets and isinstance(value, ParameterSet):
                        # Get a hash fingerprint (first 7 hash chars) of the file
                        h = mtb.parameters.get_filename(value)
                        value = '#' + h[:7]
                    return self._truncate_value(
                        attr, value, max_chars, max_lines)
            if attr == 'duration':
                s = sum(getattr(r, attr) for r in recs) / len(recs)
                h, s = s // 3600, s % 3600
                m, s = s // 60, s % 60
                return "{:01}h {:02}m {:02}s".format(int(h),int(m),int(s))
            else:
                vals = []
                for r in recs:
                    try:
                        vals.append(get(r, attr))
                    except (AttributeError, KeyError):
                        # Add string indicating this rec does not have attr
                        vals.append("undefined")
                vals = set(mtb.utils.flatten(vals))
                # Choose the most appropriate join character
                if any('\\n' in v for v in vals):
                    join_str = '\\n'
                elif sum(len(v) for v in vals) > pd.get_option('display.max_colwidth'):
                    join_str = '\\n'
                elif not any(',' in v for v in vals):
                    join_str = ', '
                elif not any(';' in v for v in vals):
                    join_str = '; '
                else:
                    join_str = ' | '
                # Join all parameters from the merged records into a single string
                return join_str.join(str(a) for a in vals)
        def format_field(field):
            # Take a field key and output the formatted string to display
            # in the dataframe header
            if field == 'duration':
                field = 'avg. duration'
            field = field.replace('.', '\n.')
            return field

        data = deque()
        # Append parameters to the list of fields
        # Each needs to be prepended with the record attribute 'parameters'
        if isinstance(parameters, str):
            parameters = (parameters,)
        fields += tuple('parameters.' + p for p in parameters)
        for lbl, sr in self.items():
            entry = tuple(combine(sr, field) for field in fields)
            entry = (len(sr),) + entry
            data.append(entry)
        data = np.array(data)

        fieldnames = tuple(format_field(field) for field in fields)
            # Add line breaks to make parameters easier to read, and take less horizontal space
        if pandas_loaded:
            if len(data) == 0:
                data = data.reshape((0, len(fieldnames)+1))
            return pd.DataFrame(data, index=self.keys(),
                                columns=('# records',) + fieldnames).sort_index(ascending=False)
        else:
            # TODO: Add index to data; make structured array
            logger.info("Pandas library not loaded; returning plain Numpy array.")
            return data

    def array(self, fields=('reason', 'tags', 'main_file', 'duration'),
                  parameters=()):
        """
        Return the summary as a NumPy array.
        NOTE: Not implemented yet
        """
        raise NotImplementedError

    @staticmethod
    def _add_newlines(val, colwidth):
        # TODO: Preferentially break on whitespace, '_'
        if isinstance(val, pd.Series):
            return val.transform(RecordListSummary._add_newlines,
                                 colwidth=colwidth)
        s = str(val)
        l = colwidth
        nlines = int(np.ceil(len(s) / l))
        return '\n'.join([s[i*l:(i+1)*l] for i in range(nlines)])
