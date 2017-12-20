# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:56:37 2017

@author: alex
"""

import os
import os.path
import io
from collections import namedtuple
import logging
import numpy as np
import dill
from inspect import isclass
logger = logging.getLogger('mackelab.iotools')

Format = namedtuple("Format", ['ext'])
    # TODO: Extend Format to include load/save functions
defined_formats = {
    # List of formats known to the load/save functions
    'npr':  Format('npr'),
    'repr': Format('repr'),
    'dill': Format('dill')
    }

_load_types = {}

def register_datatype(type, typename=None):
    global _load_types
    assert(isclass(type))
    if typename is None:
        typename = type.__name__
    assert(isinstance(typename, str))
    _load_types[typename] = type

def get_free_file(path, bytes=True, max_files=100, force_suffix=False, start_suffix=None):
    """
    Return a file handle to an unused filename. If 'path' is free, return a handle
    to that. Otherwise, append a number to it until a free filename is found or the
    number exceeds 'max_files'. In the latter case, raise 'IOError'.

    Returning a file handle, rather than just a file name, avoids the possibility of a
    race condition (a new file of the same name could be created between the time
    where one finds a free filename and then opens the file).

    Parameters
    ----------
    path: str
        Path name. Can be absolute or relative to the current directory.
    bytes: bool (Default: True)
        (Optional) Specify whether to open the file for byte (True) or plain
text (False) output. Default is to open for byte output, which
        is suitable for passing to `numpy.save`.
    max_files: int
        (Optional) Maximum allowed number of files with the same name. If this
        number is exceeded, IOError is raised.
    force_suffix: bool (default False)
        (Optional) If True, a suffix '_#', where # is a number, is always added
        to the file name. Forcing suffixes also changes the default value
        of 'start_suffix' to 1.
    start_suffix: int (default 2)
        If creating a file with 'path' is unsuccessful (or 'force_suffix is
        set to True), this is the first number to try appending to the file name.

    Returns
    -------
    filehandle
        Write-only filehandle, as obtained from a call to
        `open(pathname, 'mode='xb')`.
    pathname: str
        Pathname (including the possibly appended number) of the opened file.
    """

    # Get a full path
    # TODO: is cwd always what we want here ?
    if path[0] == '/':
        #path is already a full path name
        pathname = path
    else:
        #Make a full path from path
        pathname = os.path.abspath(path)

    # Set the default value for start_suffix
    if start_suffix is None:
        start_suffix = 1 if force_suffix else 2

    # Set the mode
    if bytes:
        mode = 'xb'
    else:
        mode = 'x'

    # Make sure the directory exists
    os.makedirs(os.path.dirname(pathname), exist_ok=True)

    try:
        if force_suffix:
            raise IOError
        else:
            f = open(pathname, mode=mode)
            return f, pathname
    except IOError:
        name, ext = os.path.splitext(pathname)
        for i in range(start_suffix, max_files+start_suffix):
            appendedname = name + "_" + str(i) + ext
            try:
                f = open(appendedname, mode=mode)
                return f, appendedname
            except IOError:
                continue

        raise IOError("Number of files with the name '{}' has exceeded limit."
                      .format(path))


def save(file, data, format='npr', overwrite=False):
    """Save `data`. By default, only the 'numpy_repr' representation is saved, if the.
    Not only is the raw format more future-proof, but it can be an order of
    magnitude more compact.
    If the raw save is unsuccessful (possibly because 'data' does not provide a
    'raw' method), than save falls back to saving a plain (dill) pickle of 'data'.

    Parameters
    ----------
    file: str
        Path name or file handle
        TODO: Currently only path names are supported. File handles raise NotImplementedError.
    data: Python object
        Data to save
    format: str
        The format in which to save the data. Possible values are:
          - 'npr' (default) Save with the numpy_repr format. This is obtained by calling the
            method 'nprepr' on the `data`. If this call fails, a warning is issued
            and the 'dill' format is used.
            Output file have the extension 'npr'.
            Objects using this format should implement the `from_nprepr` method.
          - 'repr' Call `repr` on the data and save the resulting string to file. The save will
            fail (and fall back to 'dill' format) if the `repr` is simply inherited from object,
            as simply saving the object address is not useful for reconstructing it. Still, there
            is no way of ensuring that the `repr` is sufficiently informative to reconstruct the
            object, so make sure it is before using this format.
            Output file have the extension 'repr'.
            Objects using this format should implement the `from_repr` method.
          - 'dill' A dill pickle.
            Output file has the extension 'dill'
        Formats can also be combined as e.g. 'npr+dill'.
    overwrite: bool
        If True, allow overwriting previously saved files. Default is false, in which case
        a number is appended to the filename to make it unique.
    """
    selected_formats = set(format.split('+'))

    # Check argument - format
    bad_formats = [f for f in selected_formats if f not in defined_formats]
    selected_formats = selected_formats.difference(bad_formats)
    if len(bad_formats) > 0:
        format_names = ["'" + f + "'" for f in defined_formats]
        bad_format_names = ["'" + f + "'" for f in bad_formats]
        formatstr = "format"
        if (format_names) > 1:
            format_names = ", ".join(format_names[:-1]) + " and " + format_names[-1]
        if (bad_format_names) > 1:
            formatstr = "formats"
            bad_format_names = ", ".join(bad_format_names[:-1]) + " and " + bad_format_names[-1]
        logger.warning("Unrecognized save {} {}.".format(formatstr, bad_format_names)
                       + "Recognized formats are " + format_names)
        if len(selected_formats) == 0:
            logger.warning("Setting the format to {}.".format_names)
            # We don't want to throw away the result of a long calculation because of a
            # flag error, so instead we will try to save into every format and let the user
            # sort out the files later.
            format = '+'.join(format_names)

    # Check argument - file
    if isinstance(file, io.IOBase):
        # TODO: Implement
        raise NotImplementedError
    else:
        assert(isinstance(file, str))
        filename = file

    # Ensure target directory exists
    dirname = os.path.dirname(filename)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    output_paths = []

    # Save data in as numpy representation
    if 'npr' in selected_formats:
        ext = defined_formats['npr'].ext
        with output(filename, ext, True, overwrite) as (f, output_path):
            try:
                logger.info("Saving data to 'npr' format...")
                np.savez(f, **data.repr_np)
            except AttributeError:
                # TODO: Use custom error type
                logger.warning("Unable to save to numpy representation ('npr') format.")
                if 'dill' not in selected_formats:
                    # Warn the user that we will use another format
                    logger.warning("Will try a plain (dill) pickle dump.")
                    selected_formats.add('dill')
            else:
                output_paths.append(output_path)


    # Save data as representation string
    if 'repr' in selected_formats:
        fail = False
        if data.__repr__ is object.__repr__:
            # Non-informative repr -- abort
            fail = True
        else:
            ext = defined_formats['repr'].ext
            with output(filename, ext, False, overwrite) as (f, output_path):
                try:
                    logger.info("Saving data to plain-text 'repr' format'")
                    f.write(repr(data))
                except:
                    fail = True
                else:
                    output_paths.append(output_path)
        if fail:
            logger.warning("Unable to save to numpy representation ('npr') format.")
            if 'dill' not in selected_formats:
                # Warn the user that we will use another format
                logger.warning("Will try a plain (dill) pickle dump.")
                selected_formats.add('dill')

    # Save data in dill format
    if 'dill' in selected_formats:
        ext = defined_formats['dill'].ext
        with output(filename, ext, True, overwrite) as (f, output_path):
            logger.info("Saving data as a dill pickle.")
            dill.dump(data, f)
            output_paths.append(output_path)

    # Return the list of output paths
    return output_paths


def load(filename, types=None, load_function=None, input_format=None):
    """
    Load file at `filename`. How the data is loaded is determined by the input formt,
    which is inferred from the filename extension. It can also be given by `input_format`
    explicitly.
    If `load_function` is provided, it is be applied to the loaded data. Otherwise,
    we try to infer type from the loaded data.
      - For 'npr' data, we look for a 'type' entry.
      - For 'repr' data, we look at the first part of the returned string, before
        any whitespace or special characters.
    Type inference can be completely disabled by passing `load_function=False`.
    If a type is found, we then try to match it to one of the entries in `types`. Thus
    `types` should be a dictionary of str:Type pairs, where the keys match the type name
    stored in files, and the Type is a loaded Python class which is associated to that
    name; typically, the type name will be the class name, although that is not required.
    It is also possible to add to the list of default types by calling this module's
    `add_load_type` method.
    """
    global _load_types
    if types is not None:
        types = _load_types.copy().update(types)
    else:
        types = _load_types

    basepath, ext = os.path.splitext(filename)

    if len(ext) == 0 and input_format is None:
        raise ValueError("Filename has no extension. Please specify input format.")
    if input_format is None:
        input_format = ext[1:]
    if input_format not in defined_formats:
        raise ValueError("Unrecognized format '{}'.".format(input_format))

    if os.path.exists(filename):
        openfilename = filename
    else:
        openfilename = basepath + "." + defined_formats[input_format].ext.strip('.')
    if input_format == 'npr':
        data = np.load(filename)
        if load_function is False:
            pass
        elif load_function is not None:
            data = load_function(data)
        elif 'type' in data:
            # 'type' stored as a 0D array
            if (data['type'].ndim == 0
                and data['type'].dtype.kind in {'S', 'U'}
                and str(data['type']) in types) :
                # make sure it's really 0D
                cls = types[str(data['type'])]
                if hasattr(cls, 'from_repr_np'):
                    data = cls.from_repr_np(data)
            else:
                # TODO: Error message
                pass

    elif input_format == 'repr':
        with open(openfilename, 'r') as f:
            data = f.read()
        if load_function is False:
            pass
        elif load_function is not None:
            data = load_function(data)
        else:
            # Search for the type name in the initial data
            if data[0] == '<':
                i, j = data.index('>'), data.index('<')
                if i == -1 or (j > -1 and j < i):
                    # Misformed type specification
                    pass
                else:
                    test_clsname = data[1:i]
                    if test_clsname in types:
                        cls = types[test_clsname]
                        if hasattr(cls, 'from_repr'):
                            data = cls.from_repr(data)
    elif input_format == 'dill':
        with open(openfilename, 'rb') as f:
            try:
                data = dill.load(f)
            except EOFError:
                logger.warning("File {} is corrupted or empty. A new "
                               "one is being computed, but you should "
                               "delete this one.".format(filename))
                raise FileNotFoundError

    return data

class output():
    def __init__(self, path, ext, bytes, overwrite=False):

        # Add extension
        basepath, _ = os.path.splitext(path)
            # Remove possible extension from path
        if len(ext) > 0 and ext[0] != ".":
            ext = "." + ext
        self.output_path = basepath + ext
        self.orig_output_path = self.output_path
        self.overwrite = overwrite
        self.bytes = bytes

    def __enter__(self):
        # Open file
        try:
            if not self.overwrite:
                self.f, self.actual_path = get_free_file(self.output_path, bytes=self.bytes)
            else:
                mode = 'wb' if self.bytes else 'w'
                self.f = open(self.orig_output_path, mode)
                self.actual_path = self.orig_output_path
        except IOError:
            logger.error("Could not create a file at '{}'."
                         .format(self.orig_output_path))

        return self.f, self.actual_path

    def __exit__(self, type, value, traceback):
        self.f.close()
