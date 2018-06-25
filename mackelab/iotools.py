"""
Created on Wed Sep 20 16:56:37 2017

@author: alex
"""

import os
import os.path
import io
from collections import namedtuple, OrderedDict
import logging
import numpy as np
import dill
from inspect import isclass
from parameters import ParameterSet
logger = logging.getLogger('mackelab.iotools')

Format = namedtuple("Format", ['ext'])
    # TODO: Allow multiple extensions per format
    # TODO: Extend Format to include load/save functions
defined_formats = OrderedDict([
    # List of formats known to the load/save functions
    # The order of these formats also defines a preference, for when two types might be used
    ('npr',  Format('npr')),
    ('repr', Format('repr')),
    ('brepr', Format('brepr')), # Binary version of 'repr'
    ('dill', Format('dill'))
    ])

_load_types = {}

def register_datatype(type, typename=None):
    global _load_types
    assert(isclass(type))
    if typename is None:
        typename = type.__name__
    assert(isinstance(typename, str))
    _load_types[typename] = type

def find_registered_typename(type):
    """
    If `type` is a registered datatype, return it's associated name.
    Otherwise, find the nearest parent of `type` which is registered and return its name.
    If no parents are registered, return `type.__name__`.
    """
    def get_name(type):
        for registered_name, registered_type in _load_types.items():
            if registered_type is type:
                return registered_name
        return None

    typename = get_name(type)
    if typename is None:
        for base in type.__mro__:
            typename = get_name(base)
            if typename is not None:
                break
    if typename is None:
        typename = type.__name__
            # No registered type; return something sensible (i.e. the type's name)
    return typename


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
    """Save `data`. By default, only the 'numpy_repr' representation is saved,
    if `data` defines a numpy representation.
    Not only is the numpy representation format more future-proof, it can be an
    order of magnitude more compact.
    If the numpy_repr save is unsuccessful (possibly because 'data' does not provide a
    `numpy_repr` method), than save falls back to saving a plain (dill) pickle of 'data'.

    Parameters
    ----------
    file: str
        Path name or file object
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
        if len(format_names) > 1:
            format_names = ", ".join(format_names[:-1]) + " and " + format_names[-1]
        if len(bad_format_names) > 1:
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

    get_output = None
    def set_str_file(filename):
        def _get_output(filename, ext, bytes, overwrite):
            return output(filename, ext, bytes, overwrite)
        get_output = _get_output

    # Check argument - file
    if isinstance(file, io.IOBase):
        thisfilename = os.path.realpath(file.name)
        if 'luigi' in os.path.basename(thisfilename):
            # 'file' is actually a Luigi temporary file
            luigi = True
        else:
            luigi = False
        filename = thisfilename  # thisfilename used to avoid name clashes
        if not any(c in file.mode for c in ['w', 'x', 'a', '+']):
            logger.warning("File {} not open for writing; closing and reopening.")
            file.close()
            set_str_file(thisfilename)
        else:
            def _get_output(filename, ext, bytes, overwrite):
                # Check that the file object is compatible with the arguments,
                # and if succesful, just return the file object unmodified.
                # If it is not successful, revert to opening a file as though
                # a filename was passed to `save`.
                # TODO: Put checks in `dummy_file_context`
                fail = False
                if (os.path.splitext(os.path.realpath(filename))[0]
                    != os.path.splitext(os.path.realpath(thisfilename))[0]):
                    logger.warning("[iotools.save] Given filename and file object differ.")
                    fail = True
                thisext = os.path.splitext(thisfilename)[1].replace('.', '')
                if not luigi and thisext != ext.replace('.', ''):
                    # Luigi adds 'luigi' to extensions of temporary files; we
                    # don't want that to trigger closing the file
                    logger.warning("[iotools.save] File object has wrong extension.")
                    fail = True
                if (bytes and 'b' not in file.mode
                    or not bytes and 'b' in file.mode):
                    if luigi:
                        # Luigi's LocalTarget always saves to bytes, and it's
                        # the Format class that takes care of converting data
                        # (possibly text) to and back from bytes.
                        logger.warning("\n"
                            "WARNING [iotools]: Attempted to save a 'luigi' target with the wrong "
                            "mode (binary or text). Note that Luigi targets "
                            "always use the same mode internally; use the "
                            "`format` argument to convert to/from in your code. "
                            "In particular, LocalTarget writes in binary. "
                            "Consequently, the file will not be saved as {}, "
                            "but as {}; specify the correct value to `bytes` "
                            "to avoid this message.\n"
                            .format("bytes" if bytes else "text",
                                    "text" if bytes else "bytes"))
                    else:
                        logger.warning("[iotools.save] File object has incorrect byte mode.")
                        fail = True
                if (overwrite and 'a' in file.mode):
                    # Don't check for `not overwrite`: in that case the damage is already done
                    logger.warning("[iotools.save] File object unable to overwrite.")
                    fail = True
                if fail:
                    logger.warning("[iotools.save] Closing and reopening file object.")
                    file.close()
                    set_str_file(thisfilename)
                    return output(filename, ext, bytes, overwrite)
                else:
                    return dummy_file_context(file)
            get_output = _get_output
    else:
        assert(isinstance(file, str))
        filename = file
        set_str_file(file)

    # Ensure target directory exists
    dirname = os.path.dirname(filename)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    output_paths = []

    # If data provides a "save" method, use that
    # This overrides the "format" argument – only exception is if save fails,
    # then we reset it to what it was and try the other formats
    if isinstance(data, ParameterSet):
        # Special case of data with `save` attribute
        _selected_formats_back = selected_formats
        selected_formats = []  # Don't save to another format if successful
        with get_output(filename, ext="", bytes=False, overwrite=overwrite) as (f, output_path):
            # Close the file since Parameters only accepts urls as filenames
            # FIXME: This introduces a race condition; should use `f` to save
            #        This would require fixing the parameters package to
            #        accept file objects in `save()`
            pass
        try:
            logger.info("Saving ParameterSet using its own `save` method...")
            data.save(output_path, expand_urls=True)
        except (AttributeError, PermissionError) as e:
            logger.warning("Calling the data's `save` method failed with '{}'."
                           .format(str(e)))
            selected_formats = _selected_formats_back
        else:
            output_paths.append(output_path)
    elif hasattr(data, 'save'):
        _selected_formats_back = selected_formats
        selected_formats = []  # Don't save to another format if successful
        with get_output(filename, ext="", bytes=False, overwrite=overwrite) as (f, output_path):
            # TODO: Use `f` if possible, and only `output_path` if it fails.
            pass
        try:
            logger.info("Saving data using its own `save` method...")
            data.save(output_path)
        except (AttributeError, PermissionError) as e:
            logger.warning("Calling the data's `save` method failed with '{}'."
                           .format(str(e)))
            selected_formats = _selected_formats_back
        else:
            output_paths.append(output_path)

    # Save data as numpy representation
    if 'npr' in selected_formats:
        fail = False
        ext = defined_formats['npr'].ext
        try:
            with get_output(filename, ext, True, overwrite) as (f, output_path):
                try:
                    logger.info("Saving data to 'npr' format...")
                    np.savez(f, **data.repr_np)
                except AttributeError:
                    fail = True
                else:
                    output_paths.append(output_path)
        except IOError:
            fail = True
        if fail:
            # TODO: Use custom error type
            logger.warning("Unable to save to numpy representation ('npr') format.")
            if 'dill' not in selected_formats:
                # Warn the user that we will use another format
                logger.warning("Will try a plain (dill) pickle dump.")
                selected_formats.add('dill')

    # Save data as representation string
    for format in [format
                   for format in selected_formats
                   if format in ('repr', 'brepr')]:
        bytes = False if format == 'repr' else True
        fail = False
        if data.__repr__ is object.__repr__:
            # Non-informative repr -- abort
            fail = True
        else:
            ext = defined_formats['repr'].ext
            try:
                with get_output(filename, ext=ext, bytes=bytes, overwrite=overwrite) as (f, output_path):
                    try:
                        logger.info("Saving data to plain-text 'repr' format'")
                        f.write(repr(data))
                    except:
                        fail = True
                    else:
                        output_paths.append(output_path)
            except IOError:
                fail = True
        if fail:
            logger.warning("Unable to save to numpy representation ('npr') format.")
            if 'dill' not in selected_formats:
                # Warn the user that we will use another format
                logger.warning("Will try a plain (dill) pickle dump.")
                selected_formats.add('dill')

    # Save data in dill format
    if 'dill' in selected_formats:
        ext = defined_formats['dill'].ext
        try:
            with get_output(filename, ext, True, overwrite) as (f, output_path):
                logger.info("Saving data as a dill pickle.")
                dill.dump(data, f)
                output_paths.append(output_path)
        except IOError:
            pass # There might be other things to save, so don't terminate
                 # execution because this save failed

    # Return the list of output paths
    return output_paths


def load(filename, types=None, load_function=None, format=None, input_format=None):
    """
    Load file at `filename`. How the data is loaded is determined by the input format,
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

    Parameters
    ----------
    filename: str | file object (TODO)

    types: dict
        (Optional)
    load_function: function
        (Optional) Post-processing function, called on the result of the loaded
        data. I.e. does not override `type`, but provides a handle to process
        the result.
    format: str
        Specify the format of the data; overrides any deductions from the
        filename. Effectively this specifies the loading function, and thus
        should correspond to a key in `types`.
    input_format: str (DEPRECATED)
        Deprecated synonym for `format`.
    """
    global _load_types
    if types is not None:
        types = dict(types)
        types.update(_load_types)
    else:
        types = _load_types

    basepath, ext = os.path.splitext(filename)
    dirname, basename = os.path.split(basepath)
    if dirname == '':
        dirname = '.'

    if format is not None:
        input_format = format

    if len(ext) == 0 and input_format is None:
        #raise ValueError("Filename has no extension. Please specify input format.")
        # Try every file whose name without extension matches `filename`
        match = lambda fname: os.path.splitext(fname)[0] == basename
        fnames = [name for name in os.listdir(dirname) if match(name)]
        # Order the file names so we try most likely formats first (i.e. npr, repr, dill)
        # We do not attempt to load other extensions, since we don't know the format
        ordered_fnames = []
        for formatext in defined_formats:
            name = basename + '.' + formatext
            if name in fnames:
                ordered_fnames.append(name)
        # Try to load every file name in sequence. Terminate after the first success.
        for fname in ordered_fnames:
            try:
                data = load(os.path.join(dirname, fname),
                            types, load_function)
            except (FileNotFoundError):
                # Basically only possible to reach here with a race condition, where
                # file is deleted after having been listed
                # TODO: Also catch loading errors ?
                continue
            else:
                return data
        # No file was found
        raise FileNotFoundError("No file with base name '{}' was found."
                                .format(basename))
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
            raise

        return self.f, self.actual_path

    def __exit__(self, type, value, traceback):
        self.f.close()

class dummy_file_context:
    def __init__(self, file):
        self.f = file

    def __enter__(self):
        return self.f, os.path.realpath(self.f.name)

    def __exit__(self, type, value, traceback):
        # Since the file was not created in this context, don't close it
        pass
