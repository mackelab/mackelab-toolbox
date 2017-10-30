# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:56:37 2017

@author: alex
"""

import os.path

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
        (Optional) Specify whether to open the file for byte (True) or
        plain text (False) output. Default is to open for byte output, which
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
