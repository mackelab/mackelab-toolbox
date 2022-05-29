"""
Interface to the `luigi` package simplifying it's use for scientific computing.
Should be merged into the `mackelab` package at some point.
"""

import os.path
from collections import Iterable
import psutil
import luigi
from . import parameters, iotools

default_workers = psutil.cpu_count(logical=False) - 1
    # Defaults to no. of cores-1; `logical=False` ignores hyperthreading

class Task(luigi.Task):
    data_dir = "data"
    task_dir = None
    format = 'npr'  # Format must be one in `iotools.defined_formats`

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._debug = False

    def output(self):
        return luigi.local_target.LocalTarget(self.save_name(),
                                              format=luigi.format.Nop)

    def save_name(self):
        if self.task_dir is not None:
            assert self.task_dir[0] != '/'  # Would overwrite `data_dir`
            if self.data_dir in self.task_dir:
                # `data_dir` was repeated – don't use it twice
                dir = self.task_dir
            else:
                dir = os.path.join(self.data_dir, self.task_dir)
        else:
            dir = self.data_dir
        filename = type(self).__name__ + "-" \
            + parameters.digest(self.to_str_params()) \
            + "." + iotools.defined_formats[self.format].ext
        if self._debug:
            return 'debug-' + filename
        else:
            return os.path.join(dir, filename)

    def execute(self, local_scheduler=None):
        execute(self, local_scheduler)

    def debug(self):
        """
        Output redirected to current dir, to avoid polluting data dir.
        Sets the `self._debug` flag to `True`, which can be used within `run()`.
        """
        self._debug = True
        self.run()
        self._debug = False

    def load(self):
        return iotools.load(self.save_name(), format=self.format)

def execute(tasks, local_scheduler=None, **kwargs):
    if local_scheduler is None:
        if luigi_server_is_running():
            local_scheduler = False
        else:
            local_scheduler = True
    if isinstance(tasks, luigi.Task):
        tasks = [tasks]
    default_kwargs = {'workers': default_workers}
    kwargs = {**default_kwargs, **kwargs}  # Overwrites defaults
    luigi.build(tasks, local_scheduler=local_scheduler, **kwargs)

# ================================
# Specialized parameter types
# ================================

# Specialized types that don't depend on other packages would be defined here
# Those which depend on a package should be defined within that package,
# such as what is done in sinn.models.luigi

# ===============================
# Utilities
# ===============================

# HACK: This is an awful way of checking whether a luigi server is running,
# not least because it relies on the default URL.
def luigi_server_is_running():
    import http.client

    try:
        a = http.client.HTTPConnection('localhost:8082')
        a.connect()
    except ConnectionRefusedError:
        return False
    else:
        return True
