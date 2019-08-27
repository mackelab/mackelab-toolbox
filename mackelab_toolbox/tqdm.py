# https://github.com/tqdm/tqdm#redirecting-writing

import sys
import logging
from time import sleep
import contextlib
from tqdm import tqdm

class LoggingStreamHandler(logging.StreamHandler):
    """
    tqdm-aware susbstitute for logging's standard stream handler.
    The default handler writes to stderr, which conflicts with tqdm; this
    handler wraps that call with tqdm's `write()` to avoid conflicts.

    Source: https://github.com/tqdm/tqdm/issues/193#issuecomment-233212170
    """
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)

class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err
