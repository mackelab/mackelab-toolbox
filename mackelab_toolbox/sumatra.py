import datetime

def get_records(recordstore, project, label=None, script=None, before=None, after=None):
    """
    Return the records whose labels match `label`.
    The filters may be partial, i.e. the parameter sets of all records matching
    '*label*', '*script*',... are returned.
    """
    # TODO: Use database backend so that not all records need to be loaded into memory just
    #       to filter them.
    if label is not None:
        # RecordStore has builtin functions for searching on labels
        # FIXME!!!!: Dependency on fsGIF
        lbl_gen = (fulllabel for fulllabel in recordstore.labels('fsGIF') if label in fulllabel)
        record_list = [recordstore.get(project, fulllabel) for fulllabel in lbl_gen]
    else:
        record_list = recordstore.list(project)

    if script is not None:
        record_list = [record for record in record_list if script in record.main_file]

    if before is not None:
        if isinstance(before, tuple):
            before = datetime.datetime(*before)
        if not isinstance(before, datetime.datetime):
            tnorm = lambda tstamp: tstamp.date()
        else:
            tnorm = lambda tstamp: tstamp
        record_list = [rec for rec in record_list if tnorm(rec.timestamp) < before]
    if after is not None:
        if isinstance(after, tuple):
            after = datetime.datetime(*after)
        if not isinstance(after, datetime.datetime):
            tnorm = lambda tstamp: tstamp.date()
        else:
            tnorm = lambda tstamp: tstamp
        record_list = [rec for rec in record_list if tnorm(rec.timestamp) >= after]

    return record_list
    #return [(record_listt, fulllabel) for fulllabel in lbl_gen]
