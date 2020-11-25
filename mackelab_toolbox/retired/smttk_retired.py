try:
    import click
    click_loaded = True
except ImportError:
    click_loaded = False
    
##################################
#
# Command line interface
#
##################################

if click_loaded:
    def get_free_file(path, max_files=100):
        return iotools.get_free_file(path, bytes=True, max_files=100)

    def rename_to_free_file(path):
        new_f, new_path = get_free_file(path)
        new_f.close()
        os.rename(path, new_path)
        return new_path

    @click.group()
    def cli():
        pass

    class MoveList:
        """
        Helper class for 'refile' and 'addlinks'. Each path is associated
        to a timestamp; if the same target path is added multiple times, only the one with the most recent timestamp is kept.
        """
        def __init__(self):
            self.moves = {}

        def __len__(self):
            return len(self.moves)

        def __iter__(self):
            return ({'new path': key, 'old path': val['old path'],
                    'timestamp': val['timestamp']}
                    for key, val in self.moves.items())

        def add(self, src_path, target_path, timestamp):
            if ( target_path not in self.moves
                or timestamp > self.moves[target_path]['timestamp']):
                self.moves[target_path] = {'old path': src_path,
                                           'timestamp': timestamp}

    @click.command()
    @click.option('--datadir', default="data")
    @click.option('--dumpdir', default="run_dump")
    @click.option('--link/--no-link', default=True)
    @click.option('--recent/--current', default=False)
    def refile(datadir, dumpdir, link, recent):
        """
        DEPRECATED: Use addlinks instead. It never touches the original files,
        so there's no risk of data loss.

        Walk through the data directory and move files under timestamp directories
        (generated by Sumatra to differentiate calculations) to the matching
        'label-free' directory.
        E.g. if a file has the path 'data/run_dump/20170908-120245/inputs/generated_data.dat',
        it is moved to 'data/inputs/generated_data.dat'. In general,
        '[datadir]/[dumpdir]/[timestamp]/[dir1]/.../[dirn]/[filename]' -> '[datadir]/[dir1]/.../[dirn]/[filename]'

        By default, links are created from the old location to the new one. This can be
        disabled by passing the '--no-link' option.

        TODO: Get datadir from Sumatra
        TODO: Something sane when the target file already exists
        """

        lbl_pattern = '[0-9]{8}-[0-9]{6}(_[0-9]+)?'

        move_list = MoveList()
        for dirname in os.listdir(os.path.join(datadir, dumpdir)):
            if re.fullmatch(lbl_pattern, dirname) is None:
                logger.warning("Directory {} does not match the label pattern. Skipping.")
            else:
                # This is a label directory
                path = os.path.join(datadir, dumpdir, dirname)
                ## Loop over every file it contains
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:

                        ## Assemble the current path
                        old_path = os.path.join(dirpath, filename)
                        # if os.path.islink(old_path):
                        #     # Don't move symbolic links - they point to files that have already been moved
                        #     continue

                        ## Store the new (refiled) path for this file
                        split_path = old_path.split('/')
                        if split_path[0] == '':
                            # Fix for absolute paths
                            split_path[0] = '/'

                        if dumpdir == "":
                            # Since the label directory is immediately after
                            # datadir, we can get its index by seeing how many
                            # directories deep datadir is.
                            lbl_idx = len(datadir.split('/'))
                            dump_idx = None
                        else:
                            # There's a dump directory to remove as well
                            dump_idx = len(datadir.split('/'))
                            lbl_idx = dump_idx + len(dumpdir.split('/'))
                        assert(re.fullmatch(lbl_pattern, split_path[lbl_idx]))
                            # Ensure that we really are removing a timestamp directory
                        label = split_path[lbl_idx]
                        del split_path[lbl_idx]
                        if dump_idx is not None:
                            del split_path[dump_idx:lbl_idx]
                        new_path = os.path.join(*split_path)

                        ## Move the filename and create the link
                        if os.path.exists(new_path):
                            if not recent:
                                # Keep the current file version
                                print("File '{}' already exists. It was left in the labeled directory '{}'."
                                      .format(new_path, label))
                            else:
                                move_list.add(old_path, new_path, label)

                        else:
                            move_list.add(old_path, new_path, label)

        for move in move_list:
            if not os.path.islink(move['old path']):
                # Skip over links: they have already been refiled
                if os.path.exists(move['new path']):
                    renamed_path = rename_to_free_file(move['new path'])
                    print("Previous file '{}' was renamed to '{}'"
                        .format(move['new path'], renamed_path))
                else:
                    # Make sure the directory hierarchy exists
                    os.makedirs(os.path.dirname(move['new path']), exist_ok=True)

                os.rename(move['old path'], move['new path'])
                print("Refiled '{}' to the common directory."
                      .format(move['old path']))
                if link:
                    rel_new_path = os.path.relpath(move['new path'],
                                                   os.path.dirname(move['old path']))
                    os.symlink(rel_new_path, move['old path'])

    cli.add_command(refile)

    @click.command()
    @click.option('--datadir', default="data")
    @click.option('--dumpdir', default="run_dump")
    def addlinks(datadir, dumpdir):
        """
        Walk through the data directory and create links pointing to files under timestamp directories
        (generated by Sumatra to differentiate calculations) from the matching
        'label-free' directory.
        E.g. if a file has the path 'data/run_dump/20170908-120245/inputs/generated_data.dat',
        it is moved to 'data/inputs/generated_data.dat'. In general,
        '[datadir]/[dumpdir]/[timestamp]/[dir1]/.../[dirn]/[filename]' -> '[datadir]/[dir1]/.../[dirn]/[filename]'
        Currently only two formats are recognized as timestamp directories:
        ########-######       (Default Sumatra label)
        ########-######_#...  (Default Sumatra label with arbitrary length numeric suffix)
        so this works best if you leave the Sumatra label configuration to its default value.
        If you need to further differentiate between labels (e.g. for multiple simultaneously
        launched calculations), you can add a numeric suffix.

        TODO: Get datadir and dumpdir from Sumatra
        TODO: Something sane when the target file already exists
        """

        lbl_pattern = '[0-9]{8}-[0-9]{6}(_[0-9]+)?'

        move_list = MoveList()
        for dirname in os.listdir(os.path.join(datadir, dumpdir)):
            if re.fullmatch(lbl_pattern, dirname) is None:
                logger.warning("Directory {} does not match the label pattern. Skipping.")
            else:
                # This is a label directory
                path = os.path.join(datadir, dumpdir, dirname)
                ## Loop over every file it contains
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:

                        ## Assemble the current path
                        old_path = os.path.join(dirpath, filename)

                        ## Store the new (refiled) path for this file
                        split_path = old_path.split('/')
                        if split_path[0] == '':
                            # Fix for absolute paths
                            split_path[0] = '/'

                        if dumpdir == "":
                            # Since the label directory is immediately after
                            # datadir, we can get its index by seeing how many
                            # directories deep datadir is.
                            lbl_idx = len(datadir.split('/'))
                            dump_idx = None
                        else:
                            # There's a dump directory to remove as well
                            dump_idx = len(datadir.split('/'))
                            lbl_idx = dump_idx + len(dumpdir.split('/'))
                        assert(re.fullmatch(lbl_pattern, split_path[lbl_idx]))
                            # Ensure that we really are removing a timestamp directory
                        label = split_path[lbl_idx]
                        del split_path[lbl_idx]
                        if dump_idx is not None:
                            del split_path[dump_idx:lbl_idx]
                        new_path = os.path.join(*split_path)

                        ## Add to the list of moves
                        move_list.add(old_path, new_path, label)
                            # move_list ensures we only keep the most recent move

        create_links(move_list)

    def create_links(move_list, verbose=True):
        """
        Iterate through a MoveList representing a set of links and create them.
        If a file already exists where we want to place a link, we do the
        following:
          - If it's a link that already points to the right location, do nothing
          - If it's a link that points to another location, replace it
          - If it's an actual file, append a number to its filename before
            creating the link.
        Set `verbose` to `False` to prevent printing output for every link
        created or modified.
        """
        for move in move_list:
            # if os.path.islink(move['old path']):
            #     # Skip over links: they have already been refiled
            #     continue
            if os.path.islink(move['new path']):
                if os.path.realpath(move['new path']) == os.path.realpath(move['old path']):
                    # Present link is the same we want to create; don't do anything
                    continue
                else:
                    # Just delete the old link, since data is preserved in the dump folder
                    # assert(not os.path.islink(move['old path']))
                    os.remove(move['new path'])
                    if verbose:
                        print("Removed previous link to file '{}'"
                                .format(move['old path']))
            if os.path.exists(move['new path']):
                assert(not os.path.islink(move['new path']))
                # Rename the path so as to not lose data
                renamed_path = rename_to_free_file(move['new path'])
                if verbose:
                    print("Previous file '{}' was renamed to '{}'"
                            .format(move['new path'], renamed_path))
            else:
                # Make sure the directory hierarchy exists
                os.makedirs(os.path.dirname(move['new path']), exist_ok=True)

            rel_old_path = os.path.relpath(move['old path'],
                                            os.path.dirname(move['new path']))
            os.symlink(rel_old_path, move['new path'])
            if verbose:
                print("Link to '{}' in the common directory."
                        .format(move['old path']))

    cli.add_command(addlinks)

    @click.command()
    @click.argument("param_file")
    @click.option("--subdir", default="")
    @click.option("--suffix", default="")
    def file_exists(param_file, subdir, suffix):
        # FIXME: Use Sumatra function to grab context, if this function
        # is even still relevant
        mgr = core.RunMgr()
        if subdir == "":
            subdir = None
        if suffix == "":
            suffix = None

        searchname = mgr.get_pathname(ParameterSet(param_file), suffix, subdir)
        pathnames = glob.glob(searchname + "*")
        if len(pathnames) > 0:
            print("The following matching files were found:")
            for path in pathnames:
                print(path)
        else:
            print("No file matching '{}' was found.".format(pathname))

    cli.add_command(file_exists)


    ###########################
    # Launcher
    ###########################

    tmp_dir = "tmp"

    # TODO: Use click file arguments
    @click.command()
    @click.option("--dry-run/--run", default=False,
                  help="Use --dry to skip the computation, and just print "
                  "the command(s) that would be executed. The expanded parameter "
                  "files are left in the temporary directory to allow inspection.")
    @click.option("-n", "--cores", default=1)
    @click.option("-m", "--script", nargs=1, prompt=True,
                  type=click.Path(exists=True, dir_okay=False))
    @click.option("-r", "--reason", default="")
    @click.option("--max-tasks", default=1000)
    @click.argument("args", nargs=-1)
    @click.argument("params", nargs=1)
    def run(dry_run, cores, script, reason, max_tasks, args, params):
        basename, _ = os.path.splitext(os.path.basename(script))
        # TODO: Get extension from parameter path
        tmpparam_path = os.path.join(tmp_dir, basename + ".ntparameterset")

        # Scan `args` to see if it contains indicator for multiple parametersets
        pflag = "params:"
        if pflag in args:
            i = args.index(pflag)
            if pflag in args[i+1:]:
                raise ArgumentError("You can only pass a single 'params:' "
                                    "indicator.")
            params = args[i+1:] + (params,)
            args = args[:i]
        else:
            params = (params,)

        # FIXME: Parameter expansion does not work with nested files
        param_paths = itertools.chain.from_iterable(
            mtb.parameters.expand_param_file(
                paramfile, tmpparam_path, max_files=max_tasks)
            for paramfile in params)

        # Sumatra's default is to use a timestamp which is only precise up to
        # seconds. Thus jobs launched simultaneously would have the
        # same label. To avoid this, we generate our own label by appending a
        # run-specific number to the default time stamp label

        # Generate a timestamp label same as Sumatra's default
        timestamp = datetime.now()
        label = str(timestamp.strftime(sumatra.core.TIMESTAMP_FORMAT))
            # Same function as used in sumatra.records.Record

        shared_options = ""
        if reason != "":
            shared_options += " --reason " + reason

        argv_list = [ "-m {} {} --label {}_{} {} {}"
                      .format(script, shared_options, label, i, " ".join(args), param_file)
                      for i, param_file in enumerate(param_paths, start=1)]

        # for i, param_file in enumerate(param_paths, start=1):
        #     with open(param_file) as f:
        #         s = f.read()
        #         s2 = s.encode('utf8', 'replace').decode('utf8')
        #         for j, (c, c2) in enumerate(zip(s, s2)):
        #             if c != c2:
        #                 import pdb; pdb.set_trace()
        #                 print(j, c, c2)
        #                 return
        # print("No problems found")
        # return

        # Process idx array. Used to assign a unique index to each concurrently
        # running process
        # 'b' => signed char (1 byte)
        assert('process_idcs' not in globals())
        global process_idcs
        process_idcs = multiprocessing.Array('b', [0] * cores)

        if dry_run:
            # Dry-run
            print("With these arguments, the following calls would "
                  "distributed between {} processe{}:"
                  .format(cores, '' if cores == 1 else 's'))
            for argv in argv_list:
                print("smt run " + argv)
        else:
            if cores == 1:
                # Don't use multiprocessing. This is especially useful for debugging,
                # as execution is kept within this process
                for argv in argv_list:
                    _smtrun(argv)
            else:
                with multiprocessing.Pool(cores) as pool:
                    pool.map(_smtrun, argv_list)

    def _smtrun(argv_str):
        # Find first unused process index
        if 'process_idcs' in globals():
            global process_idcs
            with process_idcs.get_lock():
                if False:#len(process_idcs) == 1:
                    pidx = None
                else:
                    found = False
                    for i, v in enumerate(process_idcs):
                        if v == 0:
                            process_idcs[i] = 1
                            pidx = i
                            found = True
                            break
                    if not found:
                        # This should never happen, but just in case
                        # do something reasonable
                        pidx = len(process_idcs)
                        print("Unable to find a free process index. Assigning "
                            "index {}; it may be shared.".format(pidx))

        if pidx is not None:
            argv_str = '--threadidx {} '.format(pidx) + argv_str
        res = sumatra.commands.run(argv_str.split())

        # Free the process index
        if pidx is not None:
            with process_idcs.get_lock():
                process_idcs[pidx] = 0

        return res

    cli.add_command(run)

def get_free_file(path, max_files=100):
    """
    Return a file handle to an unused filename. If 'path' is free, return a handle
    to that. Otherwise, append a number to it until a free filename is found or the
    number exceeds 'max_files'. In the latter case, raise 'IOError'.

    Return a file handle, rather than just a file name, avoids the possibility of a
    race condition (a new file of the same name could be created between the time
    where one finds a free filename and then opens the file).

    Parameters
    ----------
    path: str
        Path name. Can be absolute or relative to the current directory.
    max_files: int
        (Optional) Maximum allowed number of files with the same name. If this
        number is exceeded, IOError is raised.

    Returns
    -------
    filehandle
        Filehandle, as obtained from a call to `open(pathname)`.
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

    # Make sure the directory exists
    os.makedirs(os.path.dirname(pathname), exist_ok=True)

    try:
        f = open(pathname, mode='xb')
        return f, pathname
    except IOError:
        name, ext = os.path.splitext(pathname)
        for i in range(2, max_files+2):
            appendedname = name + "_" + str(i) + ext
            try:
                f = open(appendedname, mode='xb')
                return f, appendedname
            except IOError:
                continue

        raise IOError("Number of files with the name '{}' has exceeded limit."
                      .format(path))

def rename_to_free_file(path):
    new_f, new_path = get_free_file(path)
    new_f.close()
    os.rename(path, new_path)
    return new_path


if __name__ == "__main__":
    if not noclick:
        cli()
